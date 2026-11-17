"""Audit contradictory code-to-semantic prompts in a packed task database."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PINNED_TOKENIZERS_SITE = PROJECT_ROOT / "vendor" / "tokenizers_runtime_0_23"
if not (PINNED_TOKENIZERS_SITE / "tokenizers" / "tokenizers.pyd").is_file():
    raise RuntimeError("project-contained tokenizers 0.23 runtime is missing")
if "tokenizers" in sys.modules:
    raise RuntimeError("tokenizers was imported before the pinned runtime was selected")
sys.path.insert(0, str(PINNED_TOKENIZERS_SITE))

import tokenizers

if tokenizers.__version__ != "0.23.2":
    raise RuntimeError("project-contained tokenizers runtime version mismatch")

from bosscrafts_model.compression import decompress_bounded
from bosscrafts_model.resumable_task_packing import (
    _decode_u16,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-balanced-expanded-v2"
    / "tasks.sqlite"
)


class _CollisionRecord(TypedDict):
    example_id: str
    document_id: str
    language: str
    record_index: int
    split: str
    prompt: tuple[int, ...]
    fragment: tuple[int, ...]
    event_token_id: int
    source_sha256: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--maximum-sequence", type=int, default=2048)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.maximum_sequence < 64:
        parser.error("maximum-sequence must be at least 64")
    if not 1 <= arguments.sample_limit <= 100:
        parser.error("sample-limit must be between 1 and 100")

    binding = NativeTokenizerBinding.load()
    audit = audit_task_database(binding, arguments.database)
    ids = binding.structural_token_ids
    stream_begin = ids["<|bcnlt:stream:begin|>"]
    stream_end = ids["<|bcnlt:stream:end|>"]
    semantic_mode = ids["<|bcnlt:mode:semantic_code|>"]
    record_begin = ids["<|bcnlt:record:begin|>"]
    record_end = ids["<|bcnlt:record:end|>"]
    task_end = ids["<|bcnlt:task:end|>"]
    event_tokens = {
        token_id: token
        for token, token_id in ids.items()
        if token.startswith("<|bcnlt:event:")
    }

    database = arguments.database.resolve()
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    groups: dict[tuple[str, tuple[int, ...]], list[_CollisionRecord]] = defaultdict(list)
    try:
        rows = connection.execute(
            """SELECT example_id,document_id,language,record_index,split,
                      input_ids_zlib,token_count,loss_start_index,source_sha256
               FROM tasks WHERE family='code_to_semantic'
               ORDER BY document_id,record_index,example_id"""
        )
        for row in rows:
            token_count = int(row[6])
            input_ids = _decode_u16(
                decompress_bounded(
                    row[5], maximum_output_bytes=token_count * 2
                )
            )
            if len(input_ids) != token_count:
                raise ValueError("task token count mismatch")
            loss_start = int(row[7])
            prompt = tuple(input_ids[:loss_start])
            target = input_ids[loss_start:]
            if not (
                target[:3] == [stream_begin, semantic_mode, record_begin]
                and target[-3:] == [record_end, stream_end, task_end]
            ):
                raise ValueError("code-to-semantic target is not a single semantic record")
            fragment = tuple(target[2:-2])
            groups[(str(row[1]), prompt)].append(
                {
                    "example_id": str(row[0]),
                    "document_id": str(row[1]),
                    "language": str(row[2]),
                    "record_index": int(row[3]),
                    "split": str(row[4]),
                    "prompt": prompt,
                    "fragment": fragment,
                    "event_token_id": int(fragment[1]),
                    "source_sha256": str(row[8]),
                }
            )
    finally:
        connection.close()

    group_sizes: Counter[int] = Counter()
    grouped_counts: Counter[str] = Counter()
    original_counts: Counter[str] = Counter()
    conflicting_groups = 0
    conflicting_rows = 0
    over_limit_groups = 0
    maximum_grouped_sequence = 0
    samples: list[dict[str, object]] = []
    for (_, prompt), records in sorted(
        groups.items(), key=lambda item: (item[1][0]["split"], item[1][0]["language"], item[0])
    ):
        records.sort(key=lambda item: (item["record_index"], item["example_id"]))
        languages = {str(item["language"]) for item in records}
        splits = {str(item["split"]) for item in records}
        source_hashes = {str(item["source_sha256"]) for item in records}
        if len(languages) != 1 or len(splits) != 1 or len(source_hashes) != 1:
            raise ValueError("prompt group lineage is inconsistent")
        language = next(iter(languages))
        split = next(iter(splits))
        cell = f"{split}|{language}"
        group_sizes[len(records)] += 1
        grouped_counts[cell] += 1
        original_counts[cell] += len(records)
        fragments = [tuple(item["fragment"]) for item in records]
        distinct_targets = len(set(fragments))
        if distinct_targets > 1:
            conflicting_groups += 1
            conflicting_rows += len(records)
        grouped_length = len(prompt) + 2 + sum(len(fragment) for fragment in fragments) + 2
        maximum_grouped_sequence = max(maximum_grouped_sequence, grouped_length)
        if grouped_length > arguments.maximum_sequence:
            over_limit_groups += 1
        if distinct_targets > 1 and len(samples) < arguments.sample_limit:
            samples.append(
                {
                    "document_id": records[0]["document_id"],
                    "language": language,
                    "split": split,
                    "prompt_text": binding.decode(list(prompt)),
                    "record_indices": [item["record_index"] for item in records],
                    "event_tokens": [
                        event_tokens.get(int(item["event_token_id"]), str(item["event_token_id"]))
                        for item in records
                    ],
                    "original_example_ids": [item["example_id"] for item in records],
                    "grouped_sequence_tokens": grouped_length,
                }
            )

    payload = {
        "schema_version": "bosscrafts-code-to-semantic-collision-audit-v1",
        "database": str(database),
        "task_database_identity_sha256": audit.database_identity_sha256,
        "original_example_count": sum(original_counts.values()),
        "unique_prompt_group_count": len(groups),
        "conflicting_prompt_group_count": conflicting_groups,
        "rows_in_conflicting_groups": conflicting_rows,
        "prompt_group_size_histogram": {
            str(size): count for size, count in sorted(group_sizes.items())
        },
        "maximum_grouped_sequence_tokens": maximum_grouped_sequence,
        "grouped_sequences_over_limit": over_limit_groups,
        "maximum_sequence": arguments.maximum_sequence,
        "original_counts_by_split_language": dict(sorted(original_counts.items())),
        "grouped_counts_by_split_language": dict(sorted(grouped_counts.items())),
        "samples": samples,
        "audit_complete": True,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if arguments.output is not None:
        destination = arguments.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(destination)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
