"""Compare full and compact task packing on the signed held-out admission."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.compact_task_packing import CompactCurrentCorpusTaskPacker
from bosscrafts_model.current_task_packing import CurrentCorpusTaskPacker
from bosscrafts_model.tokenizer import NativeTokenizerBinding


DEFAULT_ADMISSION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v6-65536-heldout-admission-v2"
    / "admitted.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "audits" / "compact-task-packing-heldout-v6-final-v1.json"


def _key(example: dict[str, object]) -> tuple[str, str, int]:
    return (
        str(example["document_id"]),
        str(example["family"]),
        int(example["record_index"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequence", type=int, default=2048)
    arguments = parser.parse_args()

    binding = NativeTokenizerBinding.load()
    full = CurrentCorpusTaskPacker(binding, max_sequence=arguments.max_sequence)
    compact = CompactCurrentCorpusTaskPacker(
        binding,
        max_sequence=arguments.max_sequence,
    )
    admission_hash = hashlib.sha256()
    documents = 0
    full_examples: dict[tuple[str, str, int], int] = {}
    compact_examples: dict[tuple[str, str, int], int] = {}
    full_families: Counter[str] = Counter()
    compact_families: Counter[str] = Counter()
    full_rejections: Counter[str] = Counter()
    compact_rejections: Counter[str] = Counter()

    with arguments.admission.open("rb") as stream:
        for raw_line in stream:
            admission_hash.update(raw_line)
            row = json.loads(raw_line)
            documents += 1
            row_full, row_full_rejections = full.pack_row(row)
            row_compact, row_compact_rejections = compact.pack_row(row)
            for example in row_full:
                key = _key(example)
                if key in full_examples:
                    raise ValueError("full task logical identity is duplicated")
                tokens = example["input_ids"]
                if type(tokens) is not list:
                    raise ValueError("full task input IDs are invalid")
                full_examples[key] = len(tokens)
                full_families[str(example["family"])] += 1
            for example in row_compact:
                key = _key(example)
                if key in compact_examples:
                    raise ValueError("compact task logical identity is duplicated")
                tokens = example["input_ids"]
                if type(tokens) is not list or len(tokens) > arguments.max_sequence:
                    raise ValueError("compact task input IDs are invalid")
                compact_examples[key] = len(tokens)
                compact_families[str(example["family"])] += 1
            full_rejections.update(str(item["reason_code"]) for item in row_full_rejections)
            compact_rejections.update(
                str(item["reason_code"]) for item in row_compact_rejections
            )

    missing_from_compact = sorted(set(full_examples).difference(compact_examples))
    shared = set(full_examples).intersection(compact_examples)
    full_shared_tokens = sum(full_examples[key] for key in shared)
    compact_shared_tokens = sum(compact_examples[key] for key in shared)
    result = {
        "schema_version": "bosscrafts-compact-task-packing-comparison-v1",
        "status": "heldout_task_packing_validation",
        "admission_file": str(arguments.admission.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "admission_sha256": admission_hash.hexdigest(),
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "maximum_sequence_tokens": arguments.max_sequence,
        "documents": documents,
        "full": {
            "example_count": len(full_examples),
            "total_tokens": sum(full_examples.values()),
            "family_counts": dict(sorted(full_families.items())),
            "rejection_counts": dict(sorted(full_rejections.items())),
        },
        "compact": {
            "example_count": len(compact_examples),
            "total_tokens": sum(compact_examples.values()),
            "family_counts": dict(sorted(compact_families.items())),
            "rejection_counts": dict(sorted(compact_rejections.items())),
        },
        "shared_example_count": len(shared),
        "shared_full_tokens": full_shared_tokens,
        "shared_compact_tokens": compact_shared_tokens,
        "shared_token_reduction_percent": (
            (full_shared_tokens - compact_shared_tokens) * 100.0 / full_shared_tokens
        ),
        "additional_compact_examples": len(set(compact_examples).difference(full_examples)),
        "full_examples_missing_from_compact": len(missing_from_compact),
        "missing_examples": missing_from_compact[:20],
    }
    if missing_from_compact:
        raise RuntimeError("compact packing lost examples admitted by the full wire")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
