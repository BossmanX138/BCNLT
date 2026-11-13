"""Audit packed task IDs for accidental reserved-token injection."""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_TOKENIZERS_SITE = PROJECT_ROOT / "vendor" / "tokenizers_runtime_0_23"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PINNED_TOKENIZERS_SITE))

from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402
from bosscrafts_model.compression import decompress_bounded  # noqa: E402


def _decode_u16(blob: bytes, *, token_count: int) -> list[int]:
    values = array("H")
    values.frombytes(
        decompress_bounded(blob, maximum_output_bytes=token_count * 2)
    )
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def audit(database: Path) -> dict[str, object]:
    binding = NativeTokenizerBinding.load()
    control = binding.structural_token_ids
    structural = frozenset(control.values())
    standard_special = frozenset(range(4))
    expected_instruction = Counter(
        {
            control["<|bcnlt:task:mixed_instruction|>"]: 1,
            control["<|bcnlt:stream:begin|>"]: 2,
            control["<|bcnlt:mode:raw_code|>"]: 2,
            control["<|bcnlt:stream:end|>"]: 2,
            control["<|bcnlt:task:end|>"]: 1,
        }
    )
    special_rows: Counter[tuple[str, int]] = Counter()
    special_occurrences: Counter[tuple[str, int]] = Counter()
    instruction_protocol_mismatch = 0
    examples: dict[str, list[str]] = defaultdict(list)
    row_counts: Counter[str] = Counter()
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        rows = connection.execute(
            "SELECT t.example_id,s.source_kind,t.input_ids_zlib,t.token_count "
            "FROM tasks AS t JOIN source_example_identity AS s USING(example_id)"
        )
        for example_id, source_kind, blob, token_count in rows:
            kind = str(source_kind)
            row_counts[kind] += 1
            expected_count = int(token_count)
            ids = _decode_u16(blob, token_count=expected_count)
            if len(ids) != expected_count:
                raise ValueError("task token count mismatch")
            observed_special = Counter(token_id for token_id in ids if token_id in standard_special)
            for token_id, count in observed_special.items():
                special_rows[(kind, token_id)] += 1
                special_occurrences[(kind, token_id)] += count
                key = f"{kind}:{token_id}"
                if len(examples[key]) < 5:
                    examples[key].append(str(example_id))
            if kind == "instruction":
                observed_structural = Counter(token_id for token_id in ids if token_id in structural)
                if observed_structural != expected_instruction:
                    instruction_protocol_mismatch += 1
                    if len(examples["instruction_protocol_mismatch"]) < 5:
                        examples["instruction_protocol_mismatch"].append(str(example_id))
    finally:
        connection.close()
    return {
        "schema_version": "bosscrafts-task-control-id-audit-v1",
        "database": str(database.resolve()),
        "row_counts": dict(sorted(row_counts.items())),
        "standard_special_rows": {
            f"{kind}:{token_id}": count
            for (kind, token_id), count in sorted(special_rows.items())
        },
        "standard_special_occurrences": {
            f"{kind}:{token_id}": count
            for (kind, token_id), count in sorted(special_occurrences.items())
        },
        "instruction_protocol_mismatch_rows": instruction_protocol_mismatch,
        "example_ids": dict(sorted(examples.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.database)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
