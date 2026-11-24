"""Bootstrap a current task database from proof-reusable historical rows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.task_runtime_reuse import (  # noqa: E402
    migrate_reusable_tasks,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-task-database", type=Path, required=True)
    parser.add_argument("--baseline-task-audit-manifest", type=Path, required=True)
    parser.add_argument("--baseline-admission-database", type=Path, required=True)
    parser.add_argument("--candidate-admission-database", type=Path, required=True)
    parser.add_argument("--admission-delta-receipt", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--candidate-eligibility-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-eligibility-manifest-sha256",
        type=_lower_sha256,
        required=True,
    )
    parser.add_argument("--target-database", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--max-sequence", type=int, default=2048)
    parser.add_argument(
        "--semantic-wire", choices=("compact_v2", "full_v3"), default="compact_v2"
    )
    parser.add_argument(
        "--forced-split", choices=("train", "validation", "test")
    )
    parser.add_argument("--build-cell-index", action="store_true")
    arguments = parser.parse_args()

    result = migrate_reusable_tasks(
        NativeTokenizerBinding.load(),
        baseline_task_database=arguments.baseline_task_database,
        baseline_task_audit_manifest=arguments.baseline_task_audit_manifest,
        baseline_admission_database=arguments.baseline_admission_database,
        candidate_admission_database=arguments.candidate_admission_database,
        admission_delta_receipt=arguments.admission_delta_receipt,
        source_catalog=arguments.source_catalog,
        candidate_eligibility_root=arguments.candidate_eligibility_root,
        candidate_eligibility_manifest_sha256=(
            arguments.candidate_eligibility_manifest_sha256
        ),
        target_database=arguments.target_database,
        output_receipt=arguments.output_receipt,
        max_sequence=arguments.max_sequence,
        semantic_wire=arguments.semantic_wire,
        defer_cell_index=not arguments.build_cell_index,
        forced_split=arguments.forced_split,
    )
    payload = asdict(result)
    payload["target_database"] = str(result.target_database)
    payload["training_ready"] = False
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
