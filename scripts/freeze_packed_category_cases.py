"""Freeze the three packed acceptance categories from an audited test database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.packed_category_case_sources import (  # noqa: E402
    freeze_packed_category_cases,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-database", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.output_root.resolve()
    outputs = {
        category: root / f"{category}.jsonl"
        for category in (
            "tokenizer_fidelity",
            "semantic_fidelity",
            "raw_code",
        )
    }
    result = freeze_packed_category_cases(
        NativeTokenizerBinding.load(),
        arguments.task_database,
        outputs,
        root / "audit.json",
    )
    print(
        json.dumps(
            {
                "outputs": {
                    category: str(path) for category, path in result.outputs.items()
                },
                "audit_output": str(result.audit_output),
                "task_database_identity_sha256": result.task_database_identity_sha256,
                "record_counts": result.record_counts,
                "audit_sha256": result.audit_sha256,
                "training_ready": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
