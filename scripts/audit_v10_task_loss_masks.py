"""Materialize the no-training loss-mask receipt for frozen V10 tasks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.task_loss_mask_audit import TaskLossMaskAuditor
from bosscrafts_model.tokenizer import NativeTokenizerBinding


DEFAULT_TRAIN = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-audited-training-tasks-v1"
    / "tasks.sqlite"
)
DEFAULT_VALIDATION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-frozen-validation-tasks-v1"
    / "tasks.sqlite"
)
DEFAULT_TEST = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-frozen-test-tasks-v1"
    / "tasks.sqlite"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "bcnlt-v10-task-loss-mask-audit-v1"
    / "receipt.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = TaskLossMaskAuditor(NativeTokenizerBinding.load()).materialize(
        {
            "train": arguments.train,
            "validation": arguments.validation,
            "test": arguments.test,
        },
        arguments.output,
    )
    print(
        json.dumps(
            {**asdict(result), "output_path": str(result.output_path)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
