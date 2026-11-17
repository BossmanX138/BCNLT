"""Materialize the no-training loss-mask receipt for repaired V10 tasks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.task_loss_mask_audit import TaskLossMaskAuditor  # noqa: E402
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402
from scripts.pack_v10_audited_training_tasks_v3_runtime_repair import (  # noqa: E402
    DEFAULT_DATABASE as DEFAULT_TRAIN,
)

DEFAULT_VALIDATION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-frozen-validation-tasks-v3-runtime-repair"
    / "tasks.sqlite"
)
DEFAULT_TEST = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-frozen-test-tasks-v3-runtime-repair"
    / "tasks.sqlite"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "bcnlt-v10-task-loss-mask-audit-v3-runtime-repair"
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
