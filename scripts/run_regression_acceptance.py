"""Run the fixed full regression and publish its immutable receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.model_category_evidence import checkpoint_sha256
from bosscrafts_model.regression_acceptance import (
    build_regression_receipt,
    publish_regression_receipt,
    run_regression_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_regression_suite(
        arguments.python_executable,
        timeout_seconds=arguments.timeout_seconds,
    )
    receipt = build_regression_receipt(
        result,
        checkpoint_sha256=checkpoint_sha256(arguments.checkpoint),
        checkpoint_step=arguments.checkpoint_step,
        example_id=arguments.example_id,
    )
    publish_regression_receipt(arguments.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
