"""Freeze source-controlled regression and resource acceptance cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.gate_category_case_sources import (  # noqa: E402
    freeze_gate_category_cases,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.output_root.resolve()
    result = freeze_gate_category_cases(
        arguments.policy,
        root / "regression.jsonl",
        root / "resources.jsonl",
        revision=arguments.revision,
        expected_source_sha256=arguments.source_sha256,
    )
    print(
        json.dumps(
            {
                "regression_output": str(result.regression_output),
                "resources_output": str(result.resources_output),
                "policy_sha256": result.policy_sha256,
                "revision": result.revision,
                "regression_example_id": result.regression_example_id,
                "resources_example_id": result.resources_example_id,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
