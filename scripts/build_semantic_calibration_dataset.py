"""Build disjoint semantic uncertainty splits from verified dual review."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.semantic_calibration_dataset import (
    SemanticCalibrationDatasetBuilder,
)

DEFAULT_QUEUE = (
    PROJECT_ROOT
    / "artifacts"
    / "reviews"
    / "bcnlt-v10-current-heldout-review-v2-context"
)
DEFAULT_DUAL_REVIEW = (
    PROJECT_ROOT
    / "artifacts"
    / "reviews"
    / "bcnlt-v10-current-heldout-dual-local-v6-jsonschema"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "bcnlt-v10-semantic-uncertainty-v1"
)


def _load_decisions(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"manual decisions cannot be read: {exc}") from exc
    decisions = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("manual decision JSON is invalid") from exc
        if type(value) is not dict:
            raise ValueError("manual decision must be an object")
        decisions.append(value)
    return tuple(decisions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--dual-review", type=Path, default=DEFAULT_DUAL_REVIEW)
    parser.add_argument("--manual-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-per-language", type=int, default=10)
    parser.add_argument("--heldout-per-language", type=int, default=10)
    arguments = parser.parse_args()
    result = SemanticCalibrationDatasetBuilder(
        calibration_per_language=arguments.calibration_per_language,
        heldout_per_language=arguments.heldout_per_language,
    ).materialize(
        arguments.queue,
        arguments.dual_review,
        _load_decisions(arguments.manual_decisions),
        arguments.output,
    )
    print(
        json.dumps(
            {
                "calibration_count": result.calibration_count,
                "heldout_count": result.heldout_count,
                "language_counts": result.language_counts,
                "manifest_sha256": result.manifest_sha256,
                "output_root": str(result.output_root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
