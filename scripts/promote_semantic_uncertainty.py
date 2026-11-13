"""Run the frozen V10 semantic uncertainty promotion policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = (
    PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate" / "source"
)
for path in (PROJECT_ROOT, RUNTIME_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bosscrafts_model.semantic_calibration_promotion import (
    CalibrationThresholds,
    SemanticCalibrationPromotionPolicy,
    promote_semantic_uncertainty,
)
from bosscrafts_model.strict_uncertainty_acceptance import StrictUncertaintyPolicy

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "bcnlt-v10-semantic-uncertainty-v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "bcnlt-v10-semantic-uncertainty-promotion-v1"
)


def frozen_policy() -> SemanticCalibrationPromotionPolicy:
    return SemanticCalibrationPromotionPolicy(
        contained_thresholds=CalibrationThresholds(
            minimum_heldout_samples=140,
            minimum_languages=14,
            minimum_samples_per_language=10,
            maximum_calibrated_ece=0.15,
            maximum_calibrated_brier=0.25,
            maximum_brier_degradation=0.02,
        ),
        strict_policy=StrictUncertaintyPolicy(
            minimum_global_selective_coverage_at_025=0.30,
            maximum_global_selective_error_rate_at_025=0.10,
            minimum_per_language_selective_coverage_at_025=0.10,
            maximum_per_language_selective_error_rate_at_025=0.25,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print-policy", action="store_true")
    arguments = parser.parse_args()
    policy = frozen_policy()
    if arguments.print_policy:
        print(json.dumps(policy.to_dict(), indent=2, sort_keys=True))
        return 0
    result = promote_semantic_uncertainty(
        arguments.dataset,
        arguments.output,
        policy=policy,
    )
    print(
        json.dumps(
            {
                "calibration_artifact_sha256": result.calibration_artifact_sha256,
                "failure_codes": result.failure_codes,
                "manifest_sha256": result.manifest_sha256,
                "output_root": str(result.output_root),
                "promoted": result.promoted,
                "strict_receipt_sha256": result.strict_receipt_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
