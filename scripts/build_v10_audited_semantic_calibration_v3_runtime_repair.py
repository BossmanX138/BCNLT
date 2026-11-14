"""Build repaired-runtime uncertainty data from frozen validation/test audits."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.audited_semantic_calibration_dataset import (  # noqa: E402
    AuditedSemanticCalibrationDatasetBuilder,
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "bcnlt-v10-audited-semantic-uncertainty-v3-runtime-repair"
)


def _admission(split: str) -> Path:
    return (
        PROJECT_ROOT
        / "artifacts"
        / "corpora"
        / f"bcnlt-v10-frozen-{split}-admission-v3-runtime-repair"
        / "admission.sqlite"
    )


def _audit(split: str) -> Path:
    return (
        PROJECT_ROOT
        / "artifacts"
        / "audits"
        / f"bcnlt-v10-frozen-{split}-semantic-audit-v3-runtime-repair"
        / "audit.sqlite"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-admission", type=Path, default=_admission("validation")
    )
    parser.add_argument("--validation-audit", type=Path, default=_audit("validation"))
    parser.add_argument("--test-admission", type=Path, default=_admission("test"))
    parser.add_argument("--test-audit", type=Path, default=_audit("test"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-language", type=int, default=10)
    arguments = parser.parse_args()
    result = AuditedSemanticCalibrationDatasetBuilder(
        samples_per_language=arguments.samples_per_language
    ).materialize(
        arguments.validation_admission,
        arguments.validation_audit,
        arguments.test_admission,
        arguments.test_audit,
        arguments.output,
    )
    print(
        json.dumps(
            {**asdict(result), "output_root": str(result.output_root)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.calibration_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
