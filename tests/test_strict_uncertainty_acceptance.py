from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_import import select_bcnlt_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
select_bcnlt_runtime(
    PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate" / "source"
)

from bcnlt.uncertainty_calibration import (
    CalibrationThresholds,
    UncertaintyObservation,
    evaluate_uncertainty_calibration,
    fit_uncertainty_calibrator,
)


def _feature():
    if importlib.util.find_spec("bosscrafts_model.strict_uncertainty_acceptance") is None:
        pytest.fail("strict uncertainty acceptance feature is not implemented")
    return importlib.import_module("bosscrafts_model.strict_uncertainty_acceptance")


def _rows(prefix: str, *, wrong_low: bool = False, all_incorrect: bool = False):
    rows = []
    for language in ("python", "javascript"):
        for index in range(20):
            high = index >= 10
            incorrect = True if all_incorrect else high
            if wrong_low and not high:
                incorrect = True
            rows.append(
                UncertaintyObservation(
                    sample_id=f"{prefix}-{language}-{index}",
                    language=language,
                    raw_uncertainty=0.8 if high else 0.2,
                    incorrect=incorrect,
                )
            )
    return tuple(rows)


def _artifact(*, calibration=None, heldout=None):
    calibration = calibration or _rows("calibration")
    heldout = heldout or _rows("heldout")
    calibrator = fit_uncertainty_calibrator(
        calibration,
        parser_runtime_identity_sha256="a" * 64,
    )
    return evaluate_uncertainty_calibration(
        calibrator,
        heldout,
        thresholds=CalibrationThresholds(
            minimum_heldout_samples=40,
            minimum_languages=2,
            minimum_samples_per_language=20,
            maximum_calibrated_ece=1.0,
            maximum_calibrated_brier=1.0,
            maximum_brier_degradation=1.0,
        ),
        heldout_manifest_sha256="b" * 64,
    )


def _policy(feature):
    return feature.StrictUncertaintyPolicy(
        minimum_global_selective_coverage_at_025=0.4,
        maximum_global_selective_error_rate_at_025=0.1,
        minimum_per_language_selective_coverage_at_025=0.4,
        maximum_per_language_selective_error_rate_at_025=0.1,
    )


def test_strict_gate_accepts_calibrated_low_uncertainty_predictions():
    feature = _feature()
    result = feature.verify_strict_uncertainty_acceptance(
        _artifact(), _policy(feature)
    )
    assert result.passed is True
    assert result.failure_codes == ()
    assert result.receipt_sha256 == result.compute_sha256()


def test_strict_gate_rejects_zero_low_uncertainty_coverage():
    feature = _feature()
    artifact = _artifact(
        calibration=_rows("calibration", all_incorrect=True),
        heldout=_rows("heldout", all_incorrect=True),
    )
    result = feature.verify_strict_uncertainty_acceptance(
        artifact, _policy(feature)
    )
    assert result.passed is False
    assert "global_selective_coverage_below_threshold" in result.failure_codes
    assert "python_selective_coverage_below_threshold" in result.failure_codes


def test_strict_gate_rejects_wrong_low_uncertainty_predictions():
    feature = _feature()
    artifact = _artifact(heldout=_rows("heldout", wrong_low=True))
    result = feature.verify_strict_uncertainty_acceptance(
        artifact, _policy(feature)
    )
    assert result.passed is False
    assert "global_selective_error_rate_above_threshold" in result.failure_codes
    assert "javascript_selective_error_rate_above_threshold" in result.failure_codes


def test_strict_gate_rejects_an_upstream_failed_artifact():
    feature = _feature()
    calibration = _rows("calibration")
    calibrator = fit_uncertainty_calibrator(
        calibration,
        parser_runtime_identity_sha256="c" * 64,
    )
    artifact = evaluate_uncertainty_calibration(
        calibrator,
        _rows("heldout", wrong_low=True),
        thresholds=CalibrationThresholds(
            minimum_heldout_samples=40,
            minimum_languages=2,
            minimum_samples_per_language=20,
            maximum_calibrated_ece=0.01,
            maximum_calibrated_brier=0.01,
            maximum_brier_degradation=0.0,
        ),
        heldout_manifest_sha256="d" * 64,
    )
    result = feature.verify_strict_uncertainty_acceptance(
        artifact, _policy(feature)
    )
    assert result.passed is False
    assert "upstream_calibration_not_promoted" in result.failure_codes


def test_strict_gate_fails_closed_when_selective_coverage_is_missing():
    feature = _feature()
    metrics = SimpleNamespace(
        sample_count=40,
        brier=0.1,
        ece=0.1,
        selective_coverage_at_025=None,
        selective_error_rate_at_025=None,
    )

    class Artifact:
        artifact_sha256 = "e" * 64
        parser_runtime_identity_sha256 = "f" * 64
        calibrated_metrics = metrics
        per_language = {"python": metrics}
        promoted = True

        @staticmethod
        def validate():
            return None

    result = feature.verify_strict_uncertainty_acceptance(
        Artifact(), _policy(feature)
    )

    assert result.passed is False
    assert "global_selective_coverage_unavailable" in result.failure_codes
    assert "python_selective_coverage_unavailable" in result.failure_codes
