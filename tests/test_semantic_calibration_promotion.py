from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_import import select_bcnlt_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
select_bcnlt_runtime(
    PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate" / "source"
)

from bcnlt.uncertainty_calibration import CalibrationThresholds
from bosscrafts_model.strict_uncertainty_acceptance import StrictUncertaintyPolicy


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _feature():
    if importlib.util.find_spec("bosscrafts_model.semantic_calibration_promotion") is None:
        pytest.fail("semantic calibration promotion feature is not implemented")
    return importlib.import_module("bosscrafts_model.semantic_calibration_promotion")


def _rows(prefix: str, *, all_incorrect: bool = False):
    rows = []
    for language in ("javascript", "python"):
        for index in range(20):
            high = index >= 10
            rows.append(
                {
                    "schema_version": "bosscrafts-semantic-uncertainty-observation-v1",
                    "sample_id": f"{prefix}-{language}-{index}",
                    "review_id": f"{prefix}-{language}-{index}",
                    "language": language,
                    "raw_uncertainty": 0.8 if high else 0.2,
                    "incorrect": True if all_incorrect else high,
                    "decision": "reject" if (all_incorrect or high) else "keep",
                    "resolution": "dual_agreement",
                    "resolver": None,
                    "reviewer_entry_sha256": ["a" * 64, "b" * 64],
                    "parser_runtime_identity_sha256": "c" * 64,
                }
            )
    return rows


def _dataset(root: Path, *, all_incorrect: bool = False):
    calibration = _rows("calibration", all_incorrect=all_incorrect)
    heldout = _rows("heldout", all_incorrect=all_incorrect)
    calibration_text = "".join(f"{_canonical(row)}\n" for row in calibration)
    heldout_text = "".join(f"{_canonical(row)}\n" for row in heldout)
    (root / "calibration_observations.jsonl").write_text(
        calibration_text, encoding="utf-8", newline="\n"
    )
    (root / "heldout_observations.jsonl").write_text(
        heldout_text, encoding="utf-8", newline="\n"
    )
    unsigned = {
        "schema_version": "bosscrafts-semantic-calibration-dataset-v1",
        "source_review_manifest_sha256": "d" * 64,
        "source_queue_sha256": "e" * 64,
        "source_dual_review_manifest_sha256": "f" * 64,
        "parser_runtime_identity_sha256": "c" * 64,
        "calibration_per_language": 20,
        "heldout_per_language": 20,
        "language_counts": {
            language: {"calibration": 20, "heldout": 20}
            for language in ("javascript", "python")
        },
        "calibration_file": "calibration_observations.jsonl",
        "calibration_count": 40,
        "calibration_sha256": _sha(calibration_text),
        "heldout_file": "heldout_observations.jsonl",
        "heldout_count": 40,
        "heldout_sha256": _sha(heldout_text),
        "manual_decisions_file": "manual_decisions.jsonl",
        "manual_decisions_sha256": _sha(""),
        "calibration_ready": True,
        "training_ready": False,
        "training_blockers": ["fixture"],
    }
    manifest = {**unsigned, "manifest_sha256": _sha(_canonical(unsigned))}
    (root / "manifest.json").write_text(
        f"{_canonical(manifest)}\n", encoding="utf-8", newline="\n"
    )


def _policy(feature):
    return feature.SemanticCalibrationPromotionPolicy(
        contained_thresholds=CalibrationThresholds(
            minimum_heldout_samples=40,
            minimum_languages=2,
            minimum_samples_per_language=20,
            maximum_calibrated_ece=0.05,
            maximum_calibrated_brier=0.05,
            maximum_brier_degradation=0.0,
        ),
        strict_policy=StrictUncertaintyPolicy(
            minimum_global_selective_coverage_at_025=0.4,
            maximum_global_selective_error_rate_at_025=0.05,
            minimum_per_language_selective_coverage_at_025=0.4,
            maximum_per_language_selective_error_rate_at_025=0.05,
        ),
    )


def test_promotes_calibration_only_when_contained_and_strict_gates_pass():
    feature = _feature()
    with tempfile.TemporaryDirectory() as dataset_dir, tempfile.TemporaryDirectory() as output_dir:
        _dataset(Path(dataset_dir))
        result = feature.promote_semantic_uncertainty(
            dataset_dir, output_dir, policy=_policy(feature)
        )
        assert result.promoted is True
        assert result.failure_codes == ()
        artifact = json.loads(
            (Path(output_dir) / "uncertainty-calibration.json").read_text(encoding="utf-8")
        )
        strict = json.loads(
            (Path(output_dir) / "strict-uncertainty-acceptance.json").read_text(
                encoding="utf-8"
            )
        )
        assert artifact["promoted"] is True
        assert strict["passed"] is True


def test_strict_selective_gate_can_veto_contained_calibration():
    feature = _feature()
    with tempfile.TemporaryDirectory() as dataset_dir, tempfile.TemporaryDirectory() as output_dir:
        _dataset(Path(dataset_dir), all_incorrect=True)
        policy = feature.SemanticCalibrationPromotionPolicy(
            contained_thresholds=CalibrationThresholds(
                minimum_heldout_samples=40,
                minimum_languages=2,
                minimum_samples_per_language=20,
                maximum_calibrated_ece=1.0,
                maximum_calibrated_brier=1.0,
                maximum_brier_degradation=1.0,
            ),
            strict_policy=_policy(feature).strict_policy,
        )
        result = feature.promote_semantic_uncertainty(
            dataset_dir, output_dir, policy=policy
        )
        assert result.promoted is False
        assert "global_selective_coverage_below_threshold" in result.failure_codes


def test_tampered_dataset_is_rejected_before_calibration():
    feature = _feature()
    with tempfile.TemporaryDirectory() as dataset_dir, tempfile.TemporaryDirectory() as output_dir:
        root = Path(dataset_dir)
        _dataset(root)
        with (root / "heldout_observations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with pytest.raises(ValueError, match="heldout observations SHA-256"):
            feature.promote_semantic_uncertainty(
                dataset_dir, output_dir, policy=_policy(feature)
            )


def test_cli_exposes_frozen_v10_policy_before_real_labels_are_seen():
    script = PROJECT_ROOT / "scripts" / "promote_semantic_uncertainty.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--print-policy"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    policy = json.loads(completed.stdout)
    assert policy["contained_thresholds"] == {
        "minimum_heldout_samples": 140,
        "minimum_languages": 14,
        "minimum_samples_per_language": 10,
        "maximum_calibrated_ece": 0.15,
        "maximum_calibrated_brier": 0.25,
        "maximum_brier_degradation": 0.02,
    }
    assert policy["strict_policy"] == {
        "minimum_global_selective_coverage_at_025": 0.3,
        "maximum_global_selective_error_rate_at_025": 0.1,
        "minimum_per_language_selective_coverage_at_025": 0.1,
        "maximum_per_language_selective_error_rate_at_025": 0.25,
    }
