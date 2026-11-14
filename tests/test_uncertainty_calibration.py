from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_import import select_bcnlt_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = (
    PROJECT_ROOT
    / "vendor"
    / "bcnlt_runtime_v10_learning_candidate"
    / "source"
)
select_bcnlt_runtime(RUNTIME_SOURCE)

from bcnlt.cache import ProjectIndex  # noqa: E402
from bcnlt.core import ContextualUnit  # noqa: E402
from bcnlt.uncertainty import HeuristicUncertaintyScorer  # noqa: E402
from bcnlt.uncertainty_calibration import (  # noqa: E402
    CalibratedUncertaintyScorer,
    CalibrationThresholds,
    UncertaintyObservation,
    evaluate_uncertainty_calibration,
    fit_uncertainty_calibrator,
    load_calibration_artifact,
    write_calibration_artifact,
)


class UncertaintyCalibrationTests(unittest.TestCase):
    @staticmethod
    def _observations(prefix: str, *, reversed_labels: bool = False):
        rows = []
        for language in ("python", "javascript"):
            for index in range(20):
                high = index >= 10
                incorrect = high
                if reversed_labels:
                    incorrect = not incorrect
                rows.append(
                    UncertaintyObservation(
                        sample_id=f"{prefix}-{language}-{index}",
                        language=language,
                        raw_uncertainty=0.8 if high else 0.2,
                        incorrect=incorrect,
                    )
                )
        return tuple(rows)

    @staticmethod
    def _thresholds() -> CalibrationThresholds:
        return CalibrationThresholds(
            minimum_heldout_samples=40,
            minimum_languages=2,
            minimum_samples_per_language=20,
            maximum_calibrated_ece=0.05,
            maximum_calibrated_brier=0.05,
            maximum_brier_degradation=0.0,
        )

    def test_promoted_artifact_improves_heldout_calibration_and_wraps_scorer(self):
        calibration = self._observations("calibration")
        heldout = self._observations("heldout")
        calibrator = fit_uncertainty_calibrator(
            calibration,
            parser_runtime_identity_sha256="a" * 64,
        )
        artifact = evaluate_uncertainty_calibration(
            calibrator,
            heldout,
            thresholds=self._thresholds(),
            heldout_manifest_sha256="b" * 64,
        )

        self.assertTrue(artifact.promoted)
        self.assertEqual(artifact.failure_codes, ())
        self.assertLess(artifact.calibrated_metrics.brier, artifact.raw_metrics.brier)
        self.assertLess(artifact.calibrated_metrics.ece, artifact.raw_metrics.ece)
        self.assertEqual(set(artifact.per_language), {"javascript", "python"})

        scorer = CalibratedUncertaintyScorer(
            HeuristicUncertaintyScorer(),
            artifact,
        )
        score = scorer.score(
            ContextualUnit(
                surface="value",
                event="ASSIGN",
                scope="module",
                span=(1, 0, 1, 5),
                ast_relation="assignment.left",
            ),
            ProjectIndex(),
        )
        self.assertEqual(score.uncertainty, 0.0)
        self.assertEqual(score.confidence, 1.0)
        self.assertIn("held-out calibrated", score.signals)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uncertainty-calibration.json"
            write_calibration_artifact(path, artifact)
            self.assertEqual(load_calibration_artifact(path), artifact)

    def test_failed_heldout_gate_cannot_be_used_by_runtime(self):
        calibrator = fit_uncertainty_calibrator(
            self._observations("calibration"),
            parser_runtime_identity_sha256="c" * 64,
        )
        artifact = evaluate_uncertainty_calibration(
            calibrator,
            self._observations("heldout", reversed_labels=True),
            thresholds=self._thresholds(),
            heldout_manifest_sha256="d" * 64,
        )

        self.assertFalse(artifact.promoted)
        self.assertIn("calibrated_brier_above_threshold", artifact.failure_codes)
        with self.assertRaisesRegex(ValueError, "not promoted"):
            CalibratedUncertaintyScorer(HeuristicUncertaintyScorer(), artifact)

    def test_calibration_and_heldout_samples_must_be_disjoint(self):
        calibration = self._observations("shared")
        calibrator = fit_uncertainty_calibrator(
            calibration,
            parser_runtime_identity_sha256="e" * 64,
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_uncertainty_calibration(
                calibrator,
                calibration,
                thresholds=self._thresholds(),
                heldout_manifest_sha256="f" * 64,
            )

    def test_artifact_integrity_is_checked_before_runtime_use(self):
        calibrator = fit_uncertainty_calibrator(
            self._observations("calibration"),
            parser_runtime_identity_sha256="1" * 64,
        )
        artifact = evaluate_uncertainty_calibration(
            calibrator,
            self._observations("heldout"),
            thresholds=self._thresholds(),
            heldout_manifest_sha256="2" * 64,
        )
        tampered = replace(artifact, heldout_manifest_sha256="3" * 64)
        with self.assertRaisesRegex(ValueError, "integrity"):
            CalibratedUncertaintyScorer(HeuristicUncertaintyScorer(), tampered)


class EntropyProviderPolicyTests(unittest.TestCase):
    @staticmethod
    def _unit() -> ContextualUnit:
        return ContextualUnit(
            surface="value",
            event="ASSIGN",
            scope="module",
            span=(1, 0, 1, 5),
            ast_relation="assignment.left",
        )

    def test_default_scorer_explicitly_has_no_model_entropy_provider(self):
        scorer = HeuristicUncertaintyScorer()

        self.assertIsNone(scorer.entropy_provider)
        score = scorer.score(self._unit(), ProjectIndex())
        self.assertNotIn("model entropy", score.signals)

    def test_injected_model_entropy_is_applied_and_labeled(self):
        class Provider:
            def entropy_for(self, unit: ContextualUnit) -> float:
                return 0.5

        score = HeuristicUncertaintyScorer(Provider()).score(
            self._unit(), ProjectIndex()
        )

        self.assertAlmostEqual(score.uncertainty, 0.15)
        self.assertAlmostEqual(score.confidence, 0.85)
        self.assertIn("model entropy", score.signals)

    def test_invalid_model_entropy_fails_closed(self):
        class Provider:
            def __init__(self, value: float) -> None:
                self.value = value

            def entropy_for(self, unit: ContextualUnit) -> float:
                return self.value

        for value in (-0.1, 1.1, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "model entropy"):
                    HeuristicUncertaintyScorer(Provider(value)).score(
                        self._unit(), ProjectIndex()
                    )



if __name__ == "__main__":
    unittest.main()
