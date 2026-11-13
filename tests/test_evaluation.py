from __future__ import annotations

import importlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from bosscrafts_model.config import ModelConfig
from bosscrafts_model.model import EngineeringModel


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def _baseline_payload() -> dict[str, object]:
        return {
            "split": "validation",
            "evaluation_complete": True,
            "example_count": 2,
            "split_example_count": 2,
            "task_database_identity_sha256": "a" * 64,
            "model_fingerprint": "b" * 64,
            "mean_loss": 1.0,
            "perplexity": math.e,
            "target_token_accuracy": 0.5,
            "target_token_count": 6,
            "correct_target_token_count": 3,
            "exact_match_count": 1,
            "exact_match_rate": 0.5,
            "language_mean_loss": {"python": 1.0},
            "family_mean_loss": {"code_to_semantic": 1.0},
            "cell_mean_loss": {"python|code_to_semantic": 1.0},
        }

    def test_loads_lineage_bound_evaluation_baseline(self):
        module = importlib.import_module("bosscrafts_model.evaluation")
        self.assertTrue(
            hasattr(module, "load_evaluation_baseline"),
            "lineage-bound evaluation baseline reader is required",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(self._baseline_payload()), encoding="utf-8")
            baseline = module.load_evaluation_baseline(
                path,
                expected_split="validation",
                expected_example_count=2,
                expected_task_database_identity_sha256="a" * 64,
                expected_model_fingerprint="b" * 64,
            )
        self.assertEqual(baseline.mean_loss, 1.0)
        self.assertEqual(baseline.language_mean_loss, {"python": 1.0})

    def test_rejects_invalid_evaluation_baseline_metrics(self):
        module = importlib.import_module("bosscrafts_model.evaluation")
        invalid = (
            ("mean_loss", 0.0),
            ("mean_loss", float("nan")),
            ("perplexity", 0.0),
            ("perplexity", float("inf")),
            ("target_token_accuracy", -0.1),
            ("target_token_accuracy", 1.1),
            ("correct_target_token_count", True),
            ("correct_target_token_count", -1),
            ("exact_match_count", 3),
            ("exact_match_rate", 1.1),
        )
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                payload = self._baseline_payload()
                payload[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "baseline.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "baseline evaluation metrics"
                    ):
                        module.load_evaluation_baseline(
                            path,
                            expected_split="validation",
                            expected_example_count=2,
                            expected_task_database_identity_sha256="a" * 64,
                            expected_model_fingerprint="b" * 64,
                        )

    def test_rejects_invalid_evaluation_baseline_breakdown(self):
        module = importlib.import_module("bosscrafts_model.evaluation")
        invalid = (None, {}, {"": 1.0}, {"python": float("nan")}, {"python": -1.0})
        for value in invalid:
            with self.subTest(value=value):
                payload = self._baseline_payload()
                payload["language_mean_loss"] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "baseline.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "baseline evaluation breakdown"
                    ):
                        module.load_evaluation_baseline(
                            path,
                            expected_split="validation",
                            expected_example_count=2,
                            expected_task_database_identity_sha256="a" * 64,
                            expected_model_fingerprint="b" * 64,
                        )

    def test_rejects_internally_inconsistent_evaluation_baseline(self):
        module = importlib.import_module("bosscrafts_model.evaluation")
        invalid = (
            {"perplexity": 99.0},
            {"target_token_count": 0},
            {"correct_target_token_count": 4},
            {"exact_match_rate": 0.4},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                payload = self._baseline_payload()
                payload.update(changes)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "baseline.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "baseline evaluation consistency"
                    ):
                        module.load_evaluation_baseline(
                            path,
                            expected_split="validation",
                            expected_example_count=2,
                            expected_task_database_identity_sha256="a" * 64,
                            expected_model_fingerprint="b" * 64,
                        )

    def test_unspecified_cuda_index_matches_the_active_cuda_device(self):
        module = importlib.import_module("bosscrafts_model.evaluation")
        self.assertTrue(
            module._device_matches(torch.device("cuda:0"), torch.device("cuda"))
        )
        self.assertFalse(
            module._device_matches(torch.device("cuda:1"), torch.device("cuda:0"))
        )

    def test_streaming_evaluation_reports_masked_loss_perplexity_and_breakdowns(self):
        module_name = "bosscrafts_model.evaluation"
        spec = importlib.util.find_spec(module_name)
        self.assertIsNotNone(spec, "evaluation module is required")
        if spec is None:
            return
        module = importlib.import_module(module_name)
        torch.manual_seed(97)
        model = EngineeringModel(ModelConfig.tiny()).train()
        examples = [
            {
                "example_id": "a" * 64,
                "language": "python",
                "family": "code_to_semantic",
                "input_ids": [1, 2, 3, 4, 5],
                "loss_start_index": 2,
                "sampling_weight": 0.5,
            },
            {
                "example_id": "b" * 64,
                "language": "rust",
                "family": "semantic_to_code",
                "input_ids": [6, 7, 8, 9, 10, 11],
                "loss_start_index": 3,
                "sampling_weight": 2.0,
            },
        ]

        result = module.evaluate_model(
            model,
            examples,
            device=torch.device("cpu"),
            max_examples=2,
        )

        self.assertEqual(result.example_count, 2)
        self.assertEqual(result.target_token_count, 6)
        self.assertGreaterEqual(result.correct_target_token_count, 0)
        self.assertLessEqual(result.correct_target_token_count, 6)
        self.assertGreaterEqual(result.target_token_accuracy, 0.0)
        self.assertLessEqual(result.target_token_accuracy, 1.0)
        self.assertGreaterEqual(result.exact_match_count, 0)
        self.assertLessEqual(result.exact_match_count, 2)
        self.assertEqual(result.exact_match_rate, result.exact_match_count / 2)
        self.assertTrue(math.isfinite(result.mean_loss))
        self.assertGreater(result.perplexity, 0.0)
        self.assertEqual(result.language_counts, {"python": 1, "rust": 1})
        self.assertEqual(
            result.family_counts,
            {"code_to_semantic": 1, "semantic_to_code": 1},
        )
        self.assertEqual(
            result.cell_counts,
            {
                "python|code_to_semantic": 1,
                "rust|semantic_to_code": 1,
            },
        )
        self.assertEqual(set(result.cell_mean_loss), set(result.cell_counts))
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
