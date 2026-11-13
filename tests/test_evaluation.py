from __future__ import annotations

import importlib
import importlib.util
import math
import unittest

import torch

from bosscrafts_model.config import ModelConfig
from bosscrafts_model.model import EngineeringModel


class EvaluationTests(unittest.TestCase):
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
