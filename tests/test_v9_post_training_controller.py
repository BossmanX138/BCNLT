from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_v9_post_training_until_ready import (
    ACCEPTANCE_CATEGORIES,
    EXPECTED_DATABASE_IDENTITY,
    EXPECTED_MODEL_FINGERPRINT,
    EXPECTED_OPTIMIZER_STEPS,
    EXPECTED_SPLIT_COUNT,
    _acceptance_command,
    _evaluation_command,
    _inference_command,
    _verify_evaluation,
    _verify_runtime_identity,
)


class V9PostTrainingControllerTests(unittest.TestCase):
    @staticmethod
    def _evaluation(split: str) -> dict[str, object]:
        return {
            "split": split,
            "evaluation_complete": True,
            "checkpoint_training_complete": True,
            "checkpoint_lineage_verified": True,
            "full_model": True,
            "device": "cuda",
            "dtype": "bfloat16",
            "example_count": EXPECTED_SPLIT_COUNT,
            "split_example_count": EXPECTED_SPLIT_COUNT,
            "task_database_identity_sha256": EXPECTED_DATABASE_IDENTITY,
            "model_fingerprint": EXPECTED_MODEL_FINGERPRINT,
            "mean_loss": 1.0,
            "baseline_mean_loss": 2.0,
            "relative_loss_improvement": 0.5,
            "target_token_accuracy": 0.25,
        }

    def test_full_cuda_evaluation_command_is_frozen(self) -> None:
        command = _evaluation_command("validation", Path("out.json"))
        self.assertIn("--full-model", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
        self.assertEqual(
            command[command.index("--max-examples") + 1],
            str(EXPECTED_SPLIT_COUNT),
        )

    def test_evaluation_contract_accepts_exact_complete_report(self) -> None:
        _verify_evaluation(self._evaluation("test"), "test")

    def test_heldout_inference_command_is_full_cuda_and_exact_family(self) -> None:
        command = _inference_command("semantic_to_code", Path("inference.json"))
        self.assertIn("--full-model", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
        self.assertEqual(
            command[command.index("--heldout-family") + 1], "semantic_to_code"
        )
        self.assertEqual(command[command.index("--heldout-split") + 1], "test")
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "256")

    def test_formal_acceptance_command_binds_database_and_all_categories(self) -> None:
        inferences = {
            family: Path(f"{family}.json")
            for family in (
                "code_to_semantic",
                "semantic_continuation",
                "semantic_to_code",
            )
        }
        categories = {
            category: Path(f"{category}.json") for category in ACCEPTANCE_CATEGORIES
        }
        command = _acceptance_command(inferences, categories, Path("acceptance.json"))

        self.assertIn("--task-database", command)
        self.assertIn("--heldout-manifest", command)
        for category in ACCEPTANCE_CATEGORIES:
            option = f"--{category.replace('_', '-')}-evidence"
            self.assertEqual(command[command.index(option) + 1], str(categories[category]))

    def test_evaluation_contract_rejects_partial_or_nonfinite_report(self) -> None:
        for key, value in (
            ("evaluation_complete", False),
            ("example_count", EXPECTED_SPLIT_COUNT - 1),
            ("mean_loss", float("nan")),
        ):
            payload = self._evaluation("validation")
            payload[key] = value
            with self.assertRaisesRegex(ValueError, "frozen contract"):
                _verify_evaluation(payload, "validation")

    def test_expected_training_plan_size_is_one_full_epoch(self) -> None:
        self.assertEqual(EXPECTED_OPTIMIZER_STEPS, 10_752)

    def test_v9_controller_fails_closed_on_v10_runtime_tree(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "V9 runtime identity verification failed",
        ):
            _verify_runtime_identity()


if __name__ == "__main__":
    unittest.main()
