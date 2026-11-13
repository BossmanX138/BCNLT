from __future__ import annotations

import unittest

from scripts.run_v8_training_until_complete import (
    PLANNED_OPTIMIZER_STEPS,
    TRAINING_PROGRESS_SCHEMA,
    _training_command,
    _validate_training_status,
)


class V8TrainingControllerTests(unittest.TestCase):
    @staticmethod
    def _status() -> dict[str, object]:
        return {
            "schema_version": TRAINING_PROGRESS_SCHEMA,
            "checkpoint_sha256": "d" * 64,
            "checkpoint_written": True,
            "gradients_finite": True,
            "full_model": True,
            "device": "cuda",
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
            "task_database_identity_sha256": "a" * 64,
            "model_fingerprint": "b" * 64,
            "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
            "initialization": {"type": "random_seed_7"},
            "training_plan_sha256": "c" * 64,
            "total_optimizer_steps": 100,
            "optimizer_steps_remaining": PLANNED_OPTIMIZER_STEPS - 100,
        }

    def test_accepts_exact_training_status_and_returns_narrowed_values(self) -> None:
        plan, steps = _validate_training_status(
            self._status(),
            database_identity="a" * 64,
            model_fingerprint="b" * 64,
            expected_plan_sha256="c" * 64,
            expected_checkpoint_sha256="d" * 64,
        )
        self.assertEqual(plan, "c" * 64)
        self.assertEqual(steps, 100)

    def test_rejects_training_plan_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "plan identity changed"):
            _validate_training_status(
                self._status(),
                database_identity="a" * 64,
                model_fingerprint="b" * 64,
                expected_plan_sha256="d" * 64,
                expected_checkpoint_sha256="d" * 64,
            )

    def test_rejects_checkpoint_identity_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen contract"):
            _validate_training_status(
                self._status(),
                database_identity="a" * 64,
                model_fingerprint="b" * 64,
                expected_plan_sha256="c" * 64,
                expected_checkpoint_sha256="e" * 64,
            )

    def test_training_command_freezes_full_cuda_plan(self) -> None:
        command = _training_command()
        self.assertIn("--full-model", command)
        self.assertIn("--gradient-checkpointing", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
        self.assertEqual(command[command.index("--gradient-accumulation") + 1], "8")
        self.assertEqual(command[command.index("--max-steps") + 1], "100")
        self.assertEqual(command[command.index("--epochs") + 1], "1")


if __name__ == "__main__":
    unittest.main()
