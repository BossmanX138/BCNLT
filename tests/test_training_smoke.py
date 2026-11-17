from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from bosscrafts_model import model as model_module
from bosscrafts_model.config import ModelConfig


class MaskedNextTokenLossTests(unittest.TestCase):
    def test_prefix_and_final_nonpredicting_logits_receive_no_gradient(self):
        masked_loss = getattr(model_module, "masked_next_token_loss", None)
        self.assertIsNotNone(
            masked_loss,
            "masked_next_token_loss is required for packed loss boundaries",
        )
        ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        logits = torch.zeros((1, 4, 4), dtype=torch.float32, requires_grad=True)

        loss = masked_loss(logits, ids, torch.tensor([2], dtype=torch.long))
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[0, 1].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[0, 2].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[0, 3].abs().sum()), 0.0)

    def test_invalid_packed_loss_boundaries_are_rejected(self):
        masked_loss = getattr(model_module, "masked_next_token_loss", None)
        self.assertIsNotNone(masked_loss)
        ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        logits = torch.zeros((1, 4, 4), dtype=torch.float32)

        for boundary in (0, 4):
            with self.subTest(boundary=boundary):
                with self.assertRaisesRegex(ValueError, "loss_start_indices"):
                    masked_loss(
                        logits,
                        ids,
                        torch.tensor([boundary], dtype=torch.long),
                    )


class NativeTrainingSmokeScriptTests(unittest.TestCase):
    def test_tiny_cpu_smoke_executes_one_checkpointed_masked_step_without_writing_weights(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "smoke_native_65k_training.py"),
                "--device",
                "cpu",
                "--gradient-checkpointing",
                "--optimizer",
                "adamw",
                "--sequence-length",
                "24",
            ],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["trained_steps"], 1)
        self.assertFalse(payload["full_model"])
        self.assertEqual(payload["device"], "cpu")
        self.assertTrue(payload["gradient_checkpointing"])
        self.assertEqual(payload["sequence_length"], 24)
        self.assertEqual(payload["optimizer"], "AdamW")
        self.assertTrue(payload["loss_finite"])
        self.assertTrue(payload["gradient_finite"])
        self.assertGreater(payload["gradient_norm"], 0.0)
        self.assertTrue(payload["parameter_changed"])
        self.assertFalse(payload["checkpoint_written"])
        self.assertFalse(payload["training_ready"])
        self.assertIn("tokenizers_runtime_path", payload)
        self.assertIn("tokenizers_version", payload)
        runtime_path = Path(payload["tokenizers_runtime_path"]).resolve()
        expected_runtime = (project_root / "vendor" / "tokenizers_runtime_0_23").resolve()
        self.assertTrue(runtime_path.is_relative_to(expected_runtime))
        self.assertEqual(payload["tokenizers_version"], "0.23.2")


class GradientCheckpointingTests(unittest.TestCase):
    def test_opt_in_checkpointing_reduces_saved_activations_and_preserves_gradients(self):
        torch.manual_seed(19)
        config = ModelConfig.tiny(layers=4, max_sequence=16)
        baseline = model_module.EngineeringModel(config).train()
        checkpointed = model_module.EngineeringModel(config).train()
        checkpointed.load_state_dict(baseline.state_dict())
        enable = getattr(checkpointed, "gradient_checkpointing_enable", None)
        self.assertIsNotNone(enable, "model must expose opt-in gradient checkpointing")
        enable()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

        def run(model):
            saved_numel = 0

            def pack(tensor):
                nonlocal saved_numel
                saved_numel += tensor.numel()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                logits, _ = model(ids)
                loss = model_module.next_token_loss(logits, ids)
            loss.backward()
            gradient_norm = float(model.embedding.weight.grad.norm())
            return logits.detach(), saved_numel, gradient_norm

        baseline_logits, baseline_saved, baseline_gradient = run(baseline)
        checkpointed_logits, checkpointed_saved, checkpointed_gradient = run(checkpointed)
        self.assertTrue(torch.equal(checkpointed_logits, baseline_logits))
        self.assertGreater(baseline_gradient, 0.0)
        self.assertGreater(checkpointed_gradient, 0.0)
        self.assertLess(checkpointed_saved, baseline_saved)


if __name__ == "__main__":
    unittest.main()
