from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import torch

from bosscrafts_model.config import ModelConfig
from bosscrafts_model.model import EngineeringModel, next_token_loss


class TrainingCheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_restores_model_optimizer_cursor_and_rng(self):
        module_name = "bosscrafts_model.training_checkpoint"
        spec = importlib.util.find_spec(module_name)
        self.assertIsNotNone(spec, "training checkpoint module is required")
        if spec is None:
            return
        module = importlib.import_module(module_name)
        config = ModelConfig.tiny()
        torch.manual_seed(31)
        model = EngineeringModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        loss = next_token_loss(model(ids)[0], ids)
        loss.backward()
        optimizer.step()
        expected_weights = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        task_identity = "a" * 64
        binding_identity = "b" * 64
        plan_identity = "d" * 64
        expected_random = torch.rand(4)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.manual_seed(47)
            module.save_training_checkpoint(
                path,
                model,
                optimizer,
                step=7,
                examples_seen=19,
                epoch_index=2,
                target_epochs=3,
                last_example_id="c" * 64,
                cursor_after_example_id="c" * 64,
                task_database_identity_sha256=task_identity,
                binding_sha256=binding_identity,
                model_fingerprint=config.fingerprint(),
                training_plan_sha256=plan_identity,
            )
            expected_random = torch.rand(4)
            torch.manual_seed(999)
            restored_model = EngineeringModel(config)
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=0.001
            )
            state = module.load_training_checkpoint(
                path,
                restored_model,
                restored_optimizer,
                expected_task_database_identity_sha256=task_identity,
                expected_binding_sha256=binding_identity,
                expected_model_fingerprint=config.fingerprint(),
                expected_target_epochs=3,
                expected_training_plan_sha256=plan_identity,
            )
            self.assertEqual(state.step, 7)
            self.assertEqual(
                state.checkpoint_sha256,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(state.examples_seen, 19)
            self.assertEqual(state.epoch_index, 2)
            self.assertEqual(state.target_epochs, 3)
            self.assertEqual(state.last_example_id, "c" * 64)
            self.assertEqual(state.cursor_after_example_id, "c" * 64)
            for name, tensor in restored_model.state_dict().items():
                self.assertTrue(torch.equal(tensor, expected_weights[name]), name)
            self.assertTrue(torch.equal(torch.rand(4), expected_random))
            self.assertFalse(path.with_name(f".{path.name}.tmp").exists())

    def test_checkpoint_lineage_mismatch_fails_before_loading_weights(self):
        module_name = "bosscrafts_model.training_checkpoint"
        spec = importlib.util.find_spec(module_name)
        self.assertIsNotNone(spec, "training checkpoint module is required")
        if spec is None:
            return
        module = importlib.import_module(module_name)
        config = ModelConfig.tiny()
        model = EngineeringModel(config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            module.save_training_checkpoint(
                path,
                model,
                optimizer,
                step=1,
                examples_seen=1,
                epoch_index=0,
                target_epochs=2,
                last_example_id="c" * 64,
                cursor_after_example_id="c" * 64,
                task_database_identity_sha256="a" * 64,
                binding_sha256="b" * 64,
                model_fingerprint=config.fingerprint(),
                training_plan_sha256="d" * 64,
            )
            before = model.embedding.weight.detach().clone()
            with self.assertRaisesRegex(ValueError, "lineage"):
                module.load_training_checkpoint(
                    path,
                    model,
                    optimizer,
                    expected_task_database_identity_sha256="d" * 64,
                    expected_binding_sha256="b" * 64,
                    expected_model_fingerprint=config.fingerprint(),
                    expected_target_epochs=2,
                    expected_training_plan_sha256="d" * 64,
                )
            self.assertTrue(torch.equal(model.embedding.weight, before))

    def test_rejects_malformed_model_optimizer_and_rng_payloads(self):
        module = importlib.import_module("bosscrafts_model.training_checkpoint")
        config = ModelConfig.tiny()
        model = EngineeringModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            module.save_training_checkpoint(
                source,
                model,
                optimizer,
                step=1,
                examples_seen=1,
                epoch_index=0,
                target_epochs=1,
                last_example_id="c" * 64,
                cursor_after_example_id="c" * 64,
                task_database_identity_sha256="a" * 64,
                binding_sha256="b" * 64,
                model_fingerprint=config.fingerprint(),
                training_plan_sha256="d" * 64,
            )
            valid = torch.load(source, map_location="cpu", weights_only=True)
            cases = {
                "model state": ("model_state_dict", []),
                "optimizer state": ("optimizer_state_dict", []),
                "CPU RNG state": ("cpu_rng_state", "invalid"),
                "CUDA RNG state": ("cuda_rng_state_all", ["invalid"]),
            }
            for label, (key, replacement) in cases.items():
                with self.subTest(label=label):
                    malformed = dict(valid)
                    malformed[key] = replacement
                    path = root / f"{key}.pt"
                    torch.save(malformed, path)
                    with self.assertRaisesRegex(ValueError, label):
                        module.load_training_checkpoint(
                            path,
                            model,
                            optimizer,
                            expected_task_database_identity_sha256="a" * 64,
                            expected_binding_sha256="b" * 64,
                            expected_model_fingerprint=config.fingerprint(),
                            expected_target_epochs=1,
                            expected_training_plan_sha256="d" * 64,
                        )

    def test_rejects_incoherent_resume_cursor_and_unexpected_fields(self):
        module = importlib.import_module("bosscrafts_model.training_checkpoint")
        config = ModelConfig.tiny()
        model = EngineeringModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            module.save_training_checkpoint(
                source,
                model,
                optimizer,
                step=1,
                examples_seen=1,
                epoch_index=0,
                target_epochs=1,
                last_example_id="c" * 64,
                cursor_after_example_id="c" * 64,
                task_database_identity_sha256="a" * 64,
                binding_sha256="b" * 64,
                model_fingerprint=config.fingerprint(),
                training_plan_sha256="d" * 64,
            )
            valid = torch.load(source, map_location="cpu", weights_only=True)
            cases = {
                "cursor": {"cursor_after_example_id": "e" * 64},
                "cursor at completion": {
                    "epoch_index": 1,
                    "cursor_after_example_id": "c" * 64,
                },
                "fields": {"unverified_override": True},
                "progress": {"step": 2, "examples_seen": 1},
            }
            for label, changes in cases.items():
                with self.subTest(label=label):
                    malformed = dict(valid)
                    malformed.update(changes)
                    path = root / f"{label.replace(' ', '-')}.pt"
                    torch.save(malformed, path)
                    with self.assertRaisesRegex(ValueError, label):
                        module.load_training_checkpoint(
                            path,
                            model,
                            optimizer,
                            expected_task_database_identity_sha256="a" * 64,
                            expected_binding_sha256="b" * 64,
                            expected_model_fingerprint=config.fingerprint(),
                            expected_target_epochs=1,
                            expected_training_plan_sha256="d" * 64,
                        )

            with self.assertRaisesRegex(ValueError, "cursor"):
                module.save_training_checkpoint(
                    root / "invalid-save.pt",
                    model,
                    optimizer,
                    step=1,
                    examples_seen=1,
                    epoch_index=0,
                    target_epochs=1,
                    last_example_id="c" * 64,
                    cursor_after_example_id="e" * 64,
                    task_database_identity_sha256="a" * 64,
                    binding_sha256="b" * 64,
                    model_fingerprint=config.fingerprint(),
                    training_plan_sha256="d" * 64,
                )

    def test_corrupt_checkpoint_fails_as_unreadable_evidence(self):
        module = importlib.import_module("bosscrafts_model.training_checkpoint")
        config = ModelConfig.tiny()
        model = EngineeringModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(os.urandom(128))
            with self.assertRaisesRegex(ValueError, "unreadable"):
                module.load_training_checkpoint(
                    path,
                    model,
                    optimizer,
                    expected_task_database_identity_sha256="a" * 64,
                    expected_binding_sha256="b" * 64,
                    expected_model_fingerprint=config.fingerprint(),
                    expected_target_epochs=1,
                    expected_training_plan_sha256="d" * 64,
                )

    def test_rejects_nonfinite_checkpoint_state_on_save_and_load(self):
        module = importlib.import_module("bosscrafts_model.training_checkpoint")
        config = ModelConfig.tiny()
        model = EngineeringModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        common = {
            "step": 1,
            "examples_seen": 1,
            "epoch_index": 0,
            "target_epochs": 1,
            "last_example_id": "c" * 64,
            "cursor_after_example_id": "c" * 64,
            "task_database_identity_sha256": "a" * 64,
            "binding_sha256": "b" * 64,
            "model_fingerprint": config.fingerprint(),
            "training_plan_sha256": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.pt"
            module.save_training_checkpoint(valid_path, model, optimizer, **common)
            payload = torch.load(valid_path, map_location="cpu", weights_only=True)
            poisoned_state = dict(payload["model_state_dict"])
            first_name = next(iter(poisoned_state))
            poisoned_tensor = poisoned_state[first_name].clone()
            poisoned_tensor.view(-1)[0] = float("nan")
            poisoned_state[first_name] = poisoned_tensor
            payload["model_state_dict"] = poisoned_state
            poisoned_path = root / "poisoned.pt"
            torch.save(payload, poisoned_path)

            with self.assertRaisesRegex(ValueError, "non-finite model state"):
                module.load_training_checkpoint(
                    poisoned_path,
                    model,
                    optimizer,
                    expected_task_database_identity_sha256="a" * 64,
                    expected_binding_sha256="b" * 64,
                    expected_model_fingerprint=config.fingerprint(),
                    expected_target_epochs=1,
                    expected_training_plan_sha256="d" * 64,
                )

            with torch.no_grad():
                next(model.parameters()).view(-1)[0] = float("inf")
            rejected_path = root / "rejected.pt"
            with self.assertRaisesRegex(ValueError, "non-finite model state"):
                module.save_training_checkpoint(
                    rejected_path, model, optimizer, **common
                )
            self.assertFalse(rejected_path.exists())


if __name__ == "__main__":
    unittest.main()
