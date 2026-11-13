from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from bosscrafts_model.config import ModelConfig
from bosscrafts_model.inference_bundle import (
    export_inference_bundle,
    load_inference_bundle,
)
from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.tokenizer import NativeTokenizerBinding
from bosscrafts_model.training_checkpoint import save_training_checkpoint


class InferenceBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()

    def test_exports_and_reloads_complete_checkpoint_with_frozen_tokenizer(self):
        config = ModelConfig.tiny(
            vocab_size=self.binding.vocab_size,
            tokenizer_manifest=self.binding.binding_sha256,
        )
        torch.manual_seed(131)
        model = EngineeringModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            bundle = root / "bundle"
            save_training_checkpoint(
                checkpoint,
                model,
                optimizer,
                step=4,
                examples_seen=4,
                epoch_index=1,
                target_epochs=1,
                last_example_id="a" * 64,
                cursor_after_example_id=None,
                task_database_identity_sha256="b" * 64,
                binding_sha256=self.binding.binding_sha256,
                model_fingerprint=config.fingerprint(),
                training_plan_sha256="c" * 64,
            )
            status = export_inference_bundle(
                checkpoint,
                bundle,
                model,
                binding=self.binding,
                expected_task_database_identity_sha256="b" * 64,
                expected_model_fingerprint=config.fingerprint(),
            )
            self.assertTrue((bundle / "manifest.json").is_file())
            self.assertTrue((bundle / "model.pt").is_file())
            self.assertTrue((bundle / "tokenizer" / "binding.json").is_file())
            self.assertEqual(status.checkpoint_step, 4)
            self.assertRegex(status.bundle_sha256, r"^[0-9a-f]{64}$")

            restored = EngineeringModel(config)
            state = load_inference_bundle(
                bundle,
                restored,
                expected_task_database_identity_sha256="b" * 64,
                expected_binding_sha256=self.binding.binding_sha256,
                expected_model_fingerprint=config.fingerprint(),
            )
            self.assertEqual(state.checkpoint_step, 4)
            self.assertEqual(state.training_plan_sha256, "c" * 64)
            self.assertTrue(
                torch.equal(model.embedding.weight, restored.embedding.weight)
            )

            with (bundle / "model.pt").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "model file SHA-256"):
                load_inference_bundle(
                    bundle,
                    EngineeringModel(config),
                    expected_task_database_identity_sha256="b" * 64,
                    expected_binding_sha256=self.binding.binding_sha256,
                    expected_model_fingerprint=config.fingerprint(),
                )

    def test_refuses_incomplete_training_checkpoint(self):
        config = ModelConfig.tiny(
            vocab_size=self.binding.vocab_size,
            tokenizer_manifest=self.binding.binding_sha256,
        )
        model = EngineeringModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            save_training_checkpoint(
                checkpoint,
                model,
                optimizer,
                step=1,
                examples_seen=1,
                epoch_index=0,
                target_epochs=1,
                last_example_id="a" * 64,
                cursor_after_example_id="a" * 64,
                task_database_identity_sha256="b" * 64,
                binding_sha256=self.binding.binding_sha256,
                model_fingerprint=config.fingerprint(),
                training_plan_sha256="c" * 64,
            )
            with self.assertRaisesRegex(ValueError, "training is incomplete"):
                export_inference_bundle(
                    checkpoint,
                    root / "bundle",
                    model,
                    binding=self.binding,
                    expected_task_database_identity_sha256="b" * 64,
                    expected_model_fingerprint=config.fingerprint(),
                )

    def test_export_script_runs_from_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "export_inference_bundle.py"),
                "--help",
            ],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output", completed.stdout)
        self.assertIn("--full-model", completed.stdout)

        runner = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "run_inference_bundle.py"),
                "--help",
            ],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(runner.returncode, 0, runner.stderr)
        self.assertIn("--bundle", runner.stdout)
        self.assertIn("--prompt", runner.stdout)


if __name__ == "__main__":
    unittest.main()
