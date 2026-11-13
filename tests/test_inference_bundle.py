from __future__ import annotations

import hashlib
import json
import shutil
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
                expected_bundle_sha256=status.bundle_sha256,
            )
            self.assertEqual(state.checkpoint_step, 4)
            self.assertEqual(state.training_plan_sha256, "c" * 64)
            self.assertTrue(
                torch.equal(model.embedding.weight, restored.embedding.weight)
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                load_inference_bundle(
                    bundle,
                    EngineeringModel(config),
                    expected_task_database_identity_sha256="b" * 64,
                    expected_binding_sha256=self.binding.binding_sha256,
                    expected_model_fingerprint=config.fingerprint(),
                    expected_bundle_sha256="f" * 64,
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

    def test_loader_rejects_extra_manifest_fields_and_bundle_entries(self):
        config = ModelConfig.tiny(
            vocab_size=self.binding.vocab_size,
            tokenizer_manifest=self.binding.binding_sha256,
        )
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
                step=1,
                examples_seen=1,
                epoch_index=1,
                target_epochs=1,
                last_example_id="a" * 64,
                cursor_after_example_id=None,
                task_database_identity_sha256="b" * 64,
                binding_sha256=self.binding.binding_sha256,
                model_fingerprint=config.fingerprint(),
                training_plan_sha256="c" * 64,
            )
            export_inference_bundle(
                checkpoint,
                bundle,
                model,
                binding=self.binding,
                expected_task_database_identity_sha256="b" * 64,
                expected_model_fingerprint=config.fingerprint(),
            )

            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["unverified_override"] = True
            unsigned = dict(manifest)
            unsigned.pop("bundle_sha256")
            manifest["bundle_sha256"] = hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields"):
                load_inference_bundle(
                    bundle,
                    EngineeringModel(config),
                    expected_task_database_identity_sha256="b" * 64,
                    expected_binding_sha256=self.binding.binding_sha256,
                    expected_model_fingerprint=config.fingerprint(),
                )

            manifest.pop("unverified_override")
            unsigned = dict(manifest)
            unsigned.pop("bundle_sha256")
            manifest["bundle_sha256"] = hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (bundle / "unexpected.txt").write_text("not part of the bundle", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory"):
                load_inference_bundle(
                    bundle,
                    EngineeringModel(config),
                    expected_task_database_identity_sha256="b" * 64,
                    expected_binding_sha256=self.binding.binding_sha256,
                    expected_model_fingerprint=config.fingerprint(),
                )

            (bundle / "unexpected.txt").unlink()
            manifest_path.write_text(
                '{"bundle_sha256":"a","bundle_sha256":"b"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_inference_bundle(
                    bundle,
                    EngineeringModel(config),
                    expected_task_database_identity_sha256="b" * 64,
                    expected_binding_sha256=self.binding.binding_sha256,
                    expected_model_fingerprint=config.fingerprint(),
                )

    def test_loader_rejects_rehashed_model_payload_extensions(self):
        config = ModelConfig.tiny(
            vocab_size=self.binding.vocab_size,
            tokenizer_manifest=self.binding.binding_sha256,
        )
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
                step=1,
                examples_seen=1,
                epoch_index=1,
                target_epochs=1,
                last_example_id="a" * 64,
                cursor_after_example_id=None,
                task_database_identity_sha256="b" * 64,
                binding_sha256=self.binding.binding_sha256,
                model_fingerprint=config.fingerprint(),
                training_plan_sha256="c" * 64,
            )
            export_inference_bundle(
                checkpoint,
                bundle,
                model,
                binding=self.binding,
                expected_task_database_identity_sha256="b" * 64,
                expected_model_fingerprint=config.fingerprint(),
            )

            model_path = bundle / "model.pt"
            payload = torch.load(model_path, map_location="cpu", weights_only=True)
            payload["unverified_override"] = True
            torch.save(payload, model_path)
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["model_file_bytes"] = model_path.stat().st_size
            manifest["model_file_sha256"] = hashlib.sha256(
                model_path.read_bytes()
            ).hexdigest()
            unsigned = dict(manifest)
            unsigned.pop("bundle_sha256")
            manifest["bundle_sha256"] = hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model payload"):
                load_inference_bundle(
                    bundle,
                    EngineeringModel(config),
                    expected_task_database_identity_sha256="b" * 64,
                    expected_binding_sha256=self.binding.binding_sha256,
                    expected_model_fingerprint=config.fingerprint(),
                )

    def test_export_ignores_orphaned_legacy_temporary_directory(self):
        config = ModelConfig.tiny(
            vocab_size=self.binding.vocab_size,
            tokenizer_manifest=self.binding.binding_sha256,
        )
        model = EngineeringModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            bundle = root / "bundle"
            orphan = root / ".bundle.tmp"
            orphan.mkdir()
            save_training_checkpoint(
                checkpoint,
                model,
                optimizer,
                step=1,
                examples_seen=1,
                epoch_index=1,
                target_epochs=1,
                last_example_id="a" * 64,
                cursor_after_example_id=None,
                task_database_identity_sha256="b" * 64,
                binding_sha256=self.binding.binding_sha256,
                model_fingerprint=config.fingerprint(),
                training_plan_sha256="c" * 64,
            )
            export_inference_bundle(
                checkpoint,
                bundle,
                model,
                binding=self.binding,
                expected_task_database_identity_sha256="b" * 64,
                expected_model_fingerprint=config.fingerprint(),
            )
            self.assertTrue(bundle.is_dir())
            shutil.rmtree(orphan)

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
