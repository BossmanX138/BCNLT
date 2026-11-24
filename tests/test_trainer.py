from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zlib
from array import array
from pathlib import Path

import torch

from bosscrafts_model.artifact_schemas import MODEL_HELDOUT_SUITE_SCHEMA
from bosscrafts_model.bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from bosscrafts_model.config import ModelConfig
from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.model_acceptance import (
    ACCEPTANCE_CATEGORIES,
    EXPECTED_LANGUAGES,
    MINIMUM_CATEGORY_EXAMPLES,
    STRUCTURED_TASKS,
)
from bosscrafts_model.resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


class TrainerTests(unittest.TestCase):
    def test_full_model_training_requires_a_frozen_evaluation_contract(self):
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "run_bounded_training_trial.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "tasks.sqlite"
            database.write_bytes(b"not-reached")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--database",
                    str(database),
                    "--checkpoint",
                    str(root / "checkpoint.pt"),
                    "--full-model",
                    "--max-steps",
                    "1",
                ],
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "full-model training requires a frozen evaluation contract",
            completed.stderr,
        )

    def test_bounded_training_script_rejects_output_path_aliases(self):
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "run_bounded_training_trial.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "tasks.sqlite"
            database.write_bytes(b"preserve-source-database")
            checkpoint = root / "checkpoint.pt"

            cases = (
                (
                    ["--checkpoint", str(database)],
                    "checkpoint and training database must differ",
                ),
                (
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--status-output",
                        str(database),
                    ],
                    "status output and training database must differ",
                ),
                (
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--status-output",
                        str(checkpoint),
                    ],
                    "status output and checkpoint must differ",
                ),
            )
            for extra_arguments, expected_error in cases:
                with self.subTest(expected_error=expected_error):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(script),
                            "--database",
                            str(database),
                            *extra_arguments,
                            "--max-steps",
                            "1",
                        ],
                        cwd=project_root,
                        text=True,
                        encoding="utf-8",
                        capture_output=True,
                        check=False,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(expected_error, completed.stderr)
                    self.assertEqual(database.read_bytes(), b"preserve-source-database")

    def test_unspecified_cuda_index_matches_the_active_cuda_device(self):
        module = importlib.import_module("bosscrafts_model.trainer")
        self.assertTrue(
            module._device_matches(torch.device("cuda:0"), torch.device("cuda"))
        )
        self.assertFalse(
            module._device_matches(torch.device("cuda:1"), torch.device("cuda:0"))
        )

    def test_runs_bounded_loss_masked_weighted_optimizer_steps(self):
        module_name = "bosscrafts_model.trainer"
        spec = importlib.util.find_spec(module_name)
        self.assertIsNotNone(spec, "trainer module is required")
        if spec is None:
            return
        module = importlib.import_module(module_name)
        torch.manual_seed(71)
        model = EngineeringModel(ModelConfig.tiny())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.embedding.weight.detach().clone()
        examples = [
            {
                "example_id": "a" * 64,
                "input_ids": [1, 2, 3, 4, 5],
                "loss_start_index": 2,
                "sampling_weight": 0.5,
            },
            {
                "example_id": "b" * 64,
                "input_ids": [6, 7, 8, 9, 10, 11],
                "loss_start_index": 3,
                "sampling_weight": 2.0,
            },
        ]

        result = module.run_training_steps(
            model,
            optimizer,
            examples,
            device=torch.device("cpu"),
            max_steps=2,
            gradient_accumulation=1,
            step_offset=5,
            learning_rate_schedule=lambda step: 0.0001 * (step + 1),
        )

        self.assertEqual(result.optimizer_steps, 2)
        self.assertEqual(result.examples_seen, 2)
        self.assertEqual(result.sequence_tokens_seen, 11)
        self.assertEqual(result.target_tokens_seen, 6)
        self.assertEqual(result.last_example_id, "b" * 64)
        self.assertTrue(math.isfinite(result.mean_loss))
        self.assertTrue(result.gradients_finite)
        self.assertAlmostEqual(result.last_learning_rate, 0.0007)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0007)
        self.assertFalse(torch.equal(model.embedding.weight, before))

    def test_partial_accumulation_group_matches_its_true_example_mean(self):
        module = importlib.import_module("bosscrafts_model.trainer")
        config = ModelConfig.tiny()
        torch.manual_seed(211)
        reference = EngineeringModel(config)
        accumulated = EngineeringModel(config)
        accumulated.load_state_dict(reference.state_dict())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.01)
        example = {
            "example_id": "f" * 64,
            "input_ids": [1, 2, 3, 4, 5],
            "loss_start_index": 2,
            "sampling_weight": 1.0,
        }

        module.run_training_steps(
            reference,
            reference_optimizer,
            [example],
            device=torch.device("cpu"),
            max_steps=1,
            gradient_accumulation=1,
        )
        module.run_training_steps(
            accumulated,
            accumulated_optimizer,
            [example],
            device=torch.device("cpu"),
            max_steps=1,
            gradient_accumulation=8,
        )

        for expected, actual in zip(
            reference.parameters(), accumulated.parameters(), strict=True
        ):
            self.assertTrue(torch.allclose(expected, actual, atol=1e-7, rtol=1e-6))

    def test_rejects_optimizer_bound_to_a_different_model(self):
        module = importlib.import_module("bosscrafts_model.trainer")
        model = EngineeringModel(ModelConfig.tiny())
        other_model = EngineeringModel(ModelConfig.tiny())
        optimizer = torch.optim.SGD(other_model.parameters(), lr=0.01)
        example = {
            "example_id": "a" * 64,
            "input_ids": [1, 2, 3],
            "loss_start_index": 1,
            "sampling_weight": 1.0,
        }

        with self.assertRaisesRegex(ValueError, "optimizer.*model"):
            module.run_training_steps(
                model,
                optimizer,
                [example],
                device=torch.device("cpu"),
                max_steps=1,
            )

    def test_rejects_nonfinite_parameters_created_by_optimizer_step(self):
        module = importlib.import_module("bosscrafts_model.trainer")
        model = EngineeringModel(ModelConfig.tiny())

        class PoisonOptimizer(torch.optim.Optimizer):
            def __init__(self, parameters):
                super().__init__(parameters, {"lr": 0.01})

            def step(self, closure=None):
                del closure
                with torch.no_grad():
                    self.param_groups[0]["params"][0].fill_(float("inf"))

        optimizer = PoisonOptimizer(model.parameters())
        example = {
            "example_id": "b" * 64,
            "input_ids": [1, 2, 3],
            "loss_start_index": 1,
            "sampling_weight": 1.0,
        }

        with self.assertRaisesRegex(RuntimeError, "non-finite model parameter"):
            module.run_training_steps(
                model,
                optimizer,
                [example],
                device=torch.device("cpu"),
                max_steps=1,
            )

    def test_bounded_training_script_checkpoints_and_resumes_after_cursor(self):
        binding = NativeTokenizerBinding.load()
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "run_bounded_training_trial.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "tasks.sqlite"
            checkpoint = root / "checkpoint.pt"
            status_output = root / "training-progress.json"
            heldout_manifest = root / "heldout-suite-manifest.json"
            inference_output = root / "inference.json"
            heldout_inference_output = root / "heldout-inference.json"
            evaluation_output = root / "evaluation.json"
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.executemany(
                    "INSERT INTO metadata VALUES(?,?)",
                    sorted(
                        {
                            "schema_version": TASK_DATABASE_SCHEMA,
                            "binding_sha256": binding.binding_sha256,
                            "tokenizer_sha256": binding.tokenizer_sha256,
                            "vocabulary_size": str(binding.vocab_size),
                            "packing_complete": "true",
                            "maximum_sequence_tokens": "4",
                            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                        }.items()
                    ),
                )
                connection.execute(
                    """CREATE TABLE tasks (
                           example_id TEXT PRIMARY KEY,
                           document_id TEXT NOT NULL,
                           language TEXT NOT NULL,
                           family TEXT NOT NULL,
                           record_index INTEGER NOT NULL,
                           split TEXT NOT NULL,
                           input_ids_zlib BLOB NOT NULL,
                           token_count INTEGER NOT NULL,
                           loss_start_index INTEGER NOT NULL,
                           semantic_spec_sha256 TEXT NOT NULL,
                           parser_runtime_identity_sha256 TEXT NOT NULL,
                           source_sha256 TEXT NOT NULL
                       )"""
                )
                rows = []
                for document_id, token_ids in (("doc-a", [1, 2, 3, 4]), ("doc-b", [5, 6, 7, 8])):
                    input_ids_sha256 = hashlib.sha256(
                        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                    example_id = hashlib.sha256(
                        json.dumps(
                            {
                                "document_id": document_id,
                                "family": "code_to_semantic",
                                "input_ids_sha256": input_ids_sha256,
                                "record_index": 0,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest()
                    encoded = array("H", token_ids).tobytes()
                    rows.append(
                        (
                            example_id,
                            document_id,
                            "python",
                            "code_to_semantic",
                            0,
                            "train",
                            zlib.compress(encoded),
                            len(token_ids),
                            2,
                            "c" * 64,
                            PARSER_RUNTIME_IDENTITY_SHA256,
                            "d" * 64,
                        )
                    )
                connection.executemany("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                heldout_token_ids = [9, 10, 11, 12]
                heldout_input_sha256 = hashlib.sha256(
                    json.dumps(heldout_token_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                heldout_example_id = hashlib.sha256(
                    json.dumps(
                        {
                            "document_id": "doc-test",
                            "family": "code_to_semantic",
                            "input_ids_sha256": heldout_input_sha256,
                            "record_index": 0,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                connection.execute(
                    "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        heldout_example_id,
                        "doc-test",
                        "python",
                        "code_to_semantic",
                        0,
                        "test",
                        zlib.compress(array("H", heldout_token_ids).tobytes()),
                        len(heldout_token_ids),
                        2,
                        "c" * 64,
                        PARSER_RUNTIME_IDENTITY_SHA256,
                        "e" * 64,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            expected_ids = sorted(row[0] for row in rows)
            audited_identity = audit_task_database(
                binding, database
            ).database_identity_sha256
            categories: dict[str, object] = {}
            for category in ACCEPTANCE_CATEGORIES:
                example_ids = sorted(
                    hashlib.sha256(f"{category}:{index}".encode("utf-8")).hexdigest()
                    for index in range(MINIMUM_CATEGORY_EXAMPLES[category])
                )
                categories[category] = {
                    "example_ids": example_ids,
                    "example_ids_sha256": hashlib.sha256(
                        json.dumps(
                            example_ids,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            heldout_unsigned = {
                "categories": categories,
                "expected_languages": list(EXPECTED_LANGUAGES),
                "heldout_split": "test",
                "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                "schema_version": MODEL_HELDOUT_SUITE_SCHEMA,
                "structured_tasks": list(STRUCTURED_TASKS),
                "task_database_identity_sha256": audited_identity,
                "tokenizer_sha256": binding.tokenizer_sha256,
            }
            heldout_manifest_sha256 = hashlib.sha256(
                json.dumps(
                    heldout_unsigned,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            heldout_manifest.write_text(
                json.dumps(
                    {
                        **heldout_unsigned,
                        "manifest_sha256": heldout_manifest_sha256,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(script),
                "--database",
                str(database),
                "--evaluation-database",
                str(database),
                "--heldout-suite-manifest",
                str(heldout_manifest),
                "--checkpoint",
                str(checkpoint),
                "--status-output",
                str(status_output),
                "--device",
                "cpu",
                "--max-steps",
                "1",
                "--epochs",
                "2",
            ]
            first = subprocess.run(
                command,
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            first_payload = json.loads(first.stdout)
            self.assertEqual(first_payload["total_optimizer_steps"], 1)
            self.assertEqual(
                first_payload["schema_version"], "bosscrafts-training-progress-v2"
            )
            self.assertEqual(
                first_payload["checkpoint_sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            self.assertEqual(first_payload["planned_optimizer_steps"], 4)
            self.assertEqual(first_payload["optimizer_steps_remaining"], 3)
            self.assertEqual(first_payload["optimizer_step_fraction"], 0.25)
            self.assertGreater(first_payload["checkpoint_size_bytes"], 0)
            self.assertGreater(first_payload["invocation_sequence_tokens_seen"], 0)
            self.assertGreater(first_payload["invocation_target_tokens_seen"], 0)
            self.assertGreater(first_payload["training_seconds"], 0)
            self.assertGreater(first_payload["sequence_tokens_per_second"], 0)
            self.assertGreater(first_payload["target_tokens_per_second"], 0)
            self.assertGreaterEqual(first_payload["checkpoint_write_seconds"], 0)
            self.assertEqual(first_payload["last_example_id"], expected_ids[0])
            self.assertEqual(
                first_payload["task_database_identity_sha256"], audited_identity
            )
            self.assertEqual(
                first_payload["evaluation_task_database_identity_sha256"],
                audited_identity,
            )
            self.assertEqual(
                first_payload["heldout_suite_manifest_sha256"],
                heldout_manifest_sha256,
            )
            self.assertTrue(checkpoint.is_file())
            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            self.assertEqual(
                checkpoint_payload["schema_version"],
                "bosscrafts-training-checkpoint-v5",
            )
            self.assertEqual(
                checkpoint_payload["evaluation_task_database_identity_sha256"],
                audited_identity,
            )
            self.assertEqual(
                checkpoint_payload["heldout_suite_manifest_sha256"],
                heldout_manifest_sha256,
            )
            self.assertEqual(json.loads(status_output.read_text()), first_payload)
            self.assertEqual(list(root.glob(".training-progress.json.*.tmp")), [])

            changed_plan = subprocess.run(
                [*command, "--resume", "--gradient-accumulation", "2"],
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(changed_plan.returncode, 0)
            self.assertIn("training plan", changed_plan.stderr)

            resumed = subprocess.run(
                [*command, "--resume"],
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            resumed_payload = json.loads(resumed.stdout)
            self.assertEqual(resumed_payload["total_optimizer_steps"], 2)
            self.assertEqual(resumed_payload["last_example_id"], expected_ids[1])
            self.assertEqual(resumed_payload["total_examples_seen"], 2)

            next_epoch = subprocess.run(
                [*command, "--resume"],
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(next_epoch.returncode, 0, next_epoch.stderr)
            next_epoch_payload = json.loads(next_epoch.stdout)
            self.assertEqual(next_epoch_payload["epoch_index"], 1)
            self.assertEqual(next_epoch_payload["target_epochs"], 2)
            self.assertEqual(next_epoch_payload["last_example_id"], expected_ids[0])
            self.assertEqual(next_epoch_payload["total_optimizer_steps"], 3)
            self.assertEqual(next_epoch_payload["total_examples_seen"], 3)

            completed = subprocess.run(
                [*command, "--resume"],
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            completed_payload = json.loads(completed.stdout)
            self.assertEqual(completed_payload["epoch_index"], 2)
            self.assertTrue(completed_payload["training_complete"])
            self.assertEqual(completed_payload["last_example_id"], expected_ids[1])
            self.assertEqual(completed_payload["total_optimizer_steps"], 4)
            self.assertEqual(completed_payload["total_examples_seen"], 4)
            self.assertEqual(completed_payload["optimizer_steps_remaining"], 0)
            self.assertEqual(completed_payload["optimizer_step_fraction"], 1.0)
            self.assertAlmostEqual(completed_payload["learning_rate"], 0.00002)
            self.assertEqual(json.loads(status_output.read_text()), completed_payload)

            inference_command = [
                sys.executable,
                str(project_root / "scripts" / "run_checkpoint_inference.py"),
                "--database",
                str(database),
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cpu",
                "--prompt",
                "x = 1",
                "--max-new-tokens",
                "2",
                "--output",
                str(inference_output),
            ]
            inference = subprocess.run(
                inference_command,
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(inference.returncode, 0, inference.stderr)
            inference_payload = json.loads(inference.stdout)
            self.assertEqual(inference_payload["new_token_count"], 2)
            self.assertEqual(inference_payload["device"], "cpu")
            self.assertTrue(inference_payload["checkpoint_lineage_verified"])
            self.assertEqual(
                inference_payload["checkpoint_sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            self.assertTrue(inference_payload["checkpoint_training_complete"])
            self.assertEqual(
                inference_payload[
                    "checkpoint_evaluation_task_database_identity_sha256"
                ],
                audited_identity,
            )
            self.assertEqual(
                inference_payload["checkpoint_heldout_suite_manifest_sha256"],
                heldout_manifest_sha256,
            )
            self.assertEqual(json.loads(inference_output.read_text()), inference_payload)
            original_inference = inference_output.read_bytes()
            repeated_inference = subprocess.run(
                inference_command,
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(repeated_inference.returncode, 0)
            self.assertIn("already exists", repeated_inference.stderr)
            self.assertEqual(inference_output.read_bytes(), original_inference)

            heldout_inference = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "run_checkpoint_inference.py"),
                    "--database",
                    str(database),
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    "cpu",
                    "--heldout-family",
                    "code_to_semantic",
                    "--heldout-split",
                    "test",
                    "--max-new-tokens",
                    "2",
                    "--output",
                    str(heldout_inference_output),
                ],
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(heldout_inference.returncode, 0, heldout_inference.stderr)
            heldout_payload = json.loads(heldout_inference.stdout)
            self.assertEqual(heldout_payload["prompt_source"], "heldout_database")
            self.assertEqual(heldout_payload["heldout_split"], "test")
            self.assertEqual(heldout_payload["heldout_example_id"], heldout_example_id)
            self.assertEqual(heldout_payload["heldout_expected_target_token_count"], 2)
            self.assertEqual(heldout_payload["heldout_expected_token_ids"], [11, 12])
            self.assertEqual(
                heldout_payload[
                    "checkpoint_evaluation_task_database_identity_sha256"
                ],
                audited_identity,
            )
            self.assertEqual(
                heldout_payload["checkpoint_heldout_suite_manifest_sha256"],
                heldout_manifest_sha256,
            )
            self.assertEqual(
                json.loads(heldout_inference_output.read_text()), heldout_payload
            )

            evaluation_command = [
                sys.executable,
                str(project_root / "scripts" / "evaluate_checkpoint.py"),
                "--database",
                str(database),
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cpu",
                "--split",
                "train",
                "--max-examples",
                "2",
                "--output",
                str(evaluation_output),
            ]
            evaluation = subprocess.run(
                evaluation_command,
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
            evaluation_payload = json.loads(evaluation.stdout)
            self.assertEqual(evaluation_payload["example_count"], 2)
            self.assertTrue(math.isfinite(evaluation_payload["mean_loss"]))
            self.assertTrue(evaluation_payload["checkpoint_lineage_verified"])
            self.assertEqual(
                evaluation_payload["checkpoint_sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            self.assertTrue(evaluation_payload["checkpoint_training_complete"])
            self.assertEqual(
                evaluation_payload[
                    "checkpoint_evaluation_task_database_identity_sha256"
                ],
                audited_identity,
            )
            self.assertEqual(
                evaluation_payload["checkpoint_heldout_suite_manifest_sha256"],
                heldout_manifest_sha256,
            )
            self.assertTrue(evaluation_payload["evaluation_complete"])
            self.assertIn("target_token_accuracy", evaluation_payload)
            self.assertIn("baseline_target_token_accuracy", evaluation_payload)
            self.assertEqual(json.loads(evaluation_output.read_text()), evaluation_payload)
            self.assertIn("baseline_mean_loss", evaluation_payload)
            self.assertIn("relative_loss_improvement", evaluation_payload)
            self.assertTrue(math.isfinite(evaluation_payload["baseline_mean_loss"]))
            self.assertTrue(math.isfinite(evaluation_payload["relative_loss_improvement"]))
            self.assertEqual(evaluation_payload["promotion_threshold"], 0.02)
            self.assertIn("promotion_gate_passed", evaluation_payload)
            self.assertFalse(evaluation_payload["promotion_gate_passed"])
            self.assertIn(
                "full production model was not evaluated",
                evaluation_payload["promotion_blockers"],
            )
            original_evaluation = evaluation_output.read_bytes()
            repeated_evaluation = subprocess.run(
                evaluation_command,
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(repeated_evaluation.returncode, 0)
            self.assertIn("already exists", repeated_evaluation.stderr)
            self.assertEqual(evaluation_output.read_bytes(), original_evaluation)


if __name__ == "__main__":
    unittest.main()
