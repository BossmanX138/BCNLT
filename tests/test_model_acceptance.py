from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_full_model_acceptance_categories import CATEGORIES, _category

from bosscrafts_model.artifact_schemas import (
    CHECKPOINT_EVALUATION_SCHEMA,
    CHECKPOINT_INFERENCE_SCHEMA,
)
from bosscrafts_model.bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from bosscrafts_model.model_acceptance import verify_model_acceptance
from scripts.verify_model_acceptance import _read_payload

DATABASE_ID = "a" * 64
MODEL_ID = "b" * 64
CHECKPOINT_ID = "e" * 64
FAMILIES = ("code_to_semantic", "semantic_continuation", "semantic_to_code")


def _categories():
    return tuple(_category(category) for category in CATEGORIES)


def _evaluation(split: str) -> dict[str, object]:
    cells = {
        f"{language}|{family}": 128
        for language in sorted(CURRENT_RUNTIME_SPEC_SHA256)
        for family in FAMILIES
    }
    return {
        "schema_version": CHECKPOINT_EVALUATION_SCHEMA,
        "baseline_correct_target_token_count": 10,
        "baseline_evaluation_sha256": None,
        "baseline_exact_match_count": 0,
        "baseline_exact_match_rate": 0.0,
        "baseline_family_mean_loss": {family: 10.0 for family in FAMILIES},
        "baseline_language_mean_loss": {
            language: 10.0 for language in CURRENT_RUNTIME_SPEC_SHA256
        },
        "baseline_mean_loss": 10.0,
        "baseline_perplexity": math.exp(10.0),
        "baseline_source": "deterministic_random_initialization",
        "baseline_cell_mean_loss": {cell: 10.0 for cell in cells},
        "baseline_target_token_accuracy": 0.0001,
        "baseline_target_token_count": 100_000,
        "cell_counts": cells,
        "cell_mean_loss": {cell: 8.5 for cell in cells},
        "cell_regressions": [],
        "checkpoint_epoch_index": 1,
        "checkpoint_lineage_verified": True,
        "checkpoint_sha256": CHECKPOINT_ID,
        "checkpoint_step": 86_016,
        "checkpoint_target_epochs": 1,
        "checkpoint_task_database_identity_sha256": DATABASE_ID,
        "checkpoint_training_complete": True,
        "correct_target_token_count": 42_000,
        "device": "cpu",
        "dtype": "float32",
        "evaluation_complete": True,
        "evaluated_fraction": 1.0,
        "evaluation_task_database_identity_sha256": DATABASE_ID,
        "example_count": 5_376,
        "exact_match_count": 0,
        "exact_match_rate": 0.0,
        "split_example_count": 5_376,
        "family_counts": {family: 1_792 for family in FAMILIES},
        "family_mean_loss": {family: 8.5 for family in FAMILIES},
        "family_regressions": [],
        "full_model": True,
        "language_counts": {language: 384 for language in CURRENT_RUNTIME_SPEC_SHA256},
        "language_mean_loss": {
            language: 8.5 for language in CURRENT_RUNTIME_SPEC_SHA256
        },
        "language_regressions": [],
        "mean_loss": 8.5,
        "model_fingerprint": MODEL_ID,
        "perplexity": math.exp(8.5),
        "promotion_gate_passed": True,
        "promotion_threshold": 0.02,
        "relative_loss_improvement": 0.15,
        "split": split,
        "target_token_accuracy": 0.42,
        "target_token_accuracy_improved": True,
        "target_token_count": 100_000,
        "task_database_identity_sha256": DATABASE_ID,
    }


def _inference(task: str) -> dict[str, object]:
    expected_ids = list(range(1, 21))
    return {
        "schema_version": CHECKPOINT_INFERENCE_SCHEMA,
        "checkpoint_epoch_index": 1,
        "checkpoint_lineage_verified": True,
        "checkpoint_sha256": CHECKPOINT_ID,
        "checkpoint_step": 86_016,
        "checkpoint_target_epochs": 1,
        "checkpoint_task_database_identity_sha256": DATABASE_ID,
        "checkpoint_training_complete": True,
        "device": "cpu",
        "dtype": "float32",
        "evaluation_task_database_identity_sha256": DATABASE_ID,
        "full_model": True,
        "full_text": "",
        "generated_protocol_text": "",
        "generated_text": "",
        "heldout_exact_match": True,
        "model_fingerprint": MODEL_ID,
        "new_token_count": 20,
        "new_token_ids": list(expected_ids),
        "prompt_token_count": 1,
        "prompt_source": "heldout_database",
        "response_protocol_complete": True,
        "heldout_example_id": "d" * 64,
        "heldout_expected_target_token_count": 20,
        "heldout_expected_token_ids": expected_ids,
        "heldout_split": "test",
        "heldout_target_token_accuracy": 1.0,
        "task": task,
        "task_database_identity_sha256": DATABASE_ID,
    }


class ModelAcceptanceTests(unittest.TestCase):
    def test_accepts_complete_heldout_and_protocol_evidence(self):
        result = verify_model_acceptance(
            (_evaluation("validation"), _evaluation("test")),
            tuple(_inference(task) for task in FAMILIES),
            _categories(),
        )
        self.assertTrue(result.acceptance_passed)
        self.assertEqual(result.checkpoint_sha256, CHECKPOINT_ID)
        self.assertEqual(result.checkpoint_step, 86_016)
        self.assertEqual(result.task_database_identity_sha256, DATABASE_ID)
        self.assertEqual(set(result.evaluation_sha256), {"validation", "test"})
        self.assertEqual(set(result.inference_sha256), set(FAMILIES))

    def test_rejects_incomplete_or_non_improving_evaluation(self):
        incomplete = _evaluation("validation")
        incomplete["evaluation_complete"] = False
        with self.assertRaisesRegex(ValueError, "complete split"):
            verify_model_acceptance(
                (incomplete, _evaluation("test")),
                tuple(_inference(task) for task in FAMILIES),
                _categories(),
            )

        missing_cell = _evaluation("validation")
        missing_cell["cell_counts"].pop("python|code_to_semantic")
        with self.assertRaisesRegex(ValueError, "language-family cell"):
            verify_model_acceptance(
                (missing_cell, _evaluation("test")),
                tuple(_inference(task) for task in FAMILIES),
                _categories(),
            )

        cell_regressed = _evaluation("validation")
        cell_regressed["cell_regressions"] = ["python|code_to_semantic"]
        with self.assertRaisesRegex(ValueError, "promotion gate|regression"):
            verify_model_acceptance(
                (cell_regressed, _evaluation("test")),
                tuple(_inference(task) for task in FAMILIES),
                _categories(),
            )

        regressed = _evaluation("validation")
        regressed["target_token_accuracy_improved"] = False
        regressed["target_token_accuracy"] = 0.0001
        with self.assertRaisesRegex(ValueError, "accuracy"):
            verify_model_acceptance(
                (regressed, _evaluation("test")),
                tuple(_inference(task) for task in FAMILIES),
                _categories(),
            )

    def test_rejects_self_reported_improvement_that_conflicts_with_cell_losses(self):
        fabricated = _evaluation("validation")
        fabricated["cell_mean_loss"] = {
            cell: 11.0 for cell in fabricated["cell_mean_loss"]
        }
        with self.assertRaisesRegex(ValueError, "loss|regression|consistency"):
            verify_model_acceptance(
                (fabricated, _evaluation("test")),
                tuple(_inference(task) for task in FAMILIES),
                _categories(),
            )

    def test_rejects_internally_inconsistent_evaluation_artifacts(self):
        mutations = (
            ("relative_loss_improvement", 0.99, "loss consistency"),
            ("correct_target_token_count", 41_999, "accuracy consistency"),
            ("baseline_target_token_count", 99_999, "accuracy consistency"),
            ("mean_loss", 1_000.0, "loss consistency"),
            (
                "checkpoint_task_database_identity_sha256",
                "f" * 64,
                "database lineage",
            ),
            ("schema_version", "legacy", "schema"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                invalid = _evaluation("validation")
                invalid[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    verify_model_acceptance(
                        (invalid, _evaluation("test")),
                        tuple(_inference(task) for task in FAMILIES),
                        _categories(),
                    )

        unexpected = _evaluation("validation")
        unexpected["unverified_override"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            verify_model_acceptance(
                (unexpected, _evaluation("test")),
                tuple(_inference(task) for task in FAMILIES),
                _categories(),
            )

    def test_rejects_inference_artifact_lineage_or_schema_bypass(self):
        invalid = _inference("code_to_semantic")
        invalid["evaluation_task_database_identity_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "database lineage"):
            verify_model_acceptance(
                (_evaluation("validation"), _evaluation("test")),
                (
                    invalid,
                    _inference("semantic_continuation"),
                    _inference("semantic_to_code"),
                ),
                _categories(),
            )

        unexpected = _inference("code_to_semantic")
        unexpected["manual_override"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            verify_model_acceptance(
                (_evaluation("validation"), _evaluation("test")),
                (
                    unexpected,
                    _inference("semantic_continuation"),
                    _inference("semantic_to_code"),
                ),
                _categories(),
            )

    def test_rejects_incomplete_protocol_or_lineage_mismatch(self):
        broken = _inference("code_to_semantic")
        broken["response_protocol_complete"] = False
        with self.assertRaisesRegex(ValueError, "protocol"):
            verify_model_acceptance(
                (_evaluation("validation"), _evaluation("test")),
                (
                    broken,
                    _inference("semantic_continuation"),
                    _inference("semantic_to_code"),
                ),
                _categories(),
            )

        mismatched = copy.deepcopy(_evaluation("test"))
        mismatched["task_database_identity_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "lineage"):
            verify_model_acceptance(
                (_evaluation("validation"), mismatched),
                tuple(_inference(task) for task in FAMILIES),
                _categories(),
            )

        mixed_checkpoint = _inference("code_to_semantic")
        mixed_checkpoint["checkpoint_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "lineage"):
            verify_model_acceptance(
                (_evaluation("validation"), _evaluation("test")),
                (
                    mixed_checkpoint,
                    _inference("semantic_continuation"),
                    _inference("semantic_to_code"),
                ),
                _categories(),
            )

        manual = _inference("code_to_semantic")
        manual["prompt_source"] = "manual"
        with self.assertRaisesRegex(ValueError, "held-out"):
            verify_model_acceptance(
                (_evaluation("validation"), _evaluation("test")),
                (
                    manual,
                    _inference("semantic_continuation"),
                    _inference("semantic_to_code"),
                ),
                _categories(),
            )

        inexact = _inference("code_to_semantic")
        inexact["new_token_ids"][-1] = 21
        inexact["heldout_exact_match"] = False
        inexact["heldout_target_token_accuracy"] = 0.95
        with self.assertRaisesRegex(ValueError, "held-out"):
            verify_model_acceptance(
                (_evaluation("validation"), _evaluation("test")),
                (
                    inexact,
                    _inference("semantic_continuation"),
                    _inference("semantic_to_code"),
                ),
                _categories(),
            )

    def test_acceptance_script_runs_from_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "verify_model_acceptance.py"),
                "--help",
            ],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--validation-evaluation", completed.stdout)
        self.assertIn("--code-to-semantic-inference", completed.stdout)
        self.assertIn("--tokenizer-fidelity-evidence", completed.stdout)
        self.assertIn("--resources-evidence", completed.stdout)

    def test_acceptance_reader_rejects_duplicate_keys_and_oversized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"split":"test","split":"validation"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _read_payload(duplicate)

            oversized = root / "oversized.json"
            with oversized.open("wb") as handle:
                handle.seek(16 * 1024 * 1024)
                handle.write(b"{}")
            with self.assertRaisesRegex(ValueError, "size limit"):
                _read_payload(oversized)

    def test_acceptance_script_writes_hash_bound_manifest(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = {
                "validation.json": _evaluation("validation"),
                "test.json": _evaluation("test"),
                **{f"{task}.json": _inference(task) for task in FAMILIES},
                **{f"{category}.json": _category(category) for category in CATEGORIES},
            }
            for name, payload in evidence.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            output = root / "acceptance.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "verify_model_acceptance.py"),
                    "--validation-evaluation",
                    str(root / "validation.json"),
                    "--test-evaluation",
                    str(root / "test.json"),
                    "--code-to-semantic-inference",
                    str(root / "code_to_semantic.json"),
                    "--semantic-continuation-inference",
                    str(root / "semantic_continuation.json"),
                    "--semantic-to-code-inference",
                    str(root / "semantic_to_code.json"),
                    *[
                        value
                        for category in CATEGORIES
                        for value in (
                            f"--{category.replace('_', '-')}-evidence",
                            str(root / f"{category}.json"),
                        )
                    ],
                    "--output",
                    str(output),
                ],
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["acceptance_passed"])
            self.assertRegex(payload["acceptance_manifest_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(payload, json.loads(completed.stdout))
            original = output.read_bytes()
            repeated = subprocess.run(
                completed.args,
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("output file already exists", repeated.stderr)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
