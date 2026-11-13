from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_full_model_acceptance_categories import CATEGORIES, _category

from bosscrafts_model.bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from bosscrafts_model.model_acceptance import verify_model_acceptance

DATABASE_ID = "a" * 64
MODEL_ID = "b" * 64
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
        "baseline_cell_mean_loss": {cell: 10.0 for cell in cells},
        "baseline_target_token_accuracy": 0.0001,
        "cell_counts": cells,
        "cell_mean_loss": {cell: 8.5 for cell in cells},
        "cell_regressions": [],
        "checkpoint_lineage_verified": True,
        "checkpoint_step": 86_016,
        "checkpoint_training_complete": True,
        "evaluation_complete": True,
        "example_count": 5_376,
        "split_example_count": 5_376,
        "family_counts": {family: 1_792 for family in FAMILIES},
        "family_regressions": [],
        "full_model": True,
        "language_counts": {language: 384 for language in CURRENT_RUNTIME_SPEC_SHA256},
        "language_regressions": [],
        "model_fingerprint": MODEL_ID,
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
        "checkpoint_lineage_verified": True,
        "checkpoint_step": 86_016,
        "checkpoint_training_complete": True,
        "full_model": True,
        "heldout_exact_match": True,
        "model_fingerprint": MODEL_ID,
        "new_token_count": 20,
        "new_token_ids": list(expected_ids),
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
        with self.assertRaisesRegex(ValueError, "promotion gate"):
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

    def test_acceptance_script_writes_hash_bound_manifest(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = {
                "validation.json": _evaluation("validation"),
                "test.json": _evaluation("test"),
                **{
                    f"{task}.json": _inference(task)
                    for task in FAMILIES
                },
                **{
                    f"{category}.json": _category(category)
                    for category in CATEGORIES
                },
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


if __name__ == "__main__":
    unittest.main()
