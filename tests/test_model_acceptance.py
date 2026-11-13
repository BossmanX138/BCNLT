from __future__ import annotations

import copy
import hashlib
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

from test_full_model_acceptance_categories import (
    CATEGORIES,
    HELDOUT_MANIFEST,
    _category,
)

from bosscrafts_model.artifact_schemas import (
    CHECKPOINT_EVALUATION_SCHEMA,
    CHECKPOINT_INFERENCE_SCHEMA,
)
from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.inference_protocol import (
    HeldoutInferenceCase,
    response_payload_token_ids,
)
from bosscrafts_model.model_acceptance import (
    EXPECTED_MODEL_FINGERPRINT,
)
from bosscrafts_model.model_acceptance import (
    verify_model_acceptance as _verify_model_acceptance,
)
from bosscrafts_model.resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding
from scripts.verify_model_acceptance import _read_payload

DATABASE_ID = "a" * 64
MODEL_ID = EXPECTED_MODEL_FINGERPRINT
CHECKPOINT_ID = "e" * 64
FAMILIES = ("code_to_semantic", "semantic_continuation", "semantic_to_code")
BINDING = NativeTokenizerBinding.load()


def _categories():
    return tuple(_category(category) for category in CATEGORIES)


def _heldout_case(task: str) -> HeldoutInferenceCase:
    binding = BINDING
    ids = binding.structural_token_ids
    prompt_ids = (ids[f"<|bcnlt:task:{task}|>"],)
    if task == "semantic_to_code":
        expected_ids = (
            ids["<|bcnlt:stream:begin|>"],
            ids["<|bcnlt:mode:raw_code|>"],
            *binding.encode_payload("x = 1\n"),
            ids["<|bcnlt:stream:end|>"],
            ids["<|bcnlt:task:end|>"],
        )
    else:
        expected_ids = (
            ids["<|bcnlt:stream:begin|>"],
            ids["<|bcnlt:mode:semantic_code|>"],
            ids["<|bcnlt:stream:end|>"],
            ids["<|bcnlt:task:end|>"],
        )
    input_ids = [*prompt_ids, *expected_ids]
    input_sha256 = hashlib.sha256(
        json.dumps(input_ids, separators=(",", ":")).encode()
    ).hexdigest()
    document_id = f"acceptance-{task}"
    example_id = hashlib.sha256(
        json.dumps(
            {
                "document_id": document_id,
                "family": task,
                "input_ids_sha256": input_sha256,
                "record_index": 0,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return HeldoutInferenceCase(
        example_id=example_id,
        family=task,
        prompt_ids=prompt_ids,
        expected_ids=expected_ids,
    )


def _heldout_cases() -> dict[str, HeldoutInferenceCase]:
    return {task: _heldout_case(task) for task in FAMILIES}


def verify_model_acceptance(
    evaluations: object,
    inferences: object,
    categories: object,
):
    return _verify_model_acceptance(
        evaluations,
        inferences,
        categories,
        binding=BINDING,
        heldout_manifest=HELDOUT_MANIFEST,
        heldout_cases=_heldout_cases(),
        task_database_identity_sha256=DATABASE_ID,
    )


def _write_ground_truth_database(path: Path) -> str:
    binding = BINDING
    connection = sqlite3.connect(path)
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
                    "maximum_sequence_tokens": "21",
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
        for task in FAMILIES:
            case = _heldout_case(task)
            document_id = f"acceptance-{task}"
            input_ids = [*case.prompt_ids, *case.expected_ids]
            connection.execute(
                "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    case.example_id,
                    document_id,
                    "python",
                    task,
                    0,
                    "test",
                    zlib.compress(array("H", input_ids).tobytes()),
                    len(input_ids),
                    len(case.prompt_ids),
                    CURRENT_RUNTIME_SPEC_SHA256["python"],
                    PARSER_RUNTIME_IDENTITY_SHA256,
                    hashlib.sha256(document_id.encode()).hexdigest(),
                ),
            )
        connection.commit()
    finally:
        connection.close()
    return audit_task_database(binding, path).database_identity_sha256


def _rebind_database(payload: dict[str, object], database_identity: str) -> None:
    for field in (
        "task_database_identity_sha256",
        "checkpoint_task_database_identity_sha256",
        "checkpoint_evaluation_task_database_identity_sha256",
        "evaluation_task_database_identity_sha256",
    ):
        if field in payload:
            payload[field] = database_identity
    supporting = payload.get("supporting_evidence")
    if type(supporting) is dict:
        tokenizer_receipt = supporting.get("tokenizer_fidelity_receipt")
        if type(tokenizer_receipt) is dict:
            tokenizer_receipt["task_database_identity_sha256"] = database_identity
            receipt_body = {
                key: value
                for key, value in tokenizer_receipt.items()
                if key != "receipt_sha256"
            }
            tokenizer_receipt["receipt_sha256"] = hashlib.sha256(
                json.dumps(
                    receipt_body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            metrics = payload.get("metrics")
            if type(metrics) is dict:
                metrics["tokenizer_fidelity_receipt_sha256"] = (
                    tokenizer_receipt["receipt_sha256"]
                )
        code_receipt = supporting.get("code_acceptance_receipt")
        if type(code_receipt) is dict:
            code_receipt["task_database_identity_sha256"] = database_identity
            receipt_body = {
                key: value
                for key, value in code_receipt.items()
                if key != "receipt_sha256"
            }
            code_receipt["receipt_sha256"] = hashlib.sha256(
                json.dumps(
                    receipt_body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            metrics = payload.get("metrics")
            if type(metrics) is dict:
                metrics["code_acceptance_receipt_sha256"] = code_receipt[
                    "receipt_sha256"
                ]
    if "artifact_sha256" in payload:
        body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
        payload["artifact_sha256"] = hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()


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
        "checkpoint_evaluation_task_database_identity_sha256": DATABASE_ID,
        "checkpoint_heldout_suite_manifest_sha256": HELDOUT_MANIFEST[
            "manifest_sha256"
        ],
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
        "promotion_blockers": [],
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
    binding = BINDING
    case = _heldout_case(task)
    prompt_ids = list(case.prompt_ids)
    expected_ids = list(case.expected_ids)
    payload_ids = response_payload_token_ids(binding, task, expected_ids)
    return {
        "schema_version": CHECKPOINT_INFERENCE_SCHEMA,
        "checkpoint_epoch_index": 1,
        "checkpoint_lineage_verified": True,
        "checkpoint_sha256": CHECKPOINT_ID,
        "checkpoint_step": 86_016,
        "checkpoint_target_epochs": 1,
        "checkpoint_task_database_identity_sha256": DATABASE_ID,
        "checkpoint_evaluation_task_database_identity_sha256": DATABASE_ID,
        "checkpoint_heldout_suite_manifest_sha256": HELDOUT_MANIFEST[
            "manifest_sha256"
        ],
        "checkpoint_training_complete": True,
        "device": "cpu",
        "dtype": "float32",
        "evaluation_task_database_identity_sha256": DATABASE_ID,
        "full_model": True,
        "full_text": binding.decode([*prompt_ids, *expected_ids]),
        "generated_protocol_text": binding.decode(expected_ids),
        "generated_text": binding.decode(payload_ids),
        "heldout_exact_match": True,
        "model_fingerprint": MODEL_ID,
        "new_token_count": len(expected_ids),
        "new_token_ids": list(expected_ids),
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "prompt_source": "heldout_database",
        "response_protocol_complete": True,
        "heldout_example_id": _heldout_case(task).example_id,
        "heldout_expected_target_token_count": len(expected_ids),
        "heldout_expected_token_ids": expected_ids,
        "heldout_split": "test",
        "heldout_target_token_accuracy": 1.0,
        "task": task,
        "task_database_identity_sha256": DATABASE_ID,
    }


class ModelAcceptanceTests(unittest.TestCase):
    def test_rejects_post_training_heldout_suite_substitution(self):
        evaluations = [_evaluation("validation"), _evaluation("test")]
        inferences = [_inference(task) for task in FAMILIES]
        substituted_suite = "f" * 64
        for payload in [*evaluations, *inferences]:
            payload["checkpoint_heldout_suite_manifest_sha256"] = substituted_suite

        with self.assertRaisesRegex(ValueError, "held-out suite lineage"):
            verify_model_acceptance(
                evaluations,
                inferences,
                _categories(),
            )

    def test_accepts_distinct_training_database_when_frozen_evaluation_matches(self):
        evaluations = [_evaluation("validation"), _evaluation("test")]
        inferences = [_inference(task) for task in FAMILIES]
        training_database_identity = "b" * 64
        for payload in [*evaluations, *inferences]:
            payload["checkpoint_task_database_identity_sha256"] = (
                training_database_identity
            )

        result = verify_model_acceptance(
            evaluations,
            inferences,
            _categories(),
        )

        self.assertTrue(result.acceptance_passed)
        self.assertEqual(result.task_database_identity_sha256, DATABASE_ID)

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

        blocked = _evaluation("validation")
        blocked["promotion_blockers"] = ["manually blocked"]
        with self.assertRaisesRegex(ValueError, "promotion gate"):
            verify_model_acceptance(
                (blocked, _evaluation("test")),
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

    def test_rejects_self_reported_expected_ids_without_database_ground_truth(self):
        cases = _heldout_cases()
        original = cases["code_to_semantic"]
        cases["code_to_semantic"] = HeldoutInferenceCase(
            example_id=original.example_id,
            family=original.family,
            prompt_ids=original.prompt_ids,
            expected_ids=(*original.expected_ids[:-1], original.expected_ids[-1] - 1),
        )
        with self.assertRaisesRegex(ValueError, "held-out"):
            _verify_model_acceptance(
                (_evaluation("validation"), _evaluation("test")),
                tuple(_inference(task) for task in FAMILIES),
                _categories(),
                binding=BINDING,
                heldout_manifest=HELDOUT_MANIFEST,
                heldout_cases=cases,
                task_database_identity_sha256=DATABASE_ID,
            )

        with self.assertRaisesRegex(ValueError, "lineage"):
            _verify_model_acceptance(
                (_evaluation("validation"), _evaluation("test")),
                tuple(_inference(task) for task in FAMILIES),
                _categories(),
                binding=BINDING,
                heldout_manifest=HELDOUT_MANIFEST,
                heldout_cases=_heldout_cases(),
                task_database_identity_sha256="f" * 64,
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

        for field, value, message in (
            (
                "prompt_token_ids",
                [int(broken["prompt_token_ids"][0]) + 1],
                "held-out",
            ),
            ("prompt_token_count", 2, "held-out"),
            ("generated_protocol_text", "fabricated", "protocol"),
            ("generated_text", "fabricated", "protocol"),
            ("full_text", "fabricated", "protocol"),
        ):
            with self.subTest(field=field):
                fabricated = _inference("code_to_semantic")
                fabricated[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    verify_model_acceptance(
                        (_evaluation("validation"), _evaluation("test")),
                        (
                            fabricated,
                            _inference("semantic_continuation"),
                            _inference("semantic_to_code"),
                        ),
                        _categories(),
                    )

        cases = _heldout_cases()
        original = cases["code_to_semantic"]
        malformed_expected = original.expected_ids[:-1]
        cases["code_to_semantic"] = HeldoutInferenceCase(
            example_id=original.example_id,
            family=original.family,
            prompt_ids=original.prompt_ids,
            expected_ids=malformed_expected,
        )
        forged = _inference("code_to_semantic")
        malformed_ids = list(malformed_expected)
        forged["new_token_count"] = len(malformed_ids)
        forged["new_token_ids"] = malformed_ids
        forged["heldout_expected_target_token_count"] = len(malformed_ids)
        forged["heldout_expected_token_ids"] = malformed_ids
        forged["generated_protocol_text"] = BINDING.decode(malformed_ids)
        forged["generated_text"] = BINDING.decode(
            response_payload_token_ids(
                BINDING, "code_to_semantic", malformed_ids
            )
        )
        forged["full_text"] = BINDING.decode(
            [*original.prompt_ids, *malformed_ids]
        )
        forged["response_protocol_complete"] = True
        with self.assertRaisesRegex(ValueError, "protocol"):
            _verify_model_acceptance(
                (_evaluation("validation"), _evaluation("test")),
                (
                    forged,
                    _inference("semantic_continuation"),
                    _inference("semantic_to_code"),
                ),
                _categories(),
                binding=BINDING,
                heldout_manifest=HELDOUT_MANIFEST,
                heldout_cases=cases,
                task_database_identity_sha256=DATABASE_ID,
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

        wrong_architecture = tuple(_category(category) for category in CATEGORIES)
        for item in wrong_architecture:
            item["model_fingerprint"] = "f" * 64
            body = {
                key: value for key, value in item.items() if key != "artifact_sha256"
            }
            item["artifact_sha256"] = hashlib.sha256(
                json.dumps(
                    body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
        wrong_evaluations = (_evaluation("validation"), _evaluation("test"))
        for item in wrong_evaluations:
            item["model_fingerprint"] = "f" * 64
        wrong_inferences = tuple(_inference(task) for task in FAMILIES)
        for item in wrong_inferences:
            item["model_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "lineage"):
            verify_model_acceptance(
                wrong_evaluations,
                wrong_inferences,
                wrong_architecture,
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
        self.assertIn("--task-database", completed.stdout)
        self.assertIn("--heldout-manifest", completed.stdout)
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
                handle.seek(16)
                handle.write(b"{}")
            with self.assertRaisesRegex(ValueError, "size limit"):
                _read_payload(oversized, maximum_bytes=16)

    def test_acceptance_script_writes_hash_bound_manifest(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "tasks.sqlite"
            database_identity = _write_ground_truth_database(database)
            heldout_manifest = copy.deepcopy(HELDOUT_MANIFEST)
            heldout_manifest["task_database_identity_sha256"] = database_identity
            heldout_unsigned = {
                key: value
                for key, value in heldout_manifest.items()
                if key != "manifest_sha256"
            }
            heldout_manifest["manifest_sha256"] = hashlib.sha256(
                json.dumps(
                    heldout_unsigned,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            evidence = {
                "validation.json": _evaluation("validation"),
                "test.json": _evaluation("test"),
                **{f"{task}.json": _inference(task) for task in FAMILIES},
                **{f"{category}.json": _category(category) for category in CATEGORIES},
            }
            for payload in evidence.values():
                if "heldout_manifest_sha256" in payload:
                    payload["heldout_manifest_sha256"] = heldout_manifest[
                        "manifest_sha256"
                    ]
                if "checkpoint_heldout_suite_manifest_sha256" in payload:
                    payload["checkpoint_heldout_suite_manifest_sha256"] = (
                        heldout_manifest["manifest_sha256"]
                    )
                _rebind_database(payload, database_identity)
            heldout_path = root / "heldout-manifest.json"
            heldout_path.write_text(json.dumps(heldout_manifest), encoding="utf-8")
            for name, payload in evidence.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            output = root / "acceptance.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "verify_model_acceptance.py"),
                    "--task-database",
                    str(database),
                    "--heldout-manifest",
                    str(heldout_path),
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
