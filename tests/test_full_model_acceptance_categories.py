from __future__ import annotations

import copy
import hashlib
import json
import unittest

from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.model_acceptance import verify_model_acceptance

DATABASE_ID = "a" * 64
MODEL_ID = "b" * 64
TOKENIZER_ID = "c" * 64
HELDOUT_ID = "d" * 64
FAMILIES = ("code_to_semantic", "semantic_continuation", "semantic_to_code")
LANGUAGES = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))
CATEGORIES = (
    "tokenizer_fidelity",
    "semantic_fidelity",
    "raw_code",
    "ordinary_instruction",
    "execution_correctness",
    "safety",
    "regression",
    "resources",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _evaluation(split: str) -> dict[str, object]:
    cells = {
        f"{language}|{family}": 128
        for language in LANGUAGES
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
        "language_counts": {language: 384 for language in LANGUAGES},
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
        "heldout_example_id": "e" * 64,
        "heldout_expected_target_token_count": 20,
        "heldout_expected_token_ids": expected_ids,
        "heldout_split": "test",
        "heldout_target_token_accuracy": 1.0,
        "task": task,
        "task_database_identity_sha256": DATABASE_ID,
    }


def _metrics(category: str) -> dict[str, object]:
    language_passes = {language: True for language in LANGUAGES}
    return {
        "tokenizer_fidelity": {
            "exact_roundtrip_rate": 1.0,
            "id_stability_rate": 1.0,
            "reserved_control_violation_count": 0,
        },
        "semantic_fidelity": {
            "exact_semantic_record_rate": 0.99,
            "ownership_accuracy": 0.99,
            "unknown_semantic_rate": 0.01,
            "language_passes": language_passes,
            "strict_uncertainty_acceptance_passed": True,
            "strict_uncertainty_receipt_sha256": "9" * 64,
        },
        "raw_code": {
            "exact_roundtrip_rate": 1.0,
            "compile_pass_rate": 0.98,
            "silent_fallback_count": 0,
            "language_passes": language_passes,
        },
        "ordinary_instruction": {
            "task_success_rate": 0.85,
            "baseline_delta": 0.05,
            "format_adherence_rate": 0.98,
            "false_refusal_rate": 0.02,
        },
        "execution_correctness": {
            "sandbox_execution_rate": 0.94,
            "expected_output_exact_rate": 0.9,
            "compile_pass_rate": 0.98,
            "sandbox_escape_count": 0,
            "language_passes": language_passes,
        },
        "safety": {
            "security_test_pass_rate": 1.0,
            "sandbox_escape_count": 0,
            "network_violation_count": 0,
            "secret_leakage_count": 0,
            "unauthorized_file_write_count": 0,
        },
        "regression": {
            "full_regression_passed": True,
            "test_count": 248,
            "subtest_count": 133,
            "failure_count": 0,
        },
        "resources": {
            "peak_vram_bytes": 8 * 1024**3,
            "vram_limit_bytes": 12 * 1024**3,
            "peak_ram_bytes": 16 * 1024**3,
            "ram_limit_bytes": 64 * 1024**3,
            "oom_count": 0,
            "inference_tokens_per_second": 5.0,
        },
    }[category]


def _category(category: str) -> dict[str, object]:
    minimum_examples = {
        "tokenizer_fidelity": 1_024,
        "semantic_fidelity": 1_792,
        "raw_code": 1_400,
        "ordinary_instruction": 512,
        "execution_correctness": 512,
        "safety": 256,
        "regression": 1,
        "resources": 1,
    }
    payload: dict[str, object] = {
        "category": category,
        "checkpoint_step": 86_016,
        "evidence_complete": True,
        "example_count": minimum_examples[category],
        "failure_codes": [],
        "heldout_manifest_sha256": HELDOUT_ID,
        "heldout_split": "test",
        "metrics": _metrics(category),
        "model_fingerprint": MODEL_ID,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "passed": True,
        "task_database_identity_sha256": DATABASE_ID,
        "tokenizer_sha256": TOKENIZER_ID,
    }
    payload["artifact_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


class FullModelAcceptanceCategoryTests(unittest.TestCase):
    def _verify(self, categories):
        return verify_model_acceptance(
            (_evaluation("validation"), _evaluation("test")),
            tuple(_inference(task) for task in FAMILIES),
            categories,
        )

    def test_acceptance_requires_all_independently_hashed_categories(self):
        result = self._verify(tuple(_category(category) for category in CATEGORIES))
        self.assertTrue(result.acceptance_passed)
        self.assertEqual(set(result.category_evidence_sha256), set(CATEGORIES))
        self.assertEqual(result.parser_runtime_identity_sha256, PARSER_RUNTIME_IDENTITY_SHA256)
        self.assertEqual(result.tokenizer_sha256, TOKENIZER_ID)

    def test_missing_or_failed_category_cannot_be_hidden_by_loss_improvement(self):
        evidence = [_category(category) for category in CATEGORIES]
        with self.assertRaisesRegex(ValueError, "every independent category"):
            self._verify(tuple(evidence[:-1]))

        weak = copy.deepcopy(evidence)
        instruction = next(
            item for item in weak if item["category"] == "ordinary_instruction"
        )
        instruction["metrics"]["task_success_rate"] = 0.79
        body = {key: value for key, value in instruction.items() if key != "artifact_sha256"}
        instruction["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        with self.assertRaisesRegex(ValueError, "ordinary_instruction"):
            self._verify(tuple(weak))

    def test_category_hash_and_lineage_are_fail_closed(self):
        evidence = [_category(category) for category in CATEGORIES]
        tampered = copy.deepcopy(evidence)
        tampered[0]["metrics"]["exact_roundtrip_rate"] = 0.0
        with self.assertRaisesRegex(ValueError, "integrity"):
            self._verify(tuple(tampered))

        mismatched = copy.deepcopy(evidence)
        mismatched[-1]["model_fingerprint"] = "f" * 64
        body = {
            key: value
            for key, value in mismatched[-1].items()
            if key != "artifact_sha256"
        }
        mismatched[-1]["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        with self.assertRaisesRegex(ValueError, "lineage"):
            self._verify(tuple(mismatched))

    def test_semantic_fidelity_requires_strict_uncertainty_receipt(self):
        evidence = [_category(category) for category in CATEGORIES]
        semantic = next(
            item for item in evidence if item["category"] == "semantic_fidelity"
        )
        semantic["metrics"]["strict_uncertainty_acceptance_passed"] = False
        body = {key: value for key, value in semantic.items() if key != "artifact_sha256"}
        semantic["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        with self.assertRaisesRegex(ValueError, "semantic_fidelity"):
            self._verify(tuple(evidence))


if __name__ == "__main__":
    unittest.main()
