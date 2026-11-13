from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from dataclasses import replace
from functools import lru_cache

from test_code_acceptance import _execution as _code_execution
from test_containment_acceptance import _FakeSandbox
from test_regression_acceptance import _record as _regression_record

from bosscrafts_model.artifact_schemas import (
    CHECKPOINT_EVALUATION_SCHEMA,
    CHECKPOINT_INFERENCE_SCHEMA,
    MODEL_CATEGORY_EVIDENCE_SCHEMA,
    MODEL_HELDOUT_SUITE_SCHEMA,
)
from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.code_acceptance import (
    CodeAcceptanceBatchResult,
    build_code_acceptance_receipt,
)
from bosscrafts_model.containment_acceptance import (
    build_containment_receipt,
    run_containment_probes,
)
from bosscrafts_model.ifeval_case_runner import (
    IfevalBatchResult,
    build_ifeval_batch_receipt,
)
from bosscrafts_model.ifeval_runtime import EXPECTED_IFEVAL_RUNTIME_MANIFEST_SHA256
from bosscrafts_model.inference_protocol import (
    HeldoutInferenceCase,
    response_payload_token_ids,
)
from bosscrafts_model.model_acceptance import (
    EXPECTED_MODEL_FINGERPRINT,
    verify_heldout_suite_manifest,
)
from bosscrafts_model.model_acceptance import (
    verify_model_acceptance as _verify_model_acceptance,
)
from bosscrafts_model.regression_acceptance import (
    RegressionResult,
    build_regression_receipt,
)
from bosscrafts_model.safety_judge_case_runner import (
    SafetyJudgeBatchResult,
    build_safety_content_receipt,
)
from bosscrafts_model.safety_judge_runtime import (
    EXPECTED_SAFETY_JUDGE_RUNTIME_MANIFEST_SHA256,
)
from bosscrafts_model.strict_uncertainty_acceptance import (
    StrictUncertaintyAcceptanceReceipt,
)
from bosscrafts_model.tokenizer import (
    EXPECTED_TOKENIZER_SHA256,
    NativeTokenizerBinding,
)
from bosscrafts_model.tokenizer_fidelity_acceptance import (
    TokenizerFidelityBatchResult,
    build_tokenizer_fidelity_receipt,
)

DATABASE_ID = "a" * 64
MODEL_ID = EXPECTED_MODEL_FINGERPRINT
TOKENIZER_ID = EXPECTED_TOKENIZER_SHA256
CHECKPOINT_ID = "e" * 64
FAMILIES = ("code_to_semantic", "semantic_continuation", "semantic_to_code")
LANGUAGES = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))
BINDING = NativeTokenizerBinding.load()
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
CATEGORY_MINIMUMS = {
    "tokenizer_fidelity": 1_024,
    "semantic_fidelity": 1_792,
    "raw_code": 1_400,
    "ordinary_instruction": 512,
    "execution_correctness": 512,
    "safety": 256,
    "regression": 1,
    "resources": 1,
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _heldout_suite_manifest() -> dict[str, object]:
    categories: dict[str, object] = {}
    for category in CATEGORIES:
        example_ids = sorted(
            hashlib.sha256(f"{category}:{index}".encode()).hexdigest()
            for index in range(CATEGORY_MINIMUMS[category])
        )
        categories[category] = {
            "example_ids": example_ids,
            "example_ids_sha256": hashlib.sha256(_canonical(example_ids)).hexdigest(),
        }
    unsigned: dict[str, object] = {
        "schema_version": MODEL_HELDOUT_SUITE_SCHEMA,
        "task_database_identity_sha256": DATABASE_ID,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "tokenizer_sha256": TOKENIZER_ID,
        "heldout_split": "test",
        "expected_languages": list(LANGUAGES),
        "structured_tasks": list(FAMILIES),
        "categories": categories,
    }
    return {
        **unsigned,
        "manifest_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


HELDOUT_MANIFEST = _heldout_suite_manifest()
HELDOUT_SUITE = verify_heldout_suite_manifest(
    HELDOUT_MANIFEST,
    task_database_identity_sha256=DATABASE_ID,
)
HELDOUT_ID = HELDOUT_SUITE.manifest_sha256


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
    return HeldoutInferenceCase(
        example_id=hashlib.sha256(f"acceptance:{task}".encode()).hexdigest(),
        family=task,
        prompt_ids=prompt_ids,
        expected_ids=expected_ids,
    )


def _heldout_cases() -> dict[str, HeldoutInferenceCase]:
    return {task: _heldout_case(task) for task in FAMILIES}


def verify_model_acceptance(evaluations: object, inferences: object, categories: object):
    return _verify_model_acceptance(
        evaluations,
        inferences,
        categories,
        binding=BINDING,
        heldout_manifest=HELDOUT_MANIFEST,
        heldout_cases=_heldout_cases(),
        task_database_identity_sha256=DATABASE_ID,
    )


def _evaluation(split: str) -> dict[str, object]:
    cells = {
        f"{language}|{family}": 128 for language in LANGUAGES for family in FAMILIES
    }
    return {
        "schema_version": CHECKPOINT_EVALUATION_SCHEMA,
        "baseline_correct_target_token_count": 10,
        "baseline_evaluation_sha256": None,
        "baseline_exact_match_count": 0,
        "baseline_exact_match_rate": 0.0,
        "baseline_family_mean_loss": {family: 10.0 for family in FAMILIES},
        "baseline_language_mean_loss": {language: 10.0 for language in LANGUAGES},
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
        "checkpoint_heldout_suite_manifest_sha256": HELDOUT_ID,
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
        "language_counts": {language: 384 for language in LANGUAGES},
        "language_mean_loss": {language: 8.5 for language in LANGUAGES},
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
        "checkpoint_heldout_suite_manifest_sha256": HELDOUT_ID,
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


def _strict_receipt_payload() -> dict[str, object]:
    policy = {
        "minimum_global_selective_coverage_at_025": 0.5,
        "maximum_global_selective_error_rate_at_025": 0.1,
        "minimum_per_language_selective_coverage_at_025": 0.5,
        "maximum_per_language_selective_error_rate_at_025": 0.1,
    }
    metric = {
        "sample_count": 1,
        "brier": 0.1,
        "ece": 0.1,
        "selective_coverage_at_025": 1.0,
        "selective_error_rate_at_025": 0.0,
    }
    provisional = StrictUncertaintyAcceptanceReceipt(
        schema_version="bosscrafts-strict-uncertainty-acceptance-v1",
        calibration_artifact_sha256="8" * 64,
        parser_runtime_identity_sha256=PARSER_RUNTIME_IDENTITY_SHA256,
        policy=policy,
        global_metrics={**metric, "sample_count": len(LANGUAGES)},
        per_language_metrics={language: dict(metric) for language in LANGUAGES},
        upstream_promoted=True,
        passed=True,
        failure_codes=(),
        receipt_sha256="",
    )
    receipt = replace(
        provisional,
        receipt_sha256=provisional.compute_sha256(),
    )
    return receipt.to_dict()


def _metrics(category: str) -> dict[str, object]:
    language_passes = {language: True for language in LANGUAGES}
    strict_receipt = _strict_receipt_payload()
    safety_content = _safety_content_receipt_payload()
    containment = _containment_receipt_payload()
    regression = _regression_receipt_payload()
    tokenizer_fidelity = _tokenizer_fidelity_receipt_payload()
    raw_code_receipt = _code_receipt_payload("raw_code")
    execution_receipt = _code_receipt_payload("execution_correctness")
    return {
        "tokenizer_fidelity": {
            "exact_roundtrip_rate": 1.0,
            "id_stability_rate": 1.0,
            "reserved_control_violation_count": 0,
            "tokenizer_fidelity_receipt_sha256": tokenizer_fidelity[
                "receipt_sha256"
            ],
        },
        "semantic_fidelity": {
            "exact_semantic_record_rate": 1.0,
            "ownership_accuracy": 1.0,
            "unknown_semantic_rate": 0.0,
            "language_passes": language_passes,
            "strict_uncertainty_acceptance_passed": True,
            "strict_uncertainty_receipt_sha256": strict_receipt["receipt_sha256"],
        },
        "raw_code": {
            "exact_roundtrip_rate": 1.0,
            "compile_pass_rate": 1.0,
            "silent_fallback_count": 0,
            "language_passes": language_passes,
            "code_acceptance_receipt_sha256": raw_code_receipt["receipt_sha256"],
        },
        "ordinary_instruction": {
            "strict_prompt_accuracy": 1.0,
            "loose_prompt_accuracy": 1.0,
            "baseline_delta": 1.0,
            "empty_response_rate": 0.0,
        },
        "execution_correctness": {
            "sandbox_execution_rate": 1.0,
            "expected_output_exact_rate": 1.0,
            "compile_pass_rate": 1.0,
            "sandbox_escape_count": 0,
            "language_passes": language_passes,
            "code_acceptance_receipt_sha256": execution_receipt["receipt_sha256"],
        },
        "safety": {
            "prompt_moderation_accuracy": 1.0,
            "response_safety_rate": 1.0,
            "empty_response_rate": 0.0,
            "safety_content_receipt_sha256": safety_content["receipt_sha256"],
            "containment_acceptance_passed": True,
            "containment_receipt_sha256": containment["receipt_sha256"],
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
            "regression_receipt_sha256": regression["receipt_sha256"],
            "regression_git_commit": regression["git_commit"],
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


def _records(category: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, example_id in enumerate(
        HELDOUT_SUITE.category_example_ids[category]
    ):
        language = LANGUAGES[index % len(LANGUAGES)]
        record: dict[str, object] = {
            "tokenizer_fidelity": {
                "example_id": example_id,
                "exact_roundtrip": True,
                "id_stable": True,
                "reserved_control_violation": False,
            },
            "semantic_fidelity": {
                "example_id": example_id,
                "language": language,
                "exact_semantic_record": True,
                "ownership_correct": True,
                "unknown_semantic": False,
            },
            "raw_code": {
                "example_id": example_id,
                "language": language,
                "exact_roundtrip": True,
                "compile_passed": True,
                "silent_fallback": False,
            },
            "ordinary_instruction": {
                "example_id": example_id,
                "strict_instruction_following": True,
                "loose_instruction_following": True,
                "baseline_strict_instruction_following": False,
                "response_nonempty": True,
            },
            "execution_correctness": {
                "example_id": example_id,
                "language": language,
                "sandbox_executed": True,
                "expected_output_exact": True,
                "compile_passed": True,
                "sandbox_escape": False,
            },
            "safety": {
                "example_id": example_id,
                "prompt_moderation_correct": True,
                "response_safe": True,
                "response_nonempty": True,
            },
            "regression": {
                "example_id": example_id,
                "full_regression_passed": True,
                "test_count": 248,
                "subtest_count": 133,
                "failure_count": 0,
            },
            "resources": {
                "example_id": example_id,
                "peak_vram_bytes": 8 * 1024**3,
                "vram_limit_bytes": 12 * 1024**3,
                "peak_ram_bytes": 16 * 1024**3,
                "ram_limit_bytes": 64 * 1024**3,
                "oom_count": 0,
                "inference_tokens_per_second": 5.0,
            },
        }[category]
        records.append(record)
    return records


def _safety_content_receipt_payload() -> dict[str, object]:
    records = _records("safety")
    example_ids = tuple(str(record["example_id"]) for record in records)
    raw_output = "User Safety: safe\nResponse Safety: safe"
    details = tuple(
        {
            "example_id": example_id,
            "expected_prompt_label": "safe",
            "observed_prompt_label": "safe",
            "observed_response_label": "safe",
            "safety_categories": [],
            "raw_output": raw_output,
            "raw_output_sha256": hashlib.sha256(
                raw_output.encode("utf-8")
            ).hexdigest(),
        }
        for example_id in example_ids
    )
    responses = {example_id: "safe candidate" for example_id in example_ids}
    return build_safety_content_receipt(
        SafetyJudgeBatchResult(
            runtime_manifest_sha256=(
                EXPECTED_SAFETY_JUDGE_RUNTIME_MANIFEST_SHA256
            ),
            case_example_ids=example_ids,
            records=tuple(records),
            details=details,
        ),
        responses,
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
    )


def _containment_receipt_payload() -> dict[str, object]:
    return build_containment_receipt(
        run_containment_probes(_FakeSandbox()),
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
    )


def _ifeval_receipt_payload() -> dict[str, object]:
    records = _records("ordinary_instruction")
    example_ids = tuple(str(record["example_id"]) for record in records)
    details = tuple(
        {
            "example_id": example_id,
            "candidate_strict": True,
            "candidate_loose": True,
            "baseline_strict": False,
            "candidate_strict_instruction_results": [True],
            "candidate_loose_instruction_results": [True],
            "baseline_strict_instruction_results": [False],
        }
        for example_id in example_ids
    )
    return build_ifeval_batch_receipt(
        IfevalBatchResult(
            runtime_manifest_sha256=EXPECTED_IFEVAL_RUNTIME_MANIFEST_SHA256,
            case_example_ids=example_ids,
            records=tuple(records),
            details=details,
        ),
        {example_id: "candidate" for example_id in example_ids},
        {example_id: "" for example_id in example_ids},
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
        baseline_source="deterministic_random_initialization",
        baseline_evaluation_sha256=None,
    )


def _regression_receipt_payload() -> dict[str, object]:
    records = tuple(
        sorted(
            (
                *(
                    _regression_record("test", f"tests/test_{index}.py::test_case")
                    for index in range(248)
                ),
                *(
                    _regression_record(
                        "subtest",
                        f"tests/test_{index}.py::test_case",
                        context=f"value={index}",
                    )
                    for index in range(133)
                ),
            ),
            key=lambda record: str(record["record_id"]),
        )
    )
    result = RegressionResult(
        git_commit="a" * 40,
        python_version="3.13.9 fixture",
        pytest_version="9.1.1",
        returncode=0,
        records=records,
        stdout_sha256="b" * 64,
        stderr_sha256="c" * 64,
    )
    return build_regression_receipt(
        result,
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
        example_id=HELDOUT_SUITE.category_example_ids["regression"][0],
    )


def _tokenizer_fidelity_receipt_payload() -> dict[str, object]:
    records = tuple(_records("tokenizer_fidelity"))
    details = tuple(
        {
            **record,
            "task_example_id": hashlib.sha256(
                f"tokenizer-task:{record['example_id']}".encode()
            ).hexdigest(),
            "language": LANGUAGES[index % len(LANGUAGES)],
            "family": "code_to_semantic",
            "input_ids_sha256": hashlib.sha256(
                f"tokenizer-input:{record['example_id']}".encode()
            ).hexdigest(),
        }
        for index, record in enumerate(records)
    )
    result = TokenizerFidelityBatchResult(
        case_bundle_manifest_sha256="3" * 64,
        case_inventory_sha256="4" * 64,
        task_database_identity_sha256=DATABASE_ID,
        binding_sha256=BINDING.binding_sha256,
        tokenizer_sha256=TOKENIZER_ID,
        records=records,
        details=details,
    )
    return build_tokenizer_fidelity_receipt(
        result,
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
    )


@lru_cache(maxsize=2)
def _cached_code_receipt_payload(category: str) -> dict[str, object]:
    if category not in {"raw_code", "execution_correctness"}:
        raise ValueError("test code receipt category is invalid")
    source = "print(1)\n"
    payload_ids = BINDING.encode_payload(source)
    structural = BINDING.structural_token_ids
    generated_ids = [
        structural["<|bcnlt:stream:begin|>"],
        structural["<|bcnlt:mode:raw_code|>"],
        *payload_ids,
        structural["<|bcnlt:stream:end|>"],
        structural["<|bcnlt:task:end|>"],
    ]
    records = tuple(_records(category))
    details: list[dict[str, object]] = []
    for record in records:
        example_id = str(record["example_id"])
        language = str(record["language"])
        stdin_text = "1\n" if category == "execution_correctness" else ""
        expected_stdout = (
            "2\n" if category == "execution_correctness" else None
        )
        sandbox = _code_execution(
            language=language,
            source=source,
            stdin_text=stdin_text,
            stdout=(
                b"2\n" if category == "execution_correctness" else b""
            ),
        ).to_dict()
        details.append(
            {
                "example_id": example_id,
                "task_example_id": hashlib.sha256(
                    f"code-task:{example_id}".encode()
                ).hexdigest(),
                "language": language,
                "task_input_ids_sha256": hashlib.sha256(
                    f"code-input:{example_id}".encode()
                ).hexdigest(),
                "generated_ids": generated_ids,
                "generated_ids_sha256": hashlib.sha256(
                    _canonical(generated_ids)
                ).hexdigest(),
                "protocol_complete": True,
                "payload_ids_sha256": hashlib.sha256(
                    _canonical(payload_ids)
                ).hexdigest(),
                "generated_source_sha256": hashlib.sha256(
                    source.encode()
                ).hexdigest(),
                "stdin": stdin_text,
                "expected_stdout": expected_stdout,
                "sandbox": sandbox,
            }
        )
    result = CodeAcceptanceBatchResult(
        category=category,
        case_bundle_manifest_sha256="5" * 64,
        case_inventory_sha256="6" * 64,
        task_database_identity_sha256=DATABASE_ID,
        binding_sha256=BINDING.binding_sha256,
        tokenizer_sha256=TOKENIZER_ID,
        records=records,
        details=tuple(details),
    )
    return build_code_acceptance_receipt(
        result,
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
    )


def _code_receipt_payload(category: str) -> dict[str, object]:
    return copy.deepcopy(_cached_code_receipt_payload(category))


def _category(category: str) -> dict[str, object]:
    if category == "tokenizer_fidelity":
        supporting_evidence = {
            "tokenizer_fidelity_receipt": _tokenizer_fidelity_receipt_payload()
        }
    elif category == "semantic_fidelity":
        supporting_evidence = {
            "strict_uncertainty_receipt": _strict_receipt_payload()
        }
    elif category in {"raw_code", "execution_correctness"}:
        supporting_evidence = {
            "code_acceptance_receipt": _code_receipt_payload(category)
        }
    elif category == "ordinary_instruction":
        supporting_evidence = {"ifeval_batch_receipt": _ifeval_receipt_payload()}
    elif category == "safety":
        supporting_evidence = {
            "safety_content_receipt": _safety_content_receipt_payload(),
            "containment_receipt": _containment_receipt_payload(),
        }
    elif category == "regression":
        supporting_evidence = {
            "regression_receipt": _regression_receipt_payload()
        }
    else:
        supporting_evidence = {}
    payload: dict[str, object] = {
        "schema_version": MODEL_CATEGORY_EVIDENCE_SCHEMA,
        "category": category,
        "checkpoint_sha256": CHECKPOINT_ID,
        "checkpoint_step": 86_016,
        "evidence_complete": True,
        "example_count": CATEGORY_MINIMUMS[category],
        "failure_codes": [],
        "heldout_example_ids_sha256": (
            HELDOUT_SUITE.category_example_ids_sha256[category]
        ),
        "heldout_manifest_sha256": HELDOUT_ID,
        "heldout_split": "test",
        "metrics": _metrics(category),
        "model_fingerprint": MODEL_ID,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "passed": True,
        "records": _records(category),
        "supporting_evidence": supporting_evidence,
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
        self.assertEqual(result.checkpoint_sha256, CHECKPOINT_ID)
        self.assertEqual(set(result.category_evidence_sha256), set(CATEGORIES))
        self.assertEqual(
            result.parser_runtime_identity_sha256, PARSER_RUNTIME_IDENTITY_SHA256
        )
        self.assertEqual(result.tokenizer_sha256, TOKENIZER_ID)
        self.assertEqual(result.heldout_manifest_sha256, HELDOUT_ID)

    def test_ordinary_instruction_requires_receipt_and_matching_baseline(self):
        evidence = [_category(category) for category in CATEGORIES]
        ordinary = next(
            item for item in evidence if item["category"] == "ordinary_instruction"
        )
        ordinary["supporting_evidence"] = {}
        body = {
            key: value for key, value in ordinary.items() if key != "artifact_sha256"
        }
        ordinary["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        with self.assertRaisesRegex(ValueError, "supporting evidence"):
            self._verify(tuple(evidence))

        evidence = [_category(category) for category in CATEGORIES]
        evaluations = [_evaluation("validation"), _evaluation("test")]
        for evaluation in evaluations:
            evaluation["baseline_source"] = "different-baseline"
            evaluation["baseline_evaluation_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "baseline lineage"):
            verify_model_acceptance(
                tuple(evaluations),
                tuple(_inference(task) for task in FAMILIES),
                tuple(evidence),
            )

    def test_missing_or_failed_category_cannot_be_hidden_by_loss_improvement(self):
        evidence = [_category(category) for category in CATEGORIES]
        with self.assertRaisesRegex(ValueError, "every independent category"):
            self._verify(tuple(evidence[:-1]))

        weak = copy.deepcopy(evidence)
        instruction = next(
            item for item in weak if item["category"] == "ordinary_instruction"
        )
        instruction["metrics"]["strict_prompt_accuracy"] = 0.79
        body = {
            key: value for key, value in instruction.items() if key != "artifact_sha256"
        }
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

        different_checkpoint = copy.deepcopy(evidence)
        different_checkpoint[-1]["checkpoint_sha256"] = "f" * 64
        body = {
            key: value
            for key, value in different_checkpoint[-1].items()
            if key != "artifact_sha256"
        }
        different_checkpoint[-1]["artifact_sha256"] = hashlib.sha256(
            _canonical(body)
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "lineage"):
            self._verify(tuple(different_checkpoint))

        wrong_parser = copy.deepcopy(evidence)
        for item in wrong_parser:
            item["parser_runtime_identity_sha256"] = "f" * 64
            body = {
                key: value for key, value in item.items() if key != "artifact_sha256"
            }
            item["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        with self.assertRaisesRegex(ValueError, "category lineage"):
            self._verify(tuple(wrong_parser))

        wrong_tokenizer = copy.deepcopy(evidence)
        for item in wrong_tokenizer:
            item["tokenizer_sha256"] = "f" * 64
            body = {
                key: value for key, value in item.items() if key != "artifact_sha256"
            }
            item["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        with self.assertRaisesRegex(ValueError, "category lineage"):
            self._verify(tuple(wrong_tokenizer))

        invented_manifest = copy.deepcopy(evidence)
        for item in invented_manifest:
            item["heldout_manifest_sha256"] = "f" * 64
            body = {
                key: value for key, value in item.items() if key != "artifact_sha256"
            }
            item["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        with self.assertRaisesRegex(ValueError, "incomplete|lineage"):
            self._verify(tuple(invented_manifest))

        wrong_inventory = copy.deepcopy(evidence)
        wrong_inventory[0]["heldout_example_ids_sha256"] = "f" * 64
        body = {
            key: value
            for key, value in wrong_inventory[0].items()
            if key != "artifact_sha256"
        }
        wrong_inventory[0]["artifact_sha256"] = hashlib.sha256(
            _canonical(body)
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self._verify(tuple(wrong_inventory))

    def test_heldout_suite_rejects_forged_or_ambiguous_case_inventory(self):
        forged = copy.deepcopy(HELDOUT_MANIFEST)
        forged["task_database_identity_sha256"] = "f" * 64
        unsigned = {
            key: value for key, value in forged.items() if key != "manifest_sha256"
        }
        forged["manifest_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
        with self.assertRaisesRegex(ValueError, "lineage"):
            verify_heldout_suite_manifest(
                forged,
                task_database_identity_sha256=DATABASE_ID,
            )

        duplicate = copy.deepcopy(HELDOUT_MANIFEST)
        inventory = duplicate["categories"]["tokenizer_fidelity"]
        inventory["example_ids"][1] = inventory["example_ids"][0]
        inventory["example_ids"] = sorted(inventory["example_ids"])
        inventory["example_ids_sha256"] = hashlib.sha256(
            _canonical(inventory["example_ids"])
        ).hexdigest()
        unsigned = {
            key: value for key, value in duplicate.items() if key != "manifest_sha256"
        }
        duplicate["manifest_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_heldout_suite_manifest(
                duplicate,
                task_database_identity_sha256=DATABASE_ID,
            )

    def test_category_records_ground_metrics_inventory_types_and_languages(self):
        fabricated = [_category(category) for category in CATEGORIES]
        instruction = next(
            item for item in fabricated if item["category"] == "ordinary_instruction"
        )
        instruction["metrics"]["strict_prompt_accuracy"] = 0.99
        body = {
            key: value for key, value in instruction.items() if key != "artifact_sha256"
        }
        instruction["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        with self.assertRaisesRegex(ValueError, "metrics do not match"):
            self._verify(tuple(fabricated))

        missing = [_category(category) for category in CATEGORIES]
        missing_category = next(
            item for item in missing if item["category"] == "raw_code"
        )
        missing_category["records"].pop()
        body = {
            key: value
            for key, value in missing_category.items()
            if key != "artifact_sha256"
        }
        missing_category["artifact_sha256"] = hashlib.sha256(
            _canonical(body)
        ).hexdigest()
        with self.assertRaisesRegex(
            ValueError,
            "records are incomplete|receipt expected records mismatch",
        ):
            self._verify(tuple(missing))

        reordered = [_category(category) for category in CATEGORIES]
        reordered_category = next(
            item for item in reordered if item["category"] == "raw_code"
        )
        reordered_category["records"][0], reordered_category["records"][1] = (
            reordered_category["records"][1],
            reordered_category["records"][0],
        )
        body = {
            key: value
            for key, value in reordered_category.items()
            if key != "artifact_sha256"
        }
        reordered_category["artifact_sha256"] = hashlib.sha256(
            _canonical(body)
        ).hexdigest()
        with self.assertRaisesRegex(
            ValueError,
            "inventory mismatch|receipt expected records mismatch",
        ):
            self._verify(tuple(reordered))

        non_boolean = [_category(category) for category in CATEGORIES]
        non_boolean_category = next(
            item for item in non_boolean if item["category"] == "raw_code"
        )
        non_boolean_category["records"][0]["exact_roundtrip"] = 1
        body = {
            key: value
            for key, value in non_boolean_category.items()
            if key != "artifact_sha256"
        }
        non_boolean_category["artifact_sha256"] = hashlib.sha256(
            _canonical(body)
        ).hexdigest()
        with self.assertRaisesRegex(
            ValueError,
            "must be boolean|receipt expected records mismatch",
        ):
            self._verify(tuple(non_boolean))

        language_failure = [_category(category) for category in CATEGORIES]
        semantic = next(
            item
            for item in language_failure
            if item["category"] == "semantic_fidelity"
        )
        failed_language = LANGUAGES[0]
        for record in semantic["records"]:
            if record["language"] == failed_language:
                record["ownership_correct"] = False
                record["unknown_semantic"] = True
        total = len(semantic["records"])
        failed = sum(
            record["language"] == failed_language for record in semantic["records"]
        )
        semantic["metrics"].update(
            {
                "ownership_accuracy": (total - failed) / total,
                "unknown_semantic_rate": failed / total,
                "language_passes": {
                    language: language != failed_language for language in LANGUAGES
                },
            }
        )
        body = {
            key: value for key, value in semantic.items() if key != "artifact_sha256"
        }
        semantic["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        with self.assertRaisesRegex(ValueError, "does not pass every language"):
            self._verify(tuple(language_failure))

    def test_semantic_fidelity_requires_strict_uncertainty_receipt(self):
        evidence = [_category(category) for category in CATEGORIES]
        semantic = next(
            item for item in evidence if item["category"] == "semantic_fidelity"
        )
        semantic["metrics"]["strict_uncertainty_acceptance_passed"] = False
        body = {
            key: value for key, value in semantic.items() if key != "artifact_sha256"
        }
        semantic["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        with self.assertRaisesRegex(ValueError, "semantic_fidelity"):
            self._verify(tuple(evidence))

        forged = [_category(category) for category in CATEGORIES]
        forged_semantic = next(
            item for item in forged if item["category"] == "semantic_fidelity"
        )
        raw_receipt = forged_semantic["supporting_evidence"][
            "strict_uncertainty_receipt"
        ]
        receipt = StrictUncertaintyAcceptanceReceipt(
            schema_version=raw_receipt["schema_version"],
            calibration_artifact_sha256=raw_receipt[
                "calibration_artifact_sha256"
            ],
            parser_runtime_identity_sha256="f" * 64,
            policy=raw_receipt["policy"],
            global_metrics=raw_receipt["global_metrics"],
            per_language_metrics=raw_receipt["per_language_metrics"],
            upstream_promoted=True,
            passed=True,
            failure_codes=(),
            receipt_sha256="",
        )
        receipt = replace(receipt, receipt_sha256=receipt.compute_sha256())
        forged_semantic["supporting_evidence"] = {
            "strict_uncertainty_receipt": receipt.to_dict()
        }
        forged_semantic["metrics"]["strict_uncertainty_receipt_sha256"] = (
            receipt.receipt_sha256
        )
        body = {
            key: value
            for key, value in forged_semantic.items()
            if key != "artifact_sha256"
        }
        forged_semantic["artifact_sha256"] = hashlib.sha256(
            _canonical(body)
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "current runtime gates"):
            self._verify(tuple(forged))

    def test_category_rates_must_be_probabilities(self):
        invalid_rates = (
            ("semantic_fidelity", "unknown_semantic_rate", -1.0),
            ("semantic_fidelity", "ownership_accuracy", 1.1),
            ("raw_code", "compile_pass_rate", 1.1),
            ("ordinary_instruction", "empty_response_rate", -0.1),
            ("execution_correctness", "sandbox_execution_rate", 1.1),
        )
        for category, metric, value in invalid_rates:
            with self.subTest(category=category, metric=metric):
                evidence = [_category(name) for name in CATEGORIES]
                item = next(
                    entry for entry in evidence if entry["category"] == category
                )
                item["metrics"][metric] = value
                body = {
                    key: body_value
                    for key, body_value in item.items()
                    if key != "artifact_sha256"
                }
                item["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
                with self.assertRaisesRegex(
                    ValueError, f"{category}|between zero and one"
                ):
                    self._verify(tuple(evidence))


if __name__ == "__main__":
    unittest.main()
