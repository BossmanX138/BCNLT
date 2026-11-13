"""Compile immutable model-acceptance evidence from exact per-case outcomes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .artifact_schemas import MODEL_CATEGORY_EVIDENCE_SCHEMA
from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from .code_acceptance import (
    CODE_ACCEPTANCE_CATEGORIES,
    code_acceptance_receipt_metrics,
    validate_code_acceptance_receipt,
)
from .containment_acceptance import (
    containment_receipt_metrics,
    validate_containment_receipt,
)
from .ifeval_case_runner import validate_ifeval_batch_receipt
from .immutable_artifacts import publish_new_files
from .model_acceptance import (
    ACCEPTANCE_CATEGORIES,
    EXPECTED_LANGUAGES,
    EXPECTED_MODEL_FINGERPRINT,
    recompute_category_metrics,
    verify_category_metrics,
    verify_heldout_suite_manifest,
)
from .regression_acceptance import (
    regression_receipt_metrics,
    validate_regression_receipt,
)
from .safety_judge_case_runner import (
    safety_content_receipt_metrics,
    validate_safety_content_receipt,
)
from .semantic_acceptance import (
    semantic_acceptance_receipt_metrics,
    validate_semantic_acceptance_receipt,
)
from .strict_uncertainty_acceptance import (
    StrictUncertaintyAcceptanceReceipt,
    validate_strict_uncertainty_receipt_payload,
)
from .tokenizer import EXPECTED_TOKENIZER_SHA256
from .tokenizer_fidelity_acceptance import (
    tokenizer_fidelity_receipt_metrics,
    validate_tokenizer_fidelity_receipt,
)

MAX_CATEGORY_RECORDS_BYTES = 16 * 1024 * 1024
MAX_MODEL_CATEGORY_EVIDENCE_BYTES = 512 * 1024 * 1024


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_strict_uncertainty_receipt(
    payload: dict[str, object],
) -> StrictUncertaintyAcceptanceReceipt:
    return validate_strict_uncertainty_receipt_payload(
        payload,
        expected_parser_runtime_identity_sha256=PARSER_RUNTIME_IDENTITY_SHA256,
        expected_languages=EXPECTED_LANGUAGES,
    )


def compile_model_category_evidence(
    *,
    category: str,
    records: Sequence[dict[str, object]],
    heldout_manifest: dict[str, object],
    task_database_identity_sha256: str | None = None,
    task_database_identities_sha256: Mapping[str, str] | None = None,
    checkpoint_sha256: str,
    checkpoint_step: int,
    strict_uncertainty_receipt: dict[str, object] | None = None,
    ifeval_batch_receipt: dict[str, object] | None = None,
    safety_content_receipt: dict[str, object] | None = None,
    containment_receipt: dict[str, object] | None = None,
    regression_receipt: dict[str, object] | None = None,
    tokenizer_fidelity_receipt: dict[str, object] | None = None,
    code_acceptance_receipt: dict[str, object] | None = None,
    semantic_acceptance_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    if category not in ACCEPTANCE_CATEGORIES:
        raise ValueError("model acceptance category is invalid")
    if (
        not _is_sha256(checkpoint_sha256)
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
    ):
        raise ValueError("model category evidence lineage is invalid")
    suite = verify_heldout_suite_manifest(
        heldout_manifest,
        task_database_identity_sha256=task_database_identity_sha256,
        task_database_identities_sha256=task_database_identities_sha256,
    )
    evidence_database_identity = (
        suite.evaluation_task_database_identities_sha256["test"]
    )
    declared_metrics: dict[str, object] = {}
    supporting_evidence: dict[str, object] = {}
    if category == "tokenizer_fidelity":
        if any(
            receipt is not None
            for receipt in (
                strict_uncertainty_receipt,
                ifeval_batch_receipt,
                safety_content_receipt,
                containment_receipt,
                regression_receipt,
                code_acceptance_receipt,
                semantic_acceptance_receipt,
            )
        ):
            raise ValueError("category receipt is not valid for tokenizer fidelity")
        if type(tokenizer_fidelity_receipt) is not dict:
            raise ValueError(
                "tokenizer fidelity requires a tokenizer fidelity receipt"
            )
        receipt_payload = validate_tokenizer_fidelity_receipt(
            tokenizer_fidelity_receipt,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_step=checkpoint_step,
            expected_example_ids=suite.category_example_ids[category],
            expected_records=records,
            expected_task_database_identity_sha256=evidence_database_identity,
        )
        declared_metrics = tokenizer_fidelity_receipt_metrics(receipt_payload)
        supporting_evidence = {"tokenizer_fidelity_receipt": receipt_payload}
    elif category == "semantic_fidelity":
        if any(
            receipt is not None
            for receipt in (
                ifeval_batch_receipt,
                safety_content_receipt,
                containment_receipt,
                regression_receipt,
                tokenizer_fidelity_receipt,
                code_acceptance_receipt,
            )
        ):
            raise ValueError("category receipt is not valid for semantic fidelity")
        if type(strict_uncertainty_receipt) is not dict:
            raise ValueError("semantic fidelity requires a strict uncertainty receipt")
        if type(semantic_acceptance_receipt) is not dict:
            raise ValueError("semantic fidelity requires a semantic acceptance receipt")
        receipt = validate_strict_uncertainty_receipt(strict_uncertainty_receipt)
        semantic_payload = validate_semantic_acceptance_receipt(
            semantic_acceptance_receipt,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_step=checkpoint_step,
            expected_example_ids=suite.category_example_ids[category],
            expected_records=records,
            expected_task_database_identity_sha256=evidence_database_identity,
        )
        declared_metrics = {
            "strict_uncertainty_acceptance_passed": receipt.passed,
            "strict_uncertainty_receipt_sha256": receipt.receipt_sha256,
            **semantic_acceptance_receipt_metrics(semantic_payload),
        }
        supporting_evidence = {
            "strict_uncertainty_receipt": receipt.to_dict(),
            "semantic_acceptance_receipt": semantic_payload,
        }
    elif category in CODE_ACCEPTANCE_CATEGORIES:
        if any(
            receipt is not None
            for receipt in (
                strict_uncertainty_receipt,
                ifeval_batch_receipt,
                safety_content_receipt,
                containment_receipt,
                regression_receipt,
                tokenizer_fidelity_receipt,
                semantic_acceptance_receipt,
            )
        ):
            raise ValueError("category receipt is not valid for code acceptance")
        if type(code_acceptance_receipt) is not dict:
            raise ValueError(f"{category} requires a code acceptance receipt")
        receipt_payload = validate_code_acceptance_receipt(
            code_acceptance_receipt,
            expected_category=category,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_step=checkpoint_step,
            expected_example_ids=suite.category_example_ids[category],
            expected_records=records,
            expected_task_database_identity_sha256=evidence_database_identity,
        )
        declared_metrics = code_acceptance_receipt_metrics(receipt_payload)
        supporting_evidence = {"code_acceptance_receipt": receipt_payload}
    elif category == "ordinary_instruction":
        if any(
            receipt is not None
            for receipt in (
                strict_uncertainty_receipt,
                safety_content_receipt,
                containment_receipt,
                regression_receipt,
                tokenizer_fidelity_receipt,
                code_acceptance_receipt,
                semantic_acceptance_receipt,
            )
        ):
            raise ValueError("category receipt is not valid for ordinary instruction")
        if type(ifeval_batch_receipt) is not dict:
            raise ValueError("ordinary instruction requires an IFEval batch receipt")
        receipt_payload = validate_ifeval_batch_receipt(
            ifeval_batch_receipt,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_step=checkpoint_step,
            expected_example_ids=suite.category_example_ids[category],
            expected_records=records,
        )
        supporting_evidence = {"ifeval_batch_receipt": receipt_payload}
    elif category == "safety":
        if any(
            receipt is not None
            for receipt in (
                strict_uncertainty_receipt,
                ifeval_batch_receipt,
                regression_receipt,
                tokenizer_fidelity_receipt,
                code_acceptance_receipt,
                semantic_acceptance_receipt,
            )
        ):
            raise ValueError("category receipt is not valid for safety")
        if type(safety_content_receipt) is not dict:
            raise ValueError("safety requires a safety content receipt")
        if type(containment_receipt) is not dict:
            raise ValueError("safety requires a containment receipt")
        content_payload = validate_safety_content_receipt(
            safety_content_receipt,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_step=checkpoint_step,
            expected_example_ids=suite.category_example_ids[category],
            expected_records=records,
        )
        containment_payload = validate_containment_receipt(
            containment_receipt,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_step=checkpoint_step,
        )
        content_metrics = safety_content_receipt_metrics(content_payload)
        containment_metrics = containment_receipt_metrics(containment_payload)
        content_secret_count = content_metrics.get("secret_leakage_count")
        host_secret_count = containment_metrics.get("host_secret_exposure_count")
        if type(content_secret_count) is not int or type(host_secret_count) is not int:
            raise ValueError("safety receipt metrics are invalid")
        declared_metrics = {
            "safety_content_receipt_sha256": content_metrics.get(
                "safety_content_receipt_sha256"
            ),
            "containment_acceptance_passed": containment_metrics.get(
                "containment_acceptance_passed"
            ),
            "containment_receipt_sha256": containment_metrics.get(
                "containment_receipt_sha256"
            ),
            "sandbox_escape_count": containment_metrics.get(
                "sandbox_escape_count"
            ),
            "network_violation_count": containment_metrics.get(
                "network_violation_count"
            ),
            "secret_leakage_count": content_secret_count + host_secret_count,
            "unauthorized_file_write_count": containment_metrics.get(
                "unauthorized_file_write_count"
            ),
        }
        supporting_evidence = {
            "safety_content_receipt": content_payload,
            "containment_receipt": containment_payload,
        }
    elif category == "regression":
        if any(
            receipt is not None
            for receipt in (
                strict_uncertainty_receipt,
                ifeval_batch_receipt,
                safety_content_receipt,
                containment_receipt,
                tokenizer_fidelity_receipt,
                code_acceptance_receipt,
                semantic_acceptance_receipt,
            )
        ):
            raise ValueError("category receipt is not valid for regression")
        if type(regression_receipt) is not dict:
            raise ValueError("regression requires a regression receipt")
        receipt_payload = validate_regression_receipt(
            regression_receipt,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_step=checkpoint_step,
            expected_example_ids=suite.category_example_ids[category],
            expected_records=records,
        )
        declared_metrics = regression_receipt_metrics(receipt_payload)
        supporting_evidence = {"regression_receipt": receipt_payload}
    elif any(
        receipt is not None
        for receipt in (
            strict_uncertainty_receipt,
            ifeval_batch_receipt,
            safety_content_receipt,
            containment_receipt,
            regression_receipt,
            tokenizer_fidelity_receipt,
            code_acceptance_receipt,
            semantic_acceptance_receipt,
        )
    ):
        raise ValueError("category receipt is not valid for this category")
    metrics = recompute_category_metrics(
        category,
        list(records),
        suite.category_example_ids[category],
        declared_metrics,
    )
    failure_codes: list[str] = []
    try:
        verify_category_metrics(category, metrics)
    except ValueError:
        failure_codes.append("category_gate_failed")
    body: dict[str, object] = {
        "schema_version": MODEL_CATEGORY_EVIDENCE_SCHEMA,
        "category": category,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "evidence_complete": True,
        "example_count": len(records),
        "failure_codes": failure_codes,
        "heldout_example_ids_sha256": suite.category_example_ids_sha256[category],
        "heldout_manifest_sha256": suite.manifest_sha256,
        "heldout_split": "test",
        "metrics": metrics,
        "model_fingerprint": EXPECTED_MODEL_FINGERPRINT,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "passed": not failure_codes,
        "records": list(records),
        "supporting_evidence": supporting_evidence,
        "task_database_identity_sha256": evidence_database_identity,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
    }
    return {**body, "artifact_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def publish_model_category_evidence(
    output: str | Path,
    payload: dict[str, object],
) -> Path:
    destination = Path(output).resolve()
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if len(rendered) > MAX_MODEL_CATEGORY_EVIDENCE_BYTES:
        raise ValueError("model category evidence exceeds its size bound")
    publish_new_files(
        ((destination, rendered),),
        artifact_label="model category evidence",
    )
    return destination


def checkpoint_sha256(path: str | Path) -> str:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise ValueError("model checkpoint does not exist")
    return _sha256_file(checkpoint)
