"""Compile immutable model-acceptance evidence from exact per-case outcomes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .artifact_schemas import MODEL_CATEGORY_EVIDENCE_SCHEMA
from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
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
from .strict_uncertainty_acceptance import (
    StrictUncertaintyAcceptanceReceipt,
    validate_strict_uncertainty_receipt_payload,
)
from .tokenizer import EXPECTED_TOKENIZER_SHA256

MAX_CATEGORY_RECORDS_BYTES = 16 * 1024 * 1024


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
    if category == "semantic_fidelity":
        if ifeval_batch_receipt is not None:
            raise ValueError("IFEval receipt is only valid for ordinary instruction")
        if type(strict_uncertainty_receipt) is not dict:
            raise ValueError("semantic fidelity requires a strict uncertainty receipt")
        receipt = validate_strict_uncertainty_receipt(strict_uncertainty_receipt)
        declared_metrics = {
            "strict_uncertainty_acceptance_passed": receipt.passed,
            "strict_uncertainty_receipt_sha256": receipt.receipt_sha256,
        }
        supporting_evidence = {"strict_uncertainty_receipt": receipt.to_dict()}
    elif category == "ordinary_instruction":
        if strict_uncertainty_receipt is not None:
            raise ValueError("strict uncertainty receipt is only valid for semantic fidelity")
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
    elif strict_uncertainty_receipt is not None:
        raise ValueError("strict uncertainty receipt is only valid for semantic fidelity")
    elif ifeval_batch_receipt is not None:
        raise ValueError("IFEval receipt is only valid for ordinary instruction")
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
    ) + "\n"
    publish_new_files(
        ((destination, rendered.encode("utf-8")),),
        artifact_label="model category evidence",
    )
    return destination


def checkpoint_sha256(path: str | Path) -> str:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise ValueError("model checkpoint does not exist")
    return _sha256_file(checkpoint)
