"""Compile immutable model-acceptance evidence from exact per-case outcomes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from .artifact_schemas import MODEL_CATEGORY_EVIDENCE_SCHEMA
from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from .immutable_artifacts import publish_new_files
from .model_acceptance import (
    ACCEPTANCE_CATEGORIES,
    EXPECTED_LANGUAGES,
    EXPECTED_MODEL_FINGERPRINT,
    recompute_category_metrics,
    verify_category_metrics,
    verify_heldout_suite_manifest,
)
from .strict_uncertainty_acceptance import StrictUncertaintyAcceptanceReceipt
from .tokenizer import EXPECTED_TOKENIZER_SHA256

MAX_CATEGORY_RECORDS_BYTES = 16 * 1024 * 1024
_STRICT_POLICY_FIELDS = {
    "minimum_global_selective_coverage_at_025",
    "maximum_global_selective_error_rate_at_025",
    "minimum_per_language_selective_coverage_at_025",
    "maximum_per_language_selective_error_rate_at_025",
}
_STRICT_METRIC_FIELDS = {
    "sample_count",
    "brier",
    "ece",
    "selective_coverage_at_025",
    "selective_error_rate_at_025",
}


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


def _validated_policy(value: dict[object, object]) -> dict[str, float]:
    if set(value) != _STRICT_POLICY_FIELDS:
        raise ValueError("strict uncertainty policy fields are invalid")
    policy: dict[str, float] = {}
    for key, threshold in value.items():
        if (
            type(key) is not str
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("strict uncertainty policy is invalid")
        policy[key] = float(threshold)
    return policy


def _validated_strict_metrics(
    value: dict[object, object],
) -> dict[str, float | int | None]:
    if set(value) != _STRICT_METRIC_FIELDS:
        raise ValueError("strict uncertainty metric fields are invalid")
    sample_count = value.get("sample_count")
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError("strict uncertainty sample count is invalid")
    metrics: dict[str, float | int | None] = {"sample_count": sample_count}
    for field in _STRICT_METRIC_FIELDS - {"sample_count"}:
        metric = value.get(field)
        if metric is None:
            metrics[field] = None
        elif (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            or not 0.0 <= float(metric) <= 1.0
        ):
            raise ValueError("strict uncertainty metric is invalid")
        else:
            metrics[field] = float(metric)
    return metrics


def _strict_thresholds_pass(
    policy: dict[str, float],
    global_metrics: dict[str, float | int | None],
    per_language: dict[str, dict[str, float | int | None]],
) -> bool:
    global_coverage = global_metrics["selective_coverage_at_025"]
    global_error = global_metrics["selective_error_rate_at_025"]
    if (
        global_coverage is None
        or float(global_coverage)
        < policy["minimum_global_selective_coverage_at_025"]
        or (
            global_error is None
            and policy["minimum_global_selective_coverage_at_025"] > 0.0
        )
        or (
            global_error is not None
            and float(global_error)
            > policy["maximum_global_selective_error_rate_at_025"]
        )
    ):
        return False
    for metrics in per_language.values():
        coverage = metrics["selective_coverage_at_025"]
        error = metrics["selective_error_rate_at_025"]
        if (
            coverage is None
            or float(coverage)
            < policy["minimum_per_language_selective_coverage_at_025"]
            or (
                error is None
                and policy["minimum_per_language_selective_coverage_at_025"] > 0.0
            )
            or (
                error is not None
                and float(error)
                > policy["maximum_per_language_selective_error_rate_at_025"]
            )
        ):
            return False
    return True


def validate_strict_uncertainty_receipt(
    payload: dict[str, object],
) -> StrictUncertaintyAcceptanceReceipt:
    required = {
        "calibration_artifact_sha256",
        "failure_codes",
        "global_metrics",
        "parser_runtime_identity_sha256",
        "passed",
        "per_language_metrics",
        "policy",
        "receipt_sha256",
        "schema_version",
        "upstream_promoted",
    }
    if set(payload) != required:
        raise ValueError("strict uncertainty receipt fields are invalid")
    raw_policy = payload.get("policy")
    raw_global_metrics = payload.get("global_metrics")
    raw_per_language_metrics = payload.get("per_language_metrics")
    failure_codes = payload.get("failure_codes")
    if (
        type(raw_policy) is not dict
        or type(raw_global_metrics) is not dict
        or type(raw_per_language_metrics) is not dict
        or type(failure_codes) is not list
        or any(type(code) is not str or not code for code in failure_codes)
    ):
        raise ValueError("strict uncertainty receipt payload is invalid")
    policy = _validated_policy(raw_policy)
    global_metrics = _validated_strict_metrics(raw_global_metrics)
    if set(raw_per_language_metrics) != set(EXPECTED_LANGUAGES):
        raise ValueError("strict uncertainty receipt language coverage is incomplete")
    per_language_metrics: dict[str, dict[str, float | int | None]] = {}
    for language, raw_metrics in raw_per_language_metrics.items():
        if type(language) is not str or type(raw_metrics) is not dict:
            raise ValueError("strict uncertainty language metrics are invalid")
        per_language_metrics[language] = _validated_strict_metrics(raw_metrics)
    if int(global_metrics["sample_count"] or 0) != sum(
        int(metrics["sample_count"] or 0)
        for metrics in per_language_metrics.values()
    ):
        raise ValueError("strict uncertainty sample inventory is inconsistent")
    receipt = StrictUncertaintyAcceptanceReceipt(
        schema_version=str(payload.get("schema_version")),
        calibration_artifact_sha256=str(
            payload.get("calibration_artifact_sha256")
        ),
        parser_runtime_identity_sha256=str(
            payload.get("parser_runtime_identity_sha256")
        ),
        policy=policy,
        global_metrics=global_metrics,
        per_language_metrics=per_language_metrics,
        upstream_promoted=payload.get("upstream_promoted") is True,
        passed=payload.get("passed") is True,
        failure_codes=tuple(failure_codes),
        receipt_sha256=str(payload.get("receipt_sha256")),
    )
    try:
        receipt.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("strict uncertainty receipt integrity mismatch") from error
    if (
        receipt.passed is not True
        or receipt.failure_codes
        or receipt.parser_runtime_identity_sha256
        != PARSER_RUNTIME_IDENTITY_SHA256
        or not _is_sha256(receipt.calibration_artifact_sha256)
        or not _strict_thresholds_pass(
            receipt.policy,
            receipt.global_metrics,
            receipt.per_language_metrics,
        )
    ):
        raise ValueError("strict uncertainty receipt did not pass current runtime gates")
    return receipt


def compile_model_category_evidence(
    *,
    category: str,
    records: Sequence[dict[str, object]],
    heldout_manifest: dict[str, object],
    task_database_identity_sha256: str,
    checkpoint_sha256: str,
    checkpoint_step: int,
    strict_uncertainty_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    if category not in ACCEPTANCE_CATEGORIES:
        raise ValueError("model acceptance category is invalid")
    if (
        not _is_sha256(task_database_identity_sha256)
        or not _is_sha256(checkpoint_sha256)
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
    ):
        raise ValueError("model category evidence lineage is invalid")
    suite = verify_heldout_suite_manifest(
        heldout_manifest,
        task_database_identity_sha256=task_database_identity_sha256,
    )
    declared_metrics: dict[str, object] = {}
    if category == "semantic_fidelity":
        if type(strict_uncertainty_receipt) is not dict:
            raise ValueError("semantic fidelity requires a strict uncertainty receipt")
        receipt = validate_strict_uncertainty_receipt(strict_uncertainty_receipt)
        declared_metrics = {
            "strict_uncertainty_acceptance_passed": receipt.passed,
            "strict_uncertainty_receipt_sha256": receipt.receipt_sha256,
        }
    elif strict_uncertainty_receipt is not None:
        raise ValueError("strict uncertainty receipt is only valid for semantic fidelity")
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
        "task_database_identity_sha256": task_database_identity_sha256,
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
