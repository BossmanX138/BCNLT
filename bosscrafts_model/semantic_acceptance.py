"""Evaluate frozen model-produced BCNLT semantic streams."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .bcnlt_source_pipeline import _NativeSemanticStreamCodec
from .category_case_inventory import (
    load_category_case_bundle,
    load_category_case_records,
)
from .checkpoint_case_generation import MAX_FROZEN_GENERATED_TOKENS
from .immutable_artifacts import publish_new_files
from .inference_protocol import response_protocol_complete, response_token_ids
from .jsonl import split_jsonl_records
from .resumable_task_packing import ResumableTaskDataset, audit_task_database
from .tokenizer import (
    EXPECTED_BINDING_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    NativeTokenizerBinding,
)

SEMANTIC_ACCEPTANCE_RECEIPT_SCHEMA = "bosscrafts-semantic-acceptance-receipt-v1"
MAX_SEMANTIC_ACCEPTANCE_CASES = 100_000
MAX_SEMANTIC_ACCEPTANCE_RECEIPT_BYTES = 256 * 1024 * 1024
_OWNERSHIP_FIELDS = (
    "event",
    "scope",
    "defines",
    "symbol_id",
    "span",
    "ast_relation",
)


@dataclass(frozen=True)
class SemanticAcceptanceBatchResult:
    case_bundle_manifest_sha256: str
    case_inventory_sha256: str
    task_database_identity_sha256: str
    binding_sha256: str
    tokenizer_sha256: str
    records: tuple[dict[str, object], ...]
    details: tuple[dict[str, object], ...]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_ids(
    value: object,
    *,
    vocabulary_size: int,
    label: str,
) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) > MAX_FROZEN_GENERATED_TOKENS
    ):
        raise ValueError(f"semantic acceptance {label} IDs are invalid")
    result = list(value)
    if any(
        type(token_id) is not int or not 0 <= token_id < vocabulary_size
        for token_id in result
    ):
        raise ValueError(f"semantic acceptance {label} IDs are invalid")
    return result


def _decode_semantic_response(
    binding: NativeTokenizerBinding,
    ids: list[int],
) -> tuple[bool, str, list[dict[str, object]]]:
    complete = response_protocol_complete(
        binding,
        task="code_to_semantic",
        generated_ids=ids,
    )
    if not complete:
        return False, "", []
    try:
        semantic_jsonl = _NativeSemanticStreamCodec(binding).decode(
            response_token_ids(binding, "code_to_semantic", ids)
        )
        records: list[dict[str, object]] = []
        for line in split_jsonl_records(semantic_jsonl):
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError("semantic record is not an object")
            records.append(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, "", []
    return True, semantic_jsonl, records


def _ownership_projection(records: Sequence[dict[str, object]]) -> list[object]:
    return [
        {field: record.get(field) for field in _OWNERSHIP_FIELDS}
        for record in records
    ]


def _record_from_detail(
    binding: NativeTokenizerBinding,
    detail: dict[str, object],
) -> dict[str, object]:
    expected_ids = _validated_ids(
        detail.get("expected_ids"),
        vocabulary_size=binding.vocab_size,
        label="expected",
    )
    generated_ids = _validated_ids(
        detail.get("generated_ids"),
        vocabulary_size=binding.vocab_size,
        label="generated",
    )
    expected_complete, expected_jsonl, expected_records = _decode_semantic_response(
        binding, expected_ids
    )
    if not expected_complete or not expected_records:
        raise ValueError("semantic acceptance expected task response is invalid")
    generated_complete, generated_jsonl, generated_records = (
        _decode_semantic_response(binding, generated_ids)
    )
    expected_projection = _ownership_projection(expected_records)
    generated_projection = _ownership_projection(generated_records)
    expected_jsonl_sha256 = hashlib.sha256(expected_jsonl.encode("utf-8")).hexdigest()
    generated_jsonl_sha256 = hashlib.sha256(
        generated_jsonl.encode("utf-8")
    ).hexdigest()
    checks = {
        "expected_ids_sha256": _sha256(expected_ids),
        "generated_ids_sha256": _sha256(generated_ids),
        "protocol_complete": generated_complete,
        "expected_semantic_jsonl_sha256": expected_jsonl_sha256,
        "generated_semantic_jsonl_sha256": generated_jsonl_sha256,
        "expected_record_count": len(expected_records),
        "generated_record_count": len(generated_records),
        "expected_ownership_sha256": _sha256(expected_projection),
        "generated_ownership_sha256": _sha256(generated_projection),
    }
    if any(detail.get(key) != value for key, value in checks.items()):
        raise ValueError("semantic acceptance detail is inconsistent")
    return {
        "example_id": detail["example_id"],
        "language": detail["language"],
        "exact_semantic_record": generated_jsonl == expected_jsonl,
        "ownership_correct": (
            generated_complete and generated_projection == expected_projection
        ),
        "unknown_semantic": (
            not generated_records
            or any(
                record.get("event") == "UNKNOWN_SEMANTIC"
                for record in generated_records
            )
        ),
    }


_DETAIL_FIELDS = {
    "example_id",
    "task_example_id",
    "language",
    "task_input_ids_sha256",
    "expected_ids",
    "expected_ids_sha256",
    "generated_ids",
    "generated_ids_sha256",
    "protocol_complete",
    "expected_semantic_jsonl_sha256",
    "generated_semantic_jsonl_sha256",
    "expected_record_count",
    "generated_record_count",
    "expected_ownership_sha256",
    "generated_ownership_sha256",
}
_RECORD_FIELDS = {
    "example_id",
    "language",
    "exact_semantic_record",
    "ownership_correct",
    "unknown_semantic",
}


def _validated_records_and_details(
    records_value: object,
    details_value: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if (
        type(records_value) is not list
        or type(details_value) is not list
        or not records_value
        or len(records_value) != len(details_value)
        or len(records_value) > MAX_SEMANTIC_ACCEPTANCE_CASES
        or any(type(record) is not dict for record in records_value)
        or any(type(detail) is not dict for detail in details_value)
    ):
        raise ValueError("semantic acceptance receipt results are invalid")
    binding = NativeTokenizerBinding.load()
    records: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_record, raw_detail in zip(records_value, details_value, strict=True):
        record = dict(raw_record)
        detail = dict(raw_detail)
        example_id = record.get("example_id")
        if (
            set(record) != _RECORD_FIELDS
            or set(detail) != _DETAIL_FIELDS
            or not _is_sha256(example_id)
            or example_id in seen
            or detail.get("example_id") != example_id
            or not _is_sha256(detail.get("task_example_id"))
            or not _is_sha256(detail.get("task_input_ids_sha256"))
            or type(record.get("language")) is not str
            or detail.get("language") != record.get("language")
            or any(
                type(record.get(field)) is not bool
                for field in _RECORD_FIELDS - {"example_id", "language"}
            )
        ):
            raise ValueError("semantic acceptance receipt record is invalid")
        if _record_from_detail(binding, detail) != record:
            raise ValueError("semantic acceptance receipt record is inconsistent")
        seen.add(str(example_id))
        records.append(record)
        details.append(detail)
    return records, details


def _validated_generations(
    value: Mapping[str, Sequence[int]],
    expected_ids: tuple[str, ...],
    *,
    vocabulary_size: int,
) -> dict[str, list[int]]:
    if not isinstance(value, Mapping) or set(value) != set(expected_ids):
        raise ValueError("semantic acceptance generation inventory is invalid")
    return {
        example_id: _validated_ids(
            value.get(example_id),
            vocabulary_size=vocabulary_size,
            label="generated",
        )
        for example_id in expected_ids
    }


def run_semantic_acceptance_cases(
    binding: NativeTokenizerBinding,
    *,
    case_bundle_root: str | Path,
    test_task_database: str | Path,
    generated_ids_by_example_id: Mapping[str, Sequence[int]],
) -> SemanticAcceptanceBatchResult:
    """Compare model semantic streams against exact frozen code-to-semantic tasks."""
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    bundle = load_category_case_bundle(case_bundle_root)
    references = bundle.inventory.packed_task_references["semantic_fidelity"]
    if not references or len(references) > MAX_SEMANTIC_ACCEPTANCE_CASES:
        raise ValueError("semantic acceptance case count is invalid")
    category_ids = tuple(reference.category_example_id for reference in references)
    generations = _validated_generations(
        generated_ids_by_example_id,
        category_ids,
        vocabulary_size=binding.vocab_size,
    )
    case_records = load_category_case_records(bundle, "semantic_fidelity")
    cases = {str(record["example_id"]): record["case"] for record in case_records}
    audit_before = audit_task_database(binding, test_task_database)
    if (
        not audit_before.audit_passed
        or audit_before.database_identity_sha256
        != bundle.inventory.test_task_database_identity_sha256
    ):
        raise ValueError("semantic acceptance test database lineage mismatch")
    task_ids = tuple(reference.task_example_id for reference in references)
    dataset = ResumableTaskDataset(binding, test_task_database)
    tasks = {
        task["example_id"]: task
        for task in dataset.iter_examples_by_id(task_ids, split="test")
    }
    records: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for reference in references:
        task = tasks.get(reference.task_example_id)
        case = cases.get(reference.category_example_id)
        if (
            task is None
            or type(case) is not dict
            or reference.family != "code_to_semantic"
            or task["family"] != "code_to_semantic"
            or task["language"] != reference.language
            or case.get("task_example_id") != reference.task_example_id
            or case.get("language") != reference.language
            or case.get("family") != "code_to_semantic"
            or case.get("task_database_identity_sha256")
            != audit_before.database_identity_sha256
        ):
            raise ValueError("semantic acceptance task or case lineage mismatch")
        input_ids = task["input_ids"]
        loss_start = task["loss_start_index"]
        if not 0 < loss_start < len(input_ids):
            raise ValueError("semantic acceptance packed task is invalid")
        expected_ids = input_ids[loss_start:]
        generated_ids = generations[reference.category_example_id]
        expected_complete, expected_jsonl, expected_records = (
            _decode_semantic_response(binding, expected_ids)
        )
        if not expected_complete or not expected_records:
            raise ValueError("semantic acceptance frozen answer is invalid")
        generated_complete, generated_jsonl, generated_records = (
            _decode_semantic_response(binding, generated_ids)
        )
        expected_projection = _ownership_projection(expected_records)
        generated_projection = _ownership_projection(generated_records)
        detail: dict[str, object] = {
            "example_id": reference.category_example_id,
            "task_example_id": reference.task_example_id,
            "language": reference.language,
            "task_input_ids_sha256": _sha256(input_ids),
            "expected_ids": expected_ids,
            "expected_ids_sha256": _sha256(expected_ids),
            "generated_ids": generated_ids,
            "generated_ids_sha256": _sha256(generated_ids),
            "protocol_complete": generated_complete,
            "expected_semantic_jsonl_sha256": hashlib.sha256(
                expected_jsonl.encode("utf-8")
            ).hexdigest(),
            "generated_semantic_jsonl_sha256": hashlib.sha256(
                generated_jsonl.encode("utf-8")
            ).hexdigest(),
            "expected_record_count": len(expected_records),
            "generated_record_count": len(generated_records),
            "expected_ownership_sha256": _sha256(expected_projection),
            "generated_ownership_sha256": _sha256(generated_projection),
        }
        details.append(detail)
        records.append(_record_from_detail(binding, detail))
    audit_after = audit_task_database(binding, test_task_database)
    bundle_after = load_category_case_bundle(case_bundle_root)
    if (
        audit_after != audit_before
        or bundle_after.manifest_sha256 != bundle.manifest_sha256
        or bundle_after.inventory.inventory_sha256
        != bundle.inventory.inventory_sha256
    ):
        raise ValueError("semantic acceptance inputs changed during execution")
    return SemanticAcceptanceBatchResult(
        case_bundle_manifest_sha256=bundle.manifest_sha256,
        case_inventory_sha256=bundle.inventory.inventory_sha256,
        task_database_identity_sha256=audit_before.database_identity_sha256,
        binding_sha256=binding.binding_sha256,
        tokenizer_sha256=binding.tokenizer_sha256,
        records=tuple(records),
        details=tuple(details),
    )


def build_semantic_acceptance_receipt(
    result: SemanticAcceptanceBatchResult,
    *,
    checkpoint_sha256: str,
    checkpoint_step: int,
) -> dict[str, object]:
    if type(result) is not SemanticAcceptanceBatchResult:
        raise TypeError("result must be a SemanticAcceptanceBatchResult")
    records, details = _validated_records_and_details(
        list(result.records), list(result.details)
    )
    body: dict[str, object] = {
        "schema_version": SEMANTIC_ACCEPTANCE_RECEIPT_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "case_bundle_manifest_sha256": result.case_bundle_manifest_sha256,
        "case_inventory_sha256": result.case_inventory_sha256,
        "task_database_identity_sha256": result.task_database_identity_sha256,
        "binding_sha256": result.binding_sha256,
        "tokenizer_sha256": result.tokenizer_sha256,
        "records": records,
        "records_sha256": _sha256(records),
        "details": details,
        "details_sha256": _sha256(details),
    }
    if (
        not _is_sha256(checkpoint_sha256)
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
        or result.binding_sha256 != EXPECTED_BINDING_SHA256
        or result.tokenizer_sha256 != EXPECTED_TOKENIZER_SHA256
        or any(
            not _is_sha256(value)
            for value in (
                result.case_bundle_manifest_sha256,
                result.case_inventory_sha256,
                result.task_database_identity_sha256,
            )
        )
    ):
        raise ValueError("semantic acceptance receipt lineage is invalid")
    return {**body, "receipt_sha256": _sha256(body)}


_RECEIPT_FIELDS = {
    "schema_version",
    "checkpoint_sha256",
    "checkpoint_step",
    "case_bundle_manifest_sha256",
    "case_inventory_sha256",
    "task_database_identity_sha256",
    "binding_sha256",
    "tokenizer_sha256",
    "records",
    "records_sha256",
    "details",
    "details_sha256",
    "receipt_sha256",
}


def validate_semantic_acceptance_receipt(
    payload: object,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
    expected_example_ids: Sequence[str] | None = None,
    expected_records: Sequence[dict[str, object]] | None = None,
    expected_task_database_identity_sha256: str | None = None,
) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != _RECEIPT_FIELDS:
        raise ValueError("semantic acceptance receipt fields are invalid")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != _sha256(body):
        raise ValueError("semantic acceptance receipt integrity mismatch")
    if (
        payload.get("schema_version") != SEMANTIC_ACCEPTANCE_RECEIPT_SCHEMA
        or not _is_sha256(payload.get("checkpoint_sha256"))
        or type(payload.get("checkpoint_step")) is not int
        or int(payload["checkpoint_step"]) < 1
        or payload.get("binding_sha256") != EXPECTED_BINDING_SHA256
        or payload.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256
        or any(
            not _is_sha256(payload.get(key))
            for key in (
                "case_bundle_manifest_sha256",
                "case_inventory_sha256",
                "task_database_identity_sha256",
            )
        )
    ):
        raise ValueError("semantic acceptance receipt lineage is invalid")
    records, details = _validated_records_and_details(
        payload.get("records"), payload.get("details")
    )
    if (
        payload.get("records_sha256") != _sha256(records)
        or payload.get("details_sha256") != _sha256(details)
    ):
        raise ValueError("semantic acceptance receipt results integrity mismatch")
    checks = (
        (
            expected_checkpoint_sha256,
            payload.get("checkpoint_sha256"),
            "semantic acceptance receipt checkpoint mismatch",
        ),
        (
            expected_checkpoint_step,
            payload.get("checkpoint_step"),
            "semantic acceptance receipt checkpoint step mismatch",
        ),
        (
            expected_task_database_identity_sha256,
            payload.get("task_database_identity_sha256"),
            "semantic acceptance receipt database mismatch",
        ),
    )
    for expected, observed, message in checks:
        if expected is not None and expected != observed:
            raise ValueError(message)
    if expected_example_ids is not None and tuple(expected_example_ids) != tuple(
        str(record["example_id"]) for record in records
    ):
        raise ValueError("semantic acceptance receipt expected inventory mismatch")
    if expected_records is not None and list(expected_records) != records:
        raise ValueError("semantic acceptance receipt expected records mismatch")
    return payload


def semantic_acceptance_receipt_metrics(payload: object) -> dict[str, object]:
    receipt = validate_semantic_acceptance_receipt(payload)
    return {"semantic_acceptance_receipt_sha256": receipt["receipt_sha256"]}


def publish_semantic_acceptance_receipt(
    output: str | Path,
    payload: dict[str, object],
) -> Path:
    validated = validate_semantic_acceptance_receipt(payload)
    rendered = (
        json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(rendered) > MAX_SEMANTIC_ACCEPTANCE_RECEIPT_BYTES:
        raise ValueError("semantic acceptance receipt exceeds its size bound")
    destination = Path(output).resolve()
    publish_new_files(
        ((destination, rendered),), artifact_label="semantic acceptance receipt"
    )
    return destination
