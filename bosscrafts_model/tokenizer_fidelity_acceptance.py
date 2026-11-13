"""Execute frozen tokenizer-fidelity cases and publish checkpoint-bound evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .category_case_inventory import load_category_case_bundle
from .immutable_artifacts import publish_new_files
from .resumable_task_packing import ResumableTaskDataset, audit_task_database
from .tokenizer import (
    EXPECTED_BINDING_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    NativeTokenizerBinding,
)

TOKENIZER_FIDELITY_RECEIPT_SCHEMA = "bosscrafts-tokenizer-fidelity-receipt-v1"
MAX_TOKENIZER_FIDELITY_CASES = 100_000
MAX_TOKENIZER_FIDELITY_RECEIPT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class TokenizerFidelityBatchResult:
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


def _segmented_roundtrip(
    binding: NativeTokenizerBinding,
    input_ids: Sequence[int],
) -> tuple[bool, bool]:
    reconstructed: list[int] = []
    id_stable = True
    cursor = 0
    while cursor < len(input_ids):
        token_id = input_ids[cursor]
        if token_id <= 76:
            decoded = binding.decode([token_id])
            encoded = binding.encode(decoded)
            id_stable = id_stable and encoded == [token_id]
            reconstructed.append(token_id)
            cursor += 1
            continue
        end = cursor + 1
        while end < len(input_ids) and input_ids[end] > 76:
            end += 1
        payload_ids = list(input_ids[cursor:end])
        try:
            encoded_payload = binding.encode_payload(binding.decode(payload_ids))
        except (TypeError, ValueError):
            encoded_payload = []
        id_stable = id_stable and encoded_payload == payload_ids
        reconstructed.extend(encoded_payload)
        cursor = end
    return reconstructed == list(input_ids), id_stable


def _raw_payload_ranges(
    binding: NativeTokenizerBinding,
    *,
    family: str,
    input_ids: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    structural = binding.structural_token_ids
    stream_begin = structural["<|bcnlt:stream:begin|>"]
    stream_end = structural["<|bcnlt:stream:end|>"]
    raw_mode = structural["<|bcnlt:mode:raw_code|>"]
    semantic_mode = structural["<|bcnlt:mode:semantic_code|>"]
    task_end = structural["<|bcnlt:task:end|>"]
    task_id = structural[f"<|bcnlt:task:{family}|>"]
    expected_modes = {
        "code_to_semantic": (raw_mode, semantic_mode),
        "semantic_continuation": (semantic_mode, semantic_mode),
        "semantic_to_code": (semantic_mode, raw_mode),
    }
    modes = expected_modes.get(family)
    if modes is None or input_ids[:1] != [task_id] or input_ids[-1:] != [task_end]:
        raise ValueError("tokenizer fidelity task protocol is invalid")
    cursor = 1
    raw_ranges: list[tuple[int, int]] = []
    for expected_mode in modes:
        if input_ids[cursor : cursor + 2] != [stream_begin, expected_mode]:
            raise ValueError("tokenizer fidelity stream header is invalid")
        payload_start = cursor + 2
        try:
            payload_end = input_ids.index(stream_end, payload_start)
        except ValueError as error:
            raise ValueError("tokenizer fidelity stream is truncated") from error
        if expected_mode == raw_mode:
            raw_ranges.append((payload_start, payload_end))
        cursor = payload_end + 1
    if cursor != len(input_ids) - 1:
        raise ValueError("tokenizer fidelity task has trailing protocol tokens")
    return tuple(raw_ranges)


def run_tokenizer_fidelity_cases(
    binding: NativeTokenizerBinding,
    *,
    case_bundle_root: str | Path,
    test_task_database: str | Path,
) -> TokenizerFidelityBatchResult:
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    bundle = load_category_case_bundle(case_bundle_root)
    references = bundle.inventory.packed_task_references["tokenizer_fidelity"]
    if not references or len(references) > MAX_TOKENIZER_FIDELITY_CASES:
        raise ValueError("tokenizer fidelity case count is invalid")
    audit_before = audit_task_database(binding, test_task_database)
    if (
        not audit_before.audit_passed
        or audit_before.database_identity_sha256
        != bundle.inventory.test_task_database_identity_sha256
    ):
        raise ValueError("tokenizer fidelity test database lineage mismatch")
    dataset = ResumableTaskDataset(binding, test_task_database)
    task_ids = tuple(reference.task_example_id for reference in references)
    tasks = {
        str(task["example_id"]): task
        for task in dataset.iter_examples_by_id(task_ids, split="test")
    }
    records: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for reference in references:
        task = tasks[reference.task_example_id]
        if task["language"] != reference.language:
            raise ValueError("tokenizer fidelity task language mismatch")
        input_ids = task["input_ids"]
        family = task["family"]
        if (
            type(input_ids) is not list
            or any(type(token_id) is not int for token_id in input_ids)
            or type(family) is not str
        ):
            raise ValueError("tokenizer fidelity task is invalid")
        raw_ranges = _raw_payload_ranges(
            binding,
            family=family,
            input_ids=input_ids,
        )
        reserved_violation = False
        for start, end in raw_ranges:
            payload_ids = input_ids[start:end]
            if any(token_id <= 76 for token_id in payload_ids):
                reserved_violation = True
                continue
            try:
                if binding.encode_payload(binding.decode(payload_ids)) != payload_ids:
                    reserved_violation = True
            except (TypeError, ValueError):
                reserved_violation = True
        exact_roundtrip, id_stable = _segmented_roundtrip(binding, input_ids)
        record = {
            "example_id": reference.category_example_id,
            "exact_roundtrip": exact_roundtrip,
            "id_stable": id_stable,
            "reserved_control_violation": reserved_violation,
        }
        records.append(record)
        details.append(
            {
                **record,
                "task_example_id": reference.task_example_id,
                "language": reference.language,
                "family": family,
                "input_ids_sha256": _sha256(input_ids),
            }
        )
    audit_after = audit_task_database(binding, test_task_database)
    bundle_after = load_category_case_bundle(case_bundle_root)
    if (
        audit_after != audit_before
        or bundle_after.manifest_sha256 != bundle.manifest_sha256
        or bundle_after.inventory.inventory_sha256 != bundle.inventory.inventory_sha256
    ):
        raise ValueError("tokenizer fidelity inputs changed during execution")
    return TokenizerFidelityBatchResult(
        case_bundle_manifest_sha256=bundle.manifest_sha256,
        case_inventory_sha256=bundle.inventory.inventory_sha256,
        task_database_identity_sha256=audit_before.database_identity_sha256,
        binding_sha256=binding.binding_sha256,
        tokenizer_sha256=binding.tokenizer_sha256,
        records=tuple(records),
        details=tuple(details),
    )


def _validated_result(
    payload: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_records = payload.get("records")
    raw_details = payload.get("details")
    if (
        type(raw_records) is not list
        or type(raw_details) is not list
        or not raw_records
        or len(raw_records) != len(raw_details)
        or len(raw_records) > MAX_TOKENIZER_FIDELITY_CASES
        or any(type(record) is not dict for record in raw_records)
        or any(type(detail) is not dict for detail in raw_details)
    ):
        raise ValueError("tokenizer fidelity receipt results are invalid")
    records: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    record_fields = {
        "example_id",
        "exact_roundtrip",
        "id_stable",
        "reserved_control_violation",
    }
    detail_fields = record_fields | {
        "task_example_id",
        "language",
        "family",
        "input_ids_sha256",
    }
    seen: set[str] = set()
    for raw_record, raw_detail in zip(raw_records, raw_details, strict=True):
        record = dict(raw_record)
        detail = dict(raw_detail)
        example_id = record.get("example_id")
        if (
            set(record) != record_fields
            or set(detail) != detail_fields
            or not _is_sha256(example_id)
            or example_id in seen
            or any(
                type(record.get(key)) is not bool
                for key in record_fields - {"example_id"}
            )
            or any(detail.get(key) != value for key, value in record.items())
            or not _is_sha256(detail.get("task_example_id"))
            or not _is_sha256(detail.get("input_ids_sha256"))
            or type(detail.get("language")) is not str
            or detail.get("family")
            not in {"code_to_semantic", "semantic_continuation", "semantic_to_code"}
        ):
            raise ValueError("tokenizer fidelity receipt result is invalid")
        seen.add(str(example_id))
        records.append(record)
        details.append(detail)
    if [str(record["example_id"]) for record in records] != sorted(seen):
        raise ValueError("tokenizer fidelity receipt results are not ordered")
    return records, details


def build_tokenizer_fidelity_receipt(
    result: TokenizerFidelityBatchResult,
    *,
    checkpoint_sha256: str,
    checkpoint_step: int,
) -> dict[str, object]:
    records = [dict(record) for record in result.records]
    details = [dict(detail) for detail in result.details]
    body: dict[str, object] = {
        "schema_version": TOKENIZER_FIDELITY_RECEIPT_SCHEMA,
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
                result.binding_sha256,
                result.tokenizer_sha256,
            )
        )
    ):
        raise ValueError("tokenizer fidelity receipt lineage is invalid")
    _validated_result(body)
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


def validate_tokenizer_fidelity_receipt(
    payload: object,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
    expected_example_ids: Sequence[str] | None = None,
    expected_records: Sequence[dict[str, object]] | None = None,
    expected_task_database_identity_sha256: str | None = None,
) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != _RECEIPT_FIELDS:
        raise ValueError("tokenizer fidelity receipt fields are invalid")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != _sha256(body):
        raise ValueError("tokenizer fidelity receipt integrity mismatch")
    if (
        payload.get("schema_version") != TOKENIZER_FIDELITY_RECEIPT_SCHEMA
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
                "binding_sha256",
                "tokenizer_sha256",
            )
        )
    ):
        raise ValueError("tokenizer fidelity receipt lineage is invalid")
    records, details = _validated_result(payload)
    if payload.get("records_sha256") != _sha256(records) or payload.get(
        "details_sha256"
    ) != _sha256(details):
        raise ValueError("tokenizer fidelity receipt results integrity mismatch")
    if (
        expected_checkpoint_sha256 is not None
        and payload.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        raise ValueError("tokenizer fidelity receipt checkpoint mismatch")
    if (
        expected_checkpoint_step is not None
        and payload.get("checkpoint_step") != expected_checkpoint_step
    ):
        raise ValueError("tokenizer fidelity receipt checkpoint step mismatch")
    if expected_example_ids is not None and tuple(expected_example_ids) != tuple(
        str(record["example_id"]) for record in records
    ):
        raise ValueError("tokenizer fidelity receipt expected inventory mismatch")
    if expected_records is not None and list(expected_records) != records:
        raise ValueError("tokenizer fidelity receipt expected records mismatch")
    if (
        expected_task_database_identity_sha256 is not None
        and payload.get("task_database_identity_sha256")
        != expected_task_database_identity_sha256
    ):
        raise ValueError("tokenizer fidelity receipt database mismatch")
    return payload


def tokenizer_fidelity_receipt_metrics(payload: object) -> dict[str, object]:
    receipt = validate_tokenizer_fidelity_receipt(payload)
    return {"tokenizer_fidelity_receipt_sha256": receipt["receipt_sha256"]}


def publish_tokenizer_fidelity_receipt(
    output: str | Path,
    payload: dict[str, object],
) -> Path:
    validated = validate_tokenizer_fidelity_receipt(payload)
    rendered = (
        json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(rendered) > MAX_TOKENIZER_FIDELITY_RECEIPT_BYTES:
        raise ValueError("tokenizer fidelity receipt exceeds its size bound")
    destination = Path(output).resolve()
    publish_new_files(
        ((destination, rendered),),
        artifact_label="tokenizer fidelity acceptance receipt",
    )
    return destination
