"""Execute frozen code cases and publish checkpoint-bound acceptance evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .category_case_inventory import (
    MAX_EXECUTION_STDIN_BYTES,
    MAX_EXECUTION_STDOUT_BYTES,
    load_category_case_bundle,
    load_category_case_records,
)
from .immutable_artifacts import publish_new_files
from .inference_protocol import (
    response_payload_token_ids,
    response_protocol_complete,
)
from .resumable_task_packing import ResumableTaskDataset, audit_task_database
from .scientific_sandbox import (
    SandboxExecutionResult,
    validate_sandbox_execution_payload,
    validate_sandbox_execution_result,
)
from .tokenizer import (
    EXPECTED_BINDING_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    NativeTokenizerBinding,
)

CODE_ACCEPTANCE_RECEIPT_SCHEMA = "bosscrafts-code-acceptance-receipt-v1"
CODE_ACCEPTANCE_CATEGORIES = ("raw_code", "execution_correctness")
MAX_CODE_ACCEPTANCE_CASES = 100_000
MAX_GENERATED_TOKENS = 2_048
MAX_GENERATED_SOURCE_BYTES = 64 * 1024
MAX_CODE_ACCEPTANCE_RECEIPT_BYTES = 256 * 1024 * 1024


class SandboxExecutor(Protocol):
    def execute(
        self,
        *,
        language: str,
        source: str,
        stdin_text: str,
    ) -> SandboxExecutionResult: ...


@dataclass(frozen=True)
class CodeAcceptanceBatchResult:
    category: str
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


def _validated_generations(
    value: Mapping[str, Sequence[int]],
    expected_ids: tuple[str, ...],
    *,
    vocabulary_size: int,
) -> dict[str, list[int]]:
    if not isinstance(value, Mapping) or set(value) != set(expected_ids):
        raise ValueError("code acceptance generation inventory is invalid")
    result: dict[str, list[int]] = {}
    for example_id in expected_ids:
        raw_ids = value.get(example_id)
        if (
            not isinstance(raw_ids, Sequence)
            or isinstance(raw_ids, (str, bytes))
            or len(raw_ids) > MAX_GENERATED_TOKENS
        ):
            raise ValueError("code acceptance generated IDs are invalid")
        generated_ids = list(raw_ids)
        if any(
            type(token_id) is not int
            or not 0 <= token_id < vocabulary_size
            for token_id in generated_ids
        ):
            raise ValueError("code acceptance generated IDs are invalid")
        result[example_id] = generated_ids
    return result


def _decoded_generation(
    binding: NativeTokenizerBinding,
    generated_ids: list[int],
) -> tuple[bool, list[int], str, bool]:
    protocol_complete = response_protocol_complete(
        binding,
        task="semantic_to_code",
        generated_ids=generated_ids,
    )
    if not protocol_complete:
        return False, [], "", False
    payload_ids = response_payload_token_ids(
        binding,
        "semantic_to_code",
        generated_ids,
    )
    try:
        source = binding.decode(payload_ids)
        exact_roundtrip = (
            bool(payload_ids)
            and binding.encode_payload(source) == payload_ids
        )
    except (TypeError, ValueError):
        return True, payload_ids, "", False
    if len(source.encode("utf-8")) > MAX_GENERATED_SOURCE_BYTES:
        return True, payload_ids, "", exact_roundtrip
    return True, payload_ids, source, exact_roundtrip


def _compile_passed(sandbox: Mapping[str, object] | None) -> bool:
    return sandbox is not None and sandbox.get("phase") == "run"


def _record_from_detail(
    binding: NativeTokenizerBinding,
    *,
    category: str,
    detail: dict[str, object],
) -> dict[str, object]:
    generated_ids = detail.get("generated_ids")
    language = detail.get("language")
    stdin_text = detail.get("stdin")
    expected_stdout = detail.get("expected_stdout")
    sandbox = detail.get("sandbox")
    if (
        category not in CODE_ACCEPTANCE_CATEGORIES
        or type(generated_ids) is not list
        or len(generated_ids) > MAX_GENERATED_TOKENS
        or any(
            type(token_id) is not int
            or not 0 <= token_id < binding.vocab_size
            for token_id in generated_ids
        )
        or type(language) is not str
        or type(stdin_text) is not str
        or len(stdin_text.encode("utf-8")) > MAX_EXECUTION_STDIN_BYTES
        or (
            expected_stdout is not None
            and (
                type(expected_stdout) is not str
                or len(expected_stdout.encode("utf-8"))
                > MAX_EXECUTION_STDOUT_BYTES
            )
        )
        or (sandbox is not None and type(sandbox) is not dict)
    ):
        raise ValueError("code acceptance detail values are invalid")
    protocol_complete, payload_ids, source, exact_roundtrip = (
        _decoded_generation(binding, generated_ids)
    )
    if (
        detail.get("generated_ids_sha256") != _sha256(generated_ids)
        or detail.get("protocol_complete") is not protocol_complete
        or detail.get("payload_ids_sha256") != _sha256(payload_ids)
        or detail.get("generated_source_sha256")
        != hashlib.sha256(source.encode("utf-8")).hexdigest()
    ):
        raise ValueError("code acceptance generated detail is inconsistent")
    stdout = b""
    if sandbox is not None:
        stdout, _ = validate_sandbox_execution_payload(
            sandbox,
            language=language,
            source=source,
            stdin_text=stdin_text,
        )
    compiled = _compile_passed(sandbox)
    common = {
        "example_id": detail["example_id"],
        "language": language,
    }
    if category == "raw_code":
        if stdin_text != "" or expected_stdout is not None:
            raise ValueError("raw code detail oracle is invalid")
        return {
            **common,
            "exact_roundtrip": exact_roundtrip,
            "compile_passed": compiled,
            "silent_fallback": not protocol_complete or not source,
        }
    if type(expected_stdout) is not str:
        raise ValueError("execution correctness detail oracle is invalid")
    sandbox_executed = (
        sandbox is not None
        and sandbox.get("phase") == "run"
        and sandbox.get("timed_out") is False
        and sandbox.get("output_limited") is False
    )
    return {
        **common,
        "sandbox_executed": sandbox_executed,
        "expected_output_exact": (
            sandbox is not None
            and sandbox.get("status") == "passed"
            and stdout == expected_stdout.encode("utf-8")
        ),
        "compile_passed": compiled,
        "sandbox_escape": False,
    }


def _validated_records_and_details(
    category: str,
    records_value: object,
    details_value: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if (
        category not in CODE_ACCEPTANCE_CATEGORIES
        or type(records_value) is not list
        or type(details_value) is not list
        or not records_value
        or len(records_value) != len(details_value)
        or len(records_value) > MAX_CODE_ACCEPTANCE_CASES
        or any(type(record) is not dict for record in records_value)
        or any(type(detail) is not dict for detail in details_value)
    ):
        raise ValueError("code acceptance receipt results are invalid")
    record_fields = {
        "raw_code": {
            "example_id",
            "language",
            "exact_roundtrip",
            "compile_passed",
            "silent_fallback",
        },
        "execution_correctness": {
            "example_id",
            "language",
            "sandbox_executed",
            "expected_output_exact",
            "compile_passed",
            "sandbox_escape",
        },
    }[category]
    detail_fields = {
        "example_id",
        "task_example_id",
        "language",
        "task_input_ids_sha256",
        "generated_ids",
        "generated_ids_sha256",
        "protocol_complete",
        "payload_ids_sha256",
        "generated_source_sha256",
        "stdin",
        "expected_stdout",
        "sandbox",
    }
    binding = NativeTokenizerBinding.load()
    records: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_record, raw_detail in zip(records_value, details_value, strict=True):
        record = dict(raw_record)
        detail = dict(raw_detail)
        example_id = record.get("example_id")
        if (
            set(record) != record_fields
            or set(detail) != detail_fields
            or not _is_sha256(example_id)
            or example_id in seen
            or detail.get("example_id") != example_id
            or not _is_sha256(detail.get("task_example_id"))
            or not _is_sha256(detail.get("task_input_ids_sha256"))
            or any(
                type(record.get(field)) is not bool
                for field in record_fields - {"example_id", "language"}
            )
        ):
            raise ValueError("code acceptance receipt record is invalid")
        recomputed = _record_from_detail(
            binding,
            category=category,
            detail=detail,
        )
        if recomputed != record:
            raise ValueError("code acceptance receipt record is inconsistent")
        seen.add(str(example_id))
        records.append(record)
        details.append(detail)
    return records, details


def run_code_acceptance_cases(
    binding: NativeTokenizerBinding,
    sandbox: SandboxExecutor,
    *,
    category: str,
    case_bundle_root: str | Path,
    test_task_database: str | Path,
    generated_ids_by_example_id: Mapping[str, Sequence[int]],
) -> CodeAcceptanceBatchResult:
    """Compile or execute exact generated outputs for one frozen category."""
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if category not in CODE_ACCEPTANCE_CATEGORIES:
        raise ValueError("code acceptance category is invalid")
    bundle = load_category_case_bundle(case_bundle_root)
    references = bundle.inventory.packed_task_references[category]
    if not references or len(references) > MAX_CODE_ACCEPTANCE_CASES:
        raise ValueError("code acceptance case count is invalid")
    expected_ids = tuple(reference.category_example_id for reference in references)
    generations = _validated_generations(
        generated_ids_by_example_id,
        expected_ids,
        vocabulary_size=binding.vocab_size,
    )
    case_records = load_category_case_records(bundle, category)
    cases = {
        str(record["example_id"]): record["case"] for record in case_records
    }
    audit_before = audit_task_database(binding, test_task_database)
    if (
        not audit_before.audit_passed
        or audit_before.database_identity_sha256
        != bundle.inventory.test_task_database_identity_sha256
    ):
        raise ValueError("code acceptance test database lineage mismatch")
    dataset = ResumableTaskDataset(binding, test_task_database)
    task_ids = tuple(reference.task_example_id for reference in references)
    tasks = {
        str(task["example_id"]): task
        for task in dataset.iter_examples_by_id(task_ids, split="test")
    }
    records: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for reference in references:
        task = tasks.get(reference.task_example_id)
        case = cases.get(reference.category_example_id)
        if (
            type(task) is not dict
            or type(case) is not dict
            or task.get("language") != reference.language
            or task.get("family") != "semantic_to_code"
            or case.get("task_example_id") != reference.task_example_id
            or case.get("language") != reference.language
            or case.get("family") != "semantic_to_code"
            or case.get("task_database_identity_sha256")
            != audit_before.database_identity_sha256
        ):
            raise ValueError("code acceptance task or case lineage mismatch")
        input_ids = task.get("input_ids")
        if type(input_ids) is not list or any(
            type(token_id) is not int for token_id in input_ids
        ):
            raise ValueError("code acceptance task input IDs are invalid")
        generated_ids = generations[reference.category_example_id]
        protocol_complete, payload_ids, source, _ = _decoded_generation(
            binding,
            generated_ids,
        )
        stdin_text = ""
        expected_stdout: str | None = None
        if category == "execution_correctness":
            stdin = case.get("stdin")
            stdout = case.get("expected_stdout")
            if type(stdin) is not str or type(stdout) is not str:
                raise ValueError("execution correctness oracle is invalid")
            stdin_text = stdin
            expected_stdout = stdout
        sandbox_payload: dict[str, object] | None = None
        if source:
            result = sandbox.execute(
                language=reference.language,
                source=source,
                stdin_text=stdin_text,
            )
            validate_sandbox_execution_result(
                result,
                language=reference.language,
                source=source,
                stdin_text=stdin_text,
            )
            sandbox_payload = result.to_dict()
        detail: dict[str, object] = {
            "example_id": reference.category_example_id,
            "task_example_id": reference.task_example_id,
            "language": reference.language,
            "task_input_ids_sha256": _sha256(input_ids),
            "generated_ids": generated_ids,
            "generated_ids_sha256": _sha256(generated_ids),
            "protocol_complete": protocol_complete,
            "payload_ids_sha256": _sha256(payload_ids),
            "generated_source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "stdin": stdin_text,
            "expected_stdout": expected_stdout,
            "sandbox": sandbox_payload,
        }
        details.append(detail)
        records.append(
            _record_from_detail(binding, category=category, detail=detail)
        )
    audit_after = audit_task_database(binding, test_task_database)
    bundle_after = load_category_case_bundle(case_bundle_root)
    if (
        audit_after != audit_before
        or bundle_after.manifest_sha256 != bundle.manifest_sha256
        or bundle_after.inventory.inventory_sha256
        != bundle.inventory.inventory_sha256
    ):
        raise ValueError("code acceptance inputs changed during execution")
    return CodeAcceptanceBatchResult(
        category=category,
        case_bundle_manifest_sha256=bundle.manifest_sha256,
        case_inventory_sha256=bundle.inventory.inventory_sha256,
        task_database_identity_sha256=audit_before.database_identity_sha256,
        binding_sha256=binding.binding_sha256,
        tokenizer_sha256=binding.tokenizer_sha256,
        records=tuple(records),
        details=tuple(details),
    )


def build_code_acceptance_receipt(
    result: CodeAcceptanceBatchResult,
    *,
    checkpoint_sha256: str,
    checkpoint_step: int,
) -> dict[str, object]:
    if type(result) is not CodeAcceptanceBatchResult:
        raise TypeError("result must be a CodeAcceptanceBatchResult")
    records, details = _validated_records_and_details(
        result.category,
        list(result.records),
        list(result.details),
    )
    generation_hashes = {
        str(detail["example_id"]): detail["generated_ids_sha256"]
        for detail in details
    }
    body: dict[str, object] = {
        "schema_version": CODE_ACCEPTANCE_RECEIPT_SCHEMA,
        "category": result.category,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "case_bundle_manifest_sha256": result.case_bundle_manifest_sha256,
        "case_inventory_sha256": result.case_inventory_sha256,
        "task_database_identity_sha256": result.task_database_identity_sha256,
        "binding_sha256": result.binding_sha256,
        "tokenizer_sha256": result.tokenizer_sha256,
        "generation_manifest_sha256": _sha256(generation_hashes),
        "records": records,
        "records_sha256": _sha256(records),
        "details": details,
        "details_sha256": _sha256(details),
    }
    if (
        not _is_sha256(checkpoint_sha256)
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
        or result.category not in CODE_ACCEPTANCE_CATEGORIES
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
        raise ValueError("code acceptance receipt lineage is invalid")
    return {**body, "receipt_sha256": _sha256(body)}


_RECEIPT_FIELDS = {
    "schema_version",
    "category",
    "checkpoint_sha256",
    "checkpoint_step",
    "case_bundle_manifest_sha256",
    "case_inventory_sha256",
    "task_database_identity_sha256",
    "binding_sha256",
    "tokenizer_sha256",
    "generation_manifest_sha256",
    "records",
    "records_sha256",
    "details",
    "details_sha256",
    "receipt_sha256",
}


def validate_code_acceptance_receipt(
    payload: object,
    *,
    expected_category: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
    expected_example_ids: Sequence[str] | None = None,
    expected_records: Sequence[dict[str, object]] | None = None,
    expected_task_database_identity_sha256: str | None = None,
) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != _RECEIPT_FIELDS:
        raise ValueError("code acceptance receipt fields are invalid")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != _sha256(body):
        raise ValueError("code acceptance receipt integrity mismatch")
    category = payload.get("category")
    if (
        payload.get("schema_version") != CODE_ACCEPTANCE_RECEIPT_SCHEMA
        or category not in CODE_ACCEPTANCE_CATEGORIES
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
                "generation_manifest_sha256",
            )
        )
    ):
        raise ValueError("code acceptance receipt lineage is invalid")
    records, details = _validated_records_and_details(
        str(category),
        payload.get("records"),
        payload.get("details"),
    )
    generation_hashes = {
        str(detail["example_id"]): detail["generated_ids_sha256"]
        for detail in details
    }
    if (
        payload.get("generation_manifest_sha256") != _sha256(generation_hashes)
        or payload.get("records_sha256") != _sha256(records)
        or payload.get("details_sha256") != _sha256(details)
    ):
        raise ValueError("code acceptance receipt results integrity mismatch")
    checks = (
        (
            expected_category,
            category,
            "code acceptance receipt category mismatch",
        ),
        (
            expected_checkpoint_sha256,
            payload.get("checkpoint_sha256"),
            "code acceptance receipt checkpoint mismatch",
        ),
        (
            expected_checkpoint_step,
            payload.get("checkpoint_step"),
            "code acceptance receipt checkpoint step mismatch",
        ),
        (
            expected_task_database_identity_sha256,
            payload.get("task_database_identity_sha256"),
            "code acceptance receipt database mismatch",
        ),
    )
    for expected, observed, message in checks:
        if expected is not None and expected != observed:
            raise ValueError(message)
    if expected_example_ids is not None and tuple(expected_example_ids) != tuple(
        str(record["example_id"]) for record in records
    ):
        raise ValueError("code acceptance receipt expected inventory mismatch")
    if expected_records is not None and list(expected_records) != records:
        raise ValueError("code acceptance receipt expected records mismatch")
    return payload


def code_acceptance_receipt_metrics(payload: object) -> dict[str, object]:
    receipt = validate_code_acceptance_receipt(payload)
    return {"code_acceptance_receipt_sha256": receipt["receipt_sha256"]}


def publish_code_acceptance_receipt(
    output: str | Path,
    payload: dict[str, object],
) -> Path:
    validated = validate_code_acceptance_receipt(payload)
    rendered = (
        json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(rendered) > MAX_CODE_ACCEPTANCE_RECEIPT_BYTES:
        raise ValueError("code acceptance receipt exceeds its size bound")
    destination = Path(output).resolve()
    publish_new_files(
        ((destination, rendered),),
        artifact_label="code acceptance receipt",
    )
    return destination
