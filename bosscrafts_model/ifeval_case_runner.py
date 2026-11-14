"""Execute pinned IFEval cases and produce grounded acceptance records."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .category_case_inventory import (
    CATEGORY_CASE_SCHEMA,
    _valid_ifeval_case,
    _valid_provenance,
)
from .ifeval_runtime import (
    EXPECTED_IFEVAL_RUNTIME_MANIFEST_SHA256,
    IfevalRuntime,
    verify_ifeval_runtime,
)
from .immutable_artifacts import publish_new_files
from .model_response_generation import (
    responses_from_model_response_generation_receipt,
)

IFEVAL_WORKER_INPUT_SCHEMA = "bosscrafts-ifeval-worker-input-v1"
IFEVAL_WORKER_OUTPUT_SCHEMA = "bosscrafts-ifeval-worker-output-v1"
IFEVAL_BATCH_RECEIPT_SCHEMA = "bosscrafts-ifeval-batch-receipt-v2"
MAX_IFEVAL_BATCH_BYTES = 64 * 1024 * 1024
MAX_IFEVAL_RESPONSE_BYTES = 1024 * 1024
MAX_IFEVAL_CASES = 4_096
_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ifeval_worker.py"


@dataclass(frozen=True)
class IfevalBatchResult:
    runtime_manifest_sha256: str
    case_example_ids: tuple[str, ...]
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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate IFEval worker key: {key}")
        result[key] = value
    return result


def _validated_case_record(value: object) -> tuple[str, dict[str, object]]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "category",
        "case_kind",
        "case",
        "provenance",
        "example_id",
    }:
        raise ValueError("IFEval category case fields are invalid")
    case = value.get("case")
    provenance = value.get("provenance")
    body = {key: item for key, item in value.items() if key != "example_id"}
    example_id = value.get("example_id")
    if (
        value.get("schema_version") != CATEGORY_CASE_SCHEMA
        or value.get("category") != "ordinary_instruction"
        or value.get("case_kind") != "ifeval_instruction"
        or type(case) is not dict
        or not _valid_ifeval_case(case)
        or not _valid_provenance(provenance)
        or not _is_sha256(example_id)
        or example_id != hashlib.sha256(_canonical(body)).hexdigest()
    ):
        raise ValueError("IFEval category case integrity is invalid")
    return str(example_id), case


def _validated_responses(
    value: Mapping[str, str],
    expected_ids: tuple[str, ...],
    label: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(expected_ids):
        raise ValueError(f"{label} response inventory is invalid")
    result: dict[str, str] = {}
    for example_id in expected_ids:
        response = value.get(example_id)
        if (
            type(response) is not str
            or len(response.encode("utf-8")) > MAX_IFEVAL_RESPONSE_BYTES
        ):
            raise ValueError(f"{label} response is invalid")
        result[example_id] = response
    return result


def _worker_environment(runtime_root: Path) -> dict[str, str]:
    allowed = {
        "APPDATA",
        "HOME",
        "LOCALAPPDATA",
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "LD_LIBRARY_PATH",
    }
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in allowed
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "NLTK_DATA": str(runtime_root.resolve() / "nltk_data"),
        }
    )
    return environment


def _invoke_worker(
    runtime_root: Path,
    payload: dict[str, object],
    timeout: int,
) -> dict[str, object]:
    encoded = _canonical(payload)
    if not encoded or len(encoded) > MAX_IFEVAL_BATCH_BYTES:
        raise ValueError("IFEval worker input size is invalid")
    try:
        # The executable and worker path are locally resolved constants, argv is
        # passed without a shell, and the payload travels over bounded stdin.
        completed = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-I",
                str(_WORKER_SCRIPT),
                "--runtime-root",
                str(runtime_root),
            ],
            input=encoded,
            capture_output=True,
            timeout=timeout,
            check=False,
            shell=False,
            env=_worker_environment(runtime_root),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("IFEval worker failed to complete") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[:4_096].strip()
        raise ValueError(f"IFEval worker rejected the batch: {detail}")
    if not completed.stdout or len(completed.stdout) > MAX_IFEVAL_BATCH_BYTES:
        raise ValueError("IFEval worker output size is invalid")
    try:
        value = json.loads(
            completed.stdout.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("IFEval worker output is invalid JSON") from error
    if type(value) is not dict:
        raise ValueError("IFEval worker output must be an object")
    return value


def _valid_bool_list(value: object, expected_length: int) -> bool:
    return (
        type(value) is list
        and len(value) == expected_length
        and all(type(item) is bool for item in value)
    )


def _validated_worker_output(
    value: object,
    *,
    runtime: IfevalRuntime,
    cases: tuple[tuple[str, dict[str, object]], ...],
) -> tuple[dict[str, object], ...]:
    raw_results = value.get("results") if type(value) is dict else None
    if (
        type(value) is not dict
        or set(value)
        != {"schema_version", "runtime_manifest_sha256", "results"}
        or value.get("schema_version") != IFEVAL_WORKER_OUTPUT_SCHEMA
        or value.get("runtime_manifest_sha256") != runtime.manifest_sha256
        or type(raw_results) is not list
        or len(raw_results) != len(cases)
    ):
        raise ValueError("IFEval worker output contract is invalid")
    details: list[dict[str, object]] = []
    observed_ids: list[str] = []
    fields = {
        "example_id",
        "candidate_strict",
        "candidate_loose",
        "baseline_strict",
        "candidate_strict_instruction_results",
        "candidate_loose_instruction_results",
        "baseline_strict_instruction_results",
    }
    for result, (_expected_id, case) in zip(raw_results, cases, strict=True):
        instruction_ids = case["instruction_id_list"]
        if type(instruction_ids) is not list:
            raise AssertionError("validated IFEval instruction IDs are not a list")
        expected_length = len(instruction_ids)
        if (
            type(result) is not dict
            or set(result) != fields
            or not _is_sha256(result.get("example_id"))
            or any(
                type(result.get(field)) is not bool
                for field in (
                    "candidate_strict",
                    "candidate_loose",
                    "baseline_strict",
                )
            )
            or not _valid_bool_list(
                result.get("candidate_strict_instruction_results"), expected_length
            )
            or not _valid_bool_list(
                result.get("candidate_loose_instruction_results"), expected_length
            )
            or not _valid_bool_list(
                result.get("baseline_strict_instruction_results"), expected_length
            )
            or (result["candidate_strict"] and not result["candidate_loose"])
        ):
            raise ValueError("IFEval worker result is invalid")
        observed_ids.append(str(result["example_id"]))
        details.append(result)
    if observed_ids != [example_id for example_id, _ in cases]:
        raise ValueError("IFEval worker result inventory mismatch")
    return tuple(details)


def run_ifeval_cases(
    runtime_root: str | Path,
    case_records: Sequence[dict[str, object]],
    candidate_responses: Mapping[str, str],
    baseline_responses: Mapping[str, str],
    *,
    timeout_seconds: int = 300,
) -> IfevalBatchResult:
    """Score exact candidate/baseline responses with the pinned evaluator."""
    if (
        not isinstance(case_records, Sequence)
        or isinstance(case_records, (str, bytes))
        or not 1 <= len(case_records) <= MAX_IFEVAL_CASES
        or type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= 3_600
    ):
        raise ValueError("IFEval runner arguments are invalid")
    cases = tuple(_validated_case_record(record) for record in case_records)
    example_ids = tuple(example_id for example_id, _ in cases)
    if example_ids != tuple(sorted(example_ids)) or len(set(example_ids)) != len(
        example_ids
    ):
        raise ValueError("IFEval category case inventory is invalid")
    candidate = _validated_responses(
        candidate_responses, example_ids, "candidate"
    )
    baseline = _validated_responses(baseline_responses, example_ids, "baseline")
    runtime = verify_ifeval_runtime(runtime_root)
    worker_cases = [
        {
            "example_id": example_id,
            **case,
            "candidate_response": candidate[example_id],
            "baseline_response": baseline[example_id],
        }
        for example_id, case in cases
    ]
    payload: dict[str, object] = {
        "schema_version": IFEVAL_WORKER_INPUT_SCHEMA,
        "runtime_manifest_sha256": runtime.manifest_sha256,
        "cases": worker_cases,
    }
    details = _validated_worker_output(
        _invoke_worker(runtime.root, payload, timeout_seconds),
        runtime=runtime,
        cases=cases,
    )
    reverified = verify_ifeval_runtime(runtime.root)
    if reverified.manifest_sha256 != runtime.manifest_sha256:
        raise ValueError("IFEval runtime changed during evaluation")
    records = tuple(
        {
            "example_id": example_id,
            "strict_instruction_following": detail["candidate_strict"],
            "loose_instruction_following": detail["candidate_loose"],
            "baseline_strict_instruction_following": detail["baseline_strict"],
            "response_nonempty": bool(candidate[example_id].strip()),
        }
        for (example_id, _), detail in zip(cases, details, strict=True)
    )
    return IfevalBatchResult(
        runtime_manifest_sha256=runtime.manifest_sha256,
        case_example_ids=example_ids,
        records=records,
        details=details,
    )


_IFEVAL_RECEIPT_FIELDS = {
    "schema_version",
    "checkpoint_sha256",
    "checkpoint_step",
    "baseline_source",
    "baseline_evaluation_sha256",
    "runtime_manifest_sha256",
    "case_example_ids",
    "case_example_ids_sha256",
    "candidate_response_sha256_by_example_id",
    "candidate_response_nonempty_by_example_id",
    "candidate_response_manifest_sha256",
    "baseline_response_sha256_by_example_id",
    "baseline_response_manifest_sha256",
    "candidate_generation_receipt",
    "baseline_generation_receipt",
    "records",
    "records_sha256",
    "details",
    "details_sha256",
    "receipt_sha256",
}


def _validate_baseline_lineage(source: object, digest: object) -> None:
    if (
        type(source) is not str
        or not source
        or source != source.strip()
        or len(source.encode("utf-8")) > 256
    ):
        raise ValueError("IFEval baseline source is invalid")
    if digest is None:
        if source != "deterministic_random_initialization":
            raise ValueError("IFEval baseline lineage is invalid")
    elif not _is_sha256(digest) or source == "deterministic_random_initialization":
        raise ValueError("IFEval baseline lineage is invalid")


def _validate_receipt_results(
    example_ids: tuple[str, ...],
    records: object,
    details: object,
    response_nonempty: Mapping[str, bool],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if (
        type(records) is not list
        or type(details) is not list
        or len(records) != len(example_ids)
        or len(details) != len(example_ids)
    ):
        raise ValueError("IFEval receipt result inventory is invalid")
    record_fields = {
        "example_id",
        "strict_instruction_following",
        "loose_instruction_following",
        "baseline_strict_instruction_following",
        "response_nonempty",
    }
    detail_fields = {
        "example_id",
        "candidate_strict",
        "candidate_loose",
        "baseline_strict",
        "candidate_strict_instruction_results",
        "candidate_loose_instruction_results",
        "baseline_strict_instruction_results",
    }
    normalized_records: list[dict[str, object]] = []
    normalized_details: list[dict[str, object]] = []
    for example_id, record, detail in zip(
        example_ids, records, details, strict=True
    ):
        if (
            type(detail) is not dict
            or set(detail) != detail_fields
            or detail.get("example_id") != example_id
        ):
            raise ValueError("IFEval receipt detail is invalid")
        strict_results = detail.get("candidate_strict_instruction_results")
        loose_results = detail.get("candidate_loose_instruction_results")
        baseline_results = detail.get("baseline_strict_instruction_results")
        if (
            type(strict_results) is not list
            or type(loose_results) is not list
            or type(baseline_results) is not list
            or not 1 <= len(strict_results) <= 64
            or len(loose_results) != len(strict_results)
            or len(baseline_results) != len(strict_results)
            or any(type(item) is not bool for item in strict_results)
            or any(type(item) is not bool for item in loose_results)
            or any(type(item) is not bool for item in baseline_results)
            or type(detail.get("candidate_strict")) is not bool
            or type(detail.get("candidate_loose")) is not bool
            or type(detail.get("baseline_strict")) is not bool
            or detail["candidate_strict"] != all(strict_results)
            or detail["candidate_loose"] != all(loose_results)
            or detail["baseline_strict"] != all(baseline_results)
            or (detail["candidate_strict"] and not detail["candidate_loose"])
        ):
            raise ValueError("IFEval receipt detail is inconsistent")
        expected_record: dict[str, object] = {
            "example_id": example_id,
            "strict_instruction_following": detail["candidate_strict"],
            "loose_instruction_following": detail["candidate_loose"],
            "baseline_strict_instruction_following": detail["baseline_strict"],
            "response_nonempty": response_nonempty[example_id],
        }
        if type(record) is not dict or set(record) != record_fields or record != expected_record:
            raise ValueError("IFEval receipt record is inconsistent")
        normalized_records.append(record)
        normalized_details.append(detail)
    return normalized_records, normalized_details


def build_ifeval_batch_receipt(
    result: IfevalBatchResult,
    candidate_generation_receipt: dict[str, object],
    baseline_generation_receipt: dict[str, object],
) -> dict[str, object]:
    """Build a self-verifying receipt from one completed pinned IFEval run."""
    if result.runtime_manifest_sha256 != EXPECTED_IFEVAL_RUNTIME_MANIFEST_SHA256:
        raise ValueError("IFEval receipt lineage is invalid")
    example_ids = result.case_example_ids
    if (
        not 1 <= len(example_ids) <= MAX_IFEVAL_CASES
        or example_ids != tuple(sorted(example_ids))
        or len(set(example_ids)) != len(example_ids)
        or any(not _is_sha256(example_id) for example_id in example_ids)
    ):
        raise ValueError("IFEval receipt case inventory is invalid")
    candidate = responses_from_model_response_generation_receipt(
        candidate_generation_receipt,
        expected_category="ordinary_instruction",
        expected_source_type="checkpoint",
        expected_example_ids=example_ids,
    )
    baseline = responses_from_model_response_generation_receipt(
        baseline_generation_receipt,
        expected_category="ordinary_instruction",
        expected_source_type="deterministic_random_initialization",
        expected_example_ids=example_ids,
    )
    checkpoint_sha256 = candidate_generation_receipt.get("checkpoint_sha256")
    checkpoint_step = candidate_generation_receipt.get("checkpoint_step")
    if not _is_sha256(checkpoint_sha256) or type(checkpoint_step) is not int:
        raise ValueError("IFEval checkpoint generation lineage is invalid")
    _validate_generation_pair(
        candidate_generation_receipt,
        baseline_generation_receipt,
    )
    candidate_hashes = {
        example_id: hashlib.sha256(candidate[example_id].encode("utf-8")).hexdigest()
        for example_id in example_ids
    }
    baseline_hashes = {
        example_id: hashlib.sha256(baseline[example_id].encode("utf-8")).hexdigest()
        for example_id in example_ids
    }
    nonempty = {
        example_id: bool(candidate[example_id].strip()) for example_id in example_ids
    }
    records, details = _validate_receipt_results(
        example_ids,
        list(result.records),
        list(result.details),
        nonempty,
    )
    body: dict[str, object] = {
        "schema_version": IFEVAL_BATCH_RECEIPT_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "baseline_source": "deterministic_random_initialization",
        "baseline_evaluation_sha256": None,
        "runtime_manifest_sha256": result.runtime_manifest_sha256,
        "case_example_ids": list(example_ids),
        "case_example_ids_sha256": _sha256(list(example_ids)),
        "candidate_response_sha256_by_example_id": candidate_hashes,
        "candidate_response_nonempty_by_example_id": nonempty,
        "candidate_response_manifest_sha256": _sha256(candidate_hashes),
        "baseline_response_sha256_by_example_id": baseline_hashes,
        "baseline_response_manifest_sha256": _sha256(baseline_hashes),
        "candidate_generation_receipt": candidate_generation_receipt,
        "baseline_generation_receipt": baseline_generation_receipt,
        "records": records,
        "records_sha256": _sha256(records),
        "details": details,
        "details_sha256": _sha256(details),
    }
    return {**body, "receipt_sha256": _sha256(body)}


def validate_ifeval_batch_receipt(
    payload: object,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
    expected_example_ids: Sequence[str] | None = None,
    expected_records: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Recompute every receipt contract rather than trusting its pass fields."""
    if type(payload) is not dict or set(payload) != _IFEVAL_RECEIPT_FIELDS:
        raise ValueError("IFEval batch receipt fields are invalid")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    raw_ids = payload.get("case_example_ids")
    if (
        payload.get("schema_version") != IFEVAL_BATCH_RECEIPT_SCHEMA
        or not _is_sha256(payload.get("checkpoint_sha256"))
        or type(payload.get("checkpoint_step")) is not int
        or int(cast(int, payload["checkpoint_step"])) < 1
        or payload.get("runtime_manifest_sha256")
        != EXPECTED_IFEVAL_RUNTIME_MANIFEST_SHA256
        or type(raw_ids) is not list
        or not 1 <= len(raw_ids) <= MAX_IFEVAL_CASES
        or any(not _is_sha256(example_id) for example_id in raw_ids)
        or raw_ids != sorted(raw_ids)
        or len(set(raw_ids)) != len(raw_ids)
        or payload.get("case_example_ids_sha256") != _sha256(raw_ids)
        or payload.get("receipt_sha256") != _sha256(body)
    ):
        raise ValueError("IFEval batch receipt integrity mismatch")
    _validate_baseline_lineage(
        payload.get("baseline_source"), payload.get("baseline_evaluation_sha256")
    )
    if (
        payload.get("baseline_source") != "deterministic_random_initialization"
        or payload.get("baseline_evaluation_sha256") is not None
    ):
        raise ValueError("IFEval baseline must use the fixed random initialization")
    example_ids = tuple(cast(list[str], raw_ids))
    raw_candidate_generation = payload.get("candidate_generation_receipt")
    raw_baseline_generation = payload.get("baseline_generation_receipt")
    candidate = responses_from_model_response_generation_receipt(
        raw_candidate_generation,
        expected_category="ordinary_instruction",
        expected_source_type="checkpoint",
        expected_checkpoint_sha256=cast(str, payload["checkpoint_sha256"]),
        expected_checkpoint_step=cast(int, payload["checkpoint_step"]),
        expected_example_ids=example_ids,
    )
    baseline = responses_from_model_response_generation_receipt(
        raw_baseline_generation,
        expected_category="ordinary_instruction",
        expected_source_type="deterministic_random_initialization",
        expected_example_ids=example_ids,
    )
    if type(raw_candidate_generation) is not dict or type(raw_baseline_generation) is not dict:
        raise ValueError("IFEval generation receipts are invalid")
    _validate_generation_pair(raw_candidate_generation, raw_baseline_generation)
    derived_candidate_hashes = {
        example_id: hashlib.sha256(candidate[example_id].encode("utf-8")).hexdigest()
        for example_id in example_ids
    }
    derived_baseline_hashes = {
        example_id: hashlib.sha256(baseline[example_id].encode("utf-8")).hexdigest()
        for example_id in example_ids
    }
    derived_nonempty = {
        example_id: bool(candidate[example_id].strip()) for example_id in example_ids
    }
    candidate_hashes = payload.get("candidate_response_sha256_by_example_id")
    baseline_hashes = payload.get("baseline_response_sha256_by_example_id")
    nonempty = payload.get("candidate_response_nonempty_by_example_id")
    if (
        type(candidate_hashes) is not dict
        or type(baseline_hashes) is not dict
        or type(nonempty) is not dict
        or set(candidate_hashes) != set(example_ids)
        or set(baseline_hashes) != set(example_ids)
        or set(nonempty) != set(example_ids)
        or any(not _is_sha256(value) for value in candidate_hashes.values())
        or any(not _is_sha256(value) for value in baseline_hashes.values())
        or any(type(value) is not bool for value in nonempty.values())
        or candidate_hashes != derived_candidate_hashes
        or baseline_hashes != derived_baseline_hashes
        or nonempty != derived_nonempty
        or payload.get("candidate_response_manifest_sha256")
        != _sha256(candidate_hashes)
        or payload.get("baseline_response_manifest_sha256")
        != _sha256(baseline_hashes)
    ):
        raise ValueError("IFEval response manifest is invalid")
    records, details = _validate_receipt_results(
        example_ids,
        payload.get("records"),
        payload.get("details"),
        cast(dict[str, bool], nonempty),
    )
    if (
        payload.get("records_sha256") != _sha256(records)
        or payload.get("details_sha256") != _sha256(details)
    ):
        raise ValueError("IFEval receipt result integrity mismatch")
    if (
        expected_checkpoint_sha256 is not None
        and payload.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        raise ValueError("IFEval receipt checkpoint lineage mismatch")
    if (
        expected_checkpoint_step is not None
        and payload.get("checkpoint_step") != expected_checkpoint_step
    ):
        raise ValueError("IFEval receipt checkpoint lineage mismatch")
    if expected_example_ids is not None and example_ids != tuple(expected_example_ids):
        raise ValueError("IFEval receipt case inventory mismatch")
    if expected_records is not None and records != list(expected_records):
        raise ValueError("IFEval receipt records mismatch")
    return payload


def _validate_generation_pair(
    candidate: dict[str, object],
    baseline: dict[str, object],
) -> None:
    """Require both generations to use the same prompts, model shape, and runtime."""
    shared_fields = (
        "category",
        "generation_algorithm",
        "generation_task",
        "maximum_new_tokens",
        "model_fingerprint",
        "binding_sha256",
        "tokenizer_sha256",
        "case_bundle_manifest_sha256",
        "case_inventory_sha256",
        "device_type",
        "dtype",
        "example_ids",
    )
    if any(candidate.get(field) != baseline.get(field) for field in shared_fields):
        raise ValueError("IFEval generation runtime lineage mismatch")
    candidate_details = candidate.get("details")
    baseline_details = baseline.get("details")
    if type(candidate_details) is not list or type(baseline_details) is not list:
        raise ValueError("IFEval generation detail lineage is invalid")
    candidate_prompts = [
        (
            detail.get("example_id"),
            detail.get("prompt_sha256"),
            detail.get("prompt_token_count"),
            detail.get("prompt_ids_sha256"),
        )
        for detail in candidate_details
        if type(detail) is dict
    ]
    baseline_prompts = [
        (
            detail.get("example_id"),
            detail.get("prompt_sha256"),
            detail.get("prompt_token_count"),
            detail.get("prompt_ids_sha256"),
        )
        for detail in baseline_details
        if type(detail) is dict
    ]
    if (
        len(candidate_prompts) != len(candidate_details)
        or len(baseline_prompts) != len(baseline_details)
        or candidate_prompts != baseline_prompts
    ):
        raise ValueError("IFEval generation prompt lineage mismatch")


def publish_ifeval_batch_receipt(
    output: str | Path,
    payload: dict[str, object],
) -> Path:
    """Publish one validated receipt without replacing prior evidence."""
    validated = validate_ifeval_batch_receipt(payload)
    rendered = json.dumps(
        validated,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if len(rendered) > MAX_IFEVAL_BATCH_BYTES:
        raise ValueError("IFEval batch receipt size is invalid")
    destination = Path(output).resolve()
    publish_new_files(
        ((destination, rendered),),
        artifact_label="IFEval batch receipt",
    )
    return destination


def _score_response(
    evaluation_lib: Any,
    case: dict[str, object],
    response: str,
    *,
    loose: bool,
) -> tuple[bool, list[bool]]:
    source_key_value = case["source_key"]
    if type(source_key_value) is not int:
        raise AssertionError("validated IFEval source key is not an integer")
    source_key = source_key_value
    random.seed(source_key)
    input_example = evaluation_lib.InputExample(
        key=source_key,
        instruction_id_list=case["instruction_id_list"],
        prompt=case["prompt"],
        kwargs=case["kwargs"],
    )
    function = (
        evaluation_lib.test_instruction_following_loose
        if loose
        else evaluation_lib.test_instruction_following_strict
    )
    output = function(input_example, {case["prompt"]: response})
    raw_result_list = list(output.follow_instruction_list)
    instruction_ids = case["instruction_id_list"]
    if (
        type(output.follow_all_instructions) is not bool
        or type(instruction_ids) is not list
        or len(raw_result_list) != len(instruction_ids)
        or any(type(item) is not bool for item in raw_result_list)
    ):
        raise ValueError("pinned IFEval evaluator returned an invalid result")
    return output.follow_all_instructions, cast(list[bool], raw_result_list)


def _evaluate_payload(runtime_root: Path, payload: dict[str, object]) -> dict[str, object]:
    raw_cases = payload.get("cases")
    if (
        set(payload) != {"schema_version", "runtime_manifest_sha256", "cases"}
        or payload.get("schema_version") != IFEVAL_WORKER_INPUT_SCHEMA
        or not _is_sha256(payload.get("runtime_manifest_sha256"))
        or type(raw_cases) is not list
        or not 1 <= len(raw_cases) <= MAX_IFEVAL_CASES
    ):
        raise ValueError("IFEval worker input contract is invalid")
    parsed_cases: list[dict[str, object]] = []
    fields = {
        "example_id",
        "source_key",
        "prompt",
        "instruction_id_list",
        "kwargs",
        "candidate_response",
        "baseline_response",
    }
    for item in raw_cases:
        if type(item) is not dict or set(item) != fields:
            raise ValueError("IFEval worker case fields are invalid")
        case = {
            "source_key": item["source_key"],
            "prompt": item["prompt"],
            "instruction_id_list": item["instruction_id_list"],
            "kwargs": item["kwargs"],
        }
        if (
            not _is_sha256(item.get("example_id"))
            or not _valid_ifeval_case(case)
            or any(
                type(item.get(field)) is not str
                or len(str(item[field]).encode("utf-8")) > MAX_IFEVAL_RESPONSE_BYTES
                for field in ("candidate_response", "baseline_response")
            )
        ):
            raise ValueError("IFEval worker case is invalid")
        parsed_cases.append(item)
    evaluator_parent = str(runtime_root.resolve())
    previous_path = list(sys.path)
    previous_modules = set(sys.modules)
    previous_random_state = random.getstate()
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_nltk_path: list[str] | None = None
    previous_detector_seed: int | None = None
    sys.dont_write_bytecode = True
    try:
        sys.path.insert(0, evaluator_parent)
        import nltk

        previous_nltk_path = list(nltk.data.path)
        from langdetect import DetectorFactory

        previous_detector_seed = DetectorFactory.seed
        nltk.data.path[:] = [str(runtime_root / "nltk_data")]
        DetectorFactory.seed = 0
        evaluation_lib = importlib.import_module(
            "instruction_following_eval.evaluation_lib"
        )
        results: list[dict[str, object]] = []
        for item in parsed_cases:
            case = {
                "source_key": item["source_key"],
                "prompt": item["prompt"],
                "instruction_id_list": item["instruction_id_list"],
                "kwargs": item["kwargs"],
            }
            candidate_strict, candidate_strict_results = _score_response(
                evaluation_lib,
                case,
                str(item["candidate_response"]),
                loose=False,
            )
            candidate_loose, candidate_loose_results = _score_response(
                evaluation_lib,
                case,
                str(item["candidate_response"]),
                loose=True,
            )
            baseline_strict, baseline_strict_results = _score_response(
                evaluation_lib,
                case,
                str(item["baseline_response"]),
                loose=False,
            )
            results.append(
                {
                    "example_id": item["example_id"],
                    "candidate_strict": candidate_strict,
                    "candidate_loose": candidate_loose,
                    "baseline_strict": baseline_strict,
                    "candidate_strict_instruction_results": (
                        candidate_strict_results
                    ),
                    "candidate_loose_instruction_results": candidate_loose_results,
                    "baseline_strict_instruction_results": baseline_strict_results,
                }
            )
    finally:
        sys.path[:] = previous_path
        random.setstate(previous_random_state)
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if "nltk" in locals() and previous_nltk_path is not None:
            nltk.data.path[:] = previous_nltk_path
        if "DetectorFactory" in locals():
            DetectorFactory.seed = previous_detector_seed
        for name in set(sys.modules) - previous_modules:
            if name == "instruction_following_eval" or name.startswith(
                "instruction_following_eval."
            ):
                sys.modules.pop(name, None)
    return {
        "schema_version": IFEVAL_WORKER_OUTPUT_SCHEMA,
        "runtime_manifest_sha256": payload["runtime_manifest_sha256"],
        "results": results,
    }


def worker_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        content = sys.stdin.buffer.read(MAX_IFEVAL_BATCH_BYTES + 1)
        if not content or len(content) > MAX_IFEVAL_BATCH_BYTES:
            raise ValueError("IFEval worker input size is invalid")
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
        if type(payload) is not dict:
            raise ValueError("IFEval worker input must be an object")
        runtime = verify_ifeval_runtime(arguments.runtime_root)
        if payload.get("runtime_manifest_sha256") != runtime.manifest_sha256:
            raise ValueError("IFEval worker runtime identity mismatch")
        output = _evaluate_payload(runtime.root, payload)
        if verify_ifeval_runtime(runtime.root).manifest_sha256 != runtime.manifest_sha256:
            raise ValueError("IFEval worker runtime changed during evaluation")
        rendered = _canonical(output)
        if len(rendered) > MAX_IFEVAL_BATCH_BYTES:
            raise ValueError("IFEval worker output size is invalid")
        sys.stdout.buffer.write(rendered)
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
