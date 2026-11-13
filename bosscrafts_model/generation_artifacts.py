"""Strict readers for resumable generation and benchmark artifacts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

MAX_GENERATION_JSONL_LINE_BYTES = 16 * 1024 * 1024
HUMANEVAL_GENERATION_SCHEMA = "bosscrafts-openai-humaneval-generation-v2"
GENERATION_GATE_SCHEMA = "bosscrafts-strict-generation-gate-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HUMANEVAL_TASK_ID = re.compile(r"HumanEval/(0|[1-9][0-9]*)")


class HumanEvalGenerationRecord(TypedDict):
    schema_version: str
    run_identity_sha256: str
    task_index: int
    task_id: str
    entry_point: str
    prompt_sha256: str
    prompt_token_count: int
    generated_token_count: int
    completion_token_count: int
    generation_seconds: float
    tokens_per_second: float
    response_protocol_complete: bool
    raw_model_response: str
    raw_completion: str
    clean_completion: str


class GenerationGateRecord(TypedDict):
    schema_version: str
    run_identity_sha256: str
    example_id: str
    family: str
    prompt_token_count: int
    expected_token_count: int
    generated_token_count: int
    expected_ids: list[int]
    generated_ids: list[int]
    expected_protocol_text: str
    generated_protocol_text: str
    generated_payload_text: str
    protocol_complete: bool
    exact_match: bool
    target_token_accuracy: float
    generation_seconds: float
    tokens_per_second: float


_HUMANEVAL_RECORD_FIELDS = frozenset(HumanEvalGenerationRecord.__annotations__)
_GENERATION_GATE_RECORD_FIELDS = frozenset(GenerationGateRecord.__annotations__)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _iter_rows(path: Path, label: str) -> Iterator[tuple[int, dict[str, object]]]:
    with path.open("rb") as handle:
        line_number = 0
        while True:
            raw = handle.readline(MAX_GENERATION_JSONL_LINE_BYTES + 1)
            if not raw:
                return
            line_number += 1
            if len(raw) > MAX_GENERATION_JSONL_LINE_BYTES or not raw.endswith(b"\n"):
                raise ValueError(f"{label} record {line_number} has invalid bounded framing")
            if not raw.strip():
                continue
            try:
                value = json.loads(
                    raw.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_keys,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{label} record {line_number} is invalid JSON") from error
            if type(value) is not dict:
                raise ValueError(f"{label} record {line_number} is not an object")
            yield line_number, value


def _string(row: dict[str, object], field: str, label: str) -> str:
    value = row.get(field)
    if type(value) is not str:
        raise ValueError(f"{label} {field} must be a string")
    return value


def _nonempty_string(row: dict[str, object], field: str, label: str) -> str:
    value = _string(row, field, label)
    if not value:
        raise ValueError(f"{label} {field} must not be empty")
    return value


def _integer(
    row: dict[str, object],
    field: str,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    value = row.get(field)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} {field} must be an integer of at least {minimum}")
    return value


def _number(
    row: dict[str, object],
    field: str,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} {field} must be a finite number")
    if result < minimum or (strictly_positive and result <= 0.0):
        qualifier = "positive" if strictly_positive else f"at least {minimum}"
        raise ValueError(f"{label} {field} must be {qualifier}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} {field} must not exceed {maximum}")
    return result


def _boolean(row: dict[str, object], field: str, label: str) -> bool:
    value = row.get(field)
    if type(value) is not bool:
        raise ValueError(f"{label} {field} must be a boolean")
    return value


def _token_ids(
    row: dict[str, object], field: str, label: str, *, nonempty: bool
) -> list[int]:
    value = row.get(field)
    if type(value) is not list or (nonempty and not value):
        suffix = " nonempty" if nonempty else ""
        raise ValueError(f"{label} {field} must be a{suffix} list of token IDs")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError(f"{label} {field} must contain nonnegative integer token IDs")
    return value


def _identity(row: dict[str, object], expected: str, label: str) -> str:
    if not _SHA256.fullmatch(expected):
        raise ValueError("expected run identity is not a lowercase SHA-256")
    actual = _string(row, "run_identity_sha256", label)
    if actual != expected:
        raise ValueError(f"{label} run_identity_sha256 does not match this run")
    return actual


def _validate_rate(
    *, generated_count: int, seconds: float, rate: float, label: str
) -> None:
    expected = generated_count / seconds
    if not math.isclose(rate, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} tokens_per_second is inconsistent")


def load_humaneval_generation_records(
    path: Path, run_identity: str
) -> dict[str, HumanEvalGenerationRecord]:
    records: dict[str, HumanEvalGenerationRecord] = {}
    if not path.exists():
        return records
    for line_number, row in _iter_rows(path, "HumanEval generation"):
        label = f"HumanEval generation record {line_number}"
        if set(row) != _HUMANEVAL_RECORD_FIELDS:
            raise ValueError(f"{label} fields are invalid")
        if _string(row, "schema_version", label) != HUMANEVAL_GENERATION_SCHEMA:
            raise ValueError(f"{label} schema_version is invalid")
        identity = _identity(row, run_identity, label)
        task_index = _integer(row, "task_index", label)
        task_id = _nonempty_string(row, "task_id", label)
        match = _HUMANEVAL_TASK_ID.fullmatch(task_id)
        if match is None or int(match.group(1)) != task_index:
            raise ValueError(f"{label} task_id does not match task_index")
        if task_id in records:
            raise ValueError(f"{label} duplicates task_id {task_id}")
        entry_point = _nonempty_string(row, "entry_point", label)
        prompt_sha256 = _string(row, "prompt_sha256", label)
        if not _SHA256.fullmatch(prompt_sha256):
            raise ValueError(f"{label} prompt_sha256 is invalid")
        prompt_count = _integer(row, "prompt_token_count", label, minimum=1)
        generated_count = _integer(row, "generated_token_count", label)
        completion_count = _integer(row, "completion_token_count", label)
        if completion_count > generated_count:
            raise ValueError(f"{label} completion_token_count exceeds generated_token_count")
        seconds = _number(
            row, "generation_seconds", label, strictly_positive=True
        )
        rate = _number(row, "tokens_per_second", label)
        _validate_rate(
            generated_count=generated_count,
            seconds=seconds,
            rate=rate,
            label=label,
        )
        record = HumanEvalGenerationRecord(
            schema_version=HUMANEVAL_GENERATION_SCHEMA,
            run_identity_sha256=identity,
            task_index=task_index,
            task_id=task_id,
            entry_point=entry_point,
            prompt_sha256=prompt_sha256,
            prompt_token_count=prompt_count,
            generated_token_count=generated_count,
            completion_token_count=completion_count,
            generation_seconds=seconds,
            tokens_per_second=rate,
            response_protocol_complete=_boolean(
                row, "response_protocol_complete", label
            ),
            raw_model_response=_string(row, "raw_model_response", label),
            raw_completion=_string(row, "raw_completion", label),
            clean_completion=_string(row, "clean_completion", label),
        )
        records[task_id] = record
    return records


def load_generation_gate_records(
    path: Path, run_identity: str
) -> dict[str, GenerationGateRecord]:
    records: dict[str, GenerationGateRecord] = {}
    if not path.exists():
        return records
    for line_number, row in _iter_rows(path, "generation gate"):
        label = f"generation gate record {line_number}"
        if set(row) != _GENERATION_GATE_RECORD_FIELDS:
            raise ValueError(f"{label} fields are invalid")
        if _string(row, "schema_version", label) != GENERATION_GATE_SCHEMA:
            raise ValueError(f"{label} schema_version is invalid")
        identity = _identity(row, run_identity, label)
        example_id = _nonempty_string(row, "example_id", label)
        if example_id in records:
            raise ValueError(f"{label} duplicates example_id {example_id}")
        family = _nonempty_string(row, "family", label)
        prompt_count = _integer(row, "prompt_token_count", label, minimum=1)
        expected_count = _integer(row, "expected_token_count", label, minimum=1)
        generated_count = _integer(row, "generated_token_count", label)
        expected_ids = _token_ids(row, "expected_ids", label, nonempty=True)
        generated_ids = _token_ids(row, "generated_ids", label, nonempty=False)
        if expected_count != len(expected_ids):
            raise ValueError(f"{label} expected_token_count is inconsistent")
        if generated_count != len(generated_ids):
            raise ValueError(f"{label} generated_token_count is inconsistent")
        exact_match = _boolean(row, "exact_match", label)
        if exact_match != (generated_ids == expected_ids):
            raise ValueError(f"{label} exact_match is inconsistent")
        correct = sum(
            actual == expected
            for actual, expected in zip(generated_ids, expected_ids, strict=False)
        )
        accuracy = _number(
            row, "target_token_accuracy", label, maximum=1.0
        )
        if not math.isclose(
            accuracy,
            correct / expected_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label} target_token_accuracy is inconsistent")
        seconds = _number(
            row, "generation_seconds", label, strictly_positive=True
        )
        rate = _number(row, "tokens_per_second", label)
        _validate_rate(
            generated_count=generated_count,
            seconds=seconds,
            rate=rate,
            label=label,
        )
        record = GenerationGateRecord(
            schema_version=GENERATION_GATE_SCHEMA,
            run_identity_sha256=identity,
            example_id=example_id,
            family=family,
            prompt_token_count=prompt_count,
            expected_token_count=expected_count,
            generated_token_count=generated_count,
            expected_ids=expected_ids,
            generated_ids=generated_ids,
            expected_protocol_text=_string(row, "expected_protocol_text", label),
            generated_protocol_text=_string(row, "generated_protocol_text", label),
            generated_payload_text=_string(row, "generated_payload_text", label),
            protocol_complete=_boolean(row, "protocol_complete", label),
            exact_match=exact_match,
            target_token_accuracy=accuracy,
            generation_seconds=seconds,
            tokens_per_second=rate,
        )
        records[example_id] = record
    return records
