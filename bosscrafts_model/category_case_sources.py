"""Freeze pinned upstream benchmarks into immutable acceptance-case catalogs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .category_case_inventory import (
    CATEGORY_CASE_SCHEMA,
    _valid_provenance,
    _validate_case_payload,
)
from .immutable_artifacts import publish_new_files
from .model_acceptance import MAXIMUM_CATEGORY_EXAMPLES, MINIMUM_CATEGORY_EXAMPLES

MAX_UPSTREAM_CASE_SOURCE_BYTES = 128 * 1024 * 1024
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FrozenCategoryCases:
    output: Path
    record_count: int
    source_sha256: str
    revision: str


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate source key: {key}")
        value[key] = item
    return value


def _read_verified_source(
    source: str | Path,
    *,
    revision: str,
    expected_source_sha256: str,
) -> tuple[bytes, str]:
    if _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError("source revision must be an exact 40-character Git SHA")
    if _SHA256.fullmatch(expected_source_sha256) is None:
        raise ValueError("expected source SHA-256 is invalid")
    path = Path(source).resolve()
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError("upstream case source is unreadable") from error
    if not content or len(content) > MAX_UPSTREAM_CASE_SOURCE_BYTES:
        raise ValueError("upstream case source size is invalid")
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if observed_sha256 != expected_source_sha256:
        raise ValueError("upstream case source SHA-256 mismatch")
    return content, observed_sha256


def _build_record(
    *,
    category: str,
    case_kind: str,
    case: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, object]:
    if not _valid_provenance(provenance):
        raise ValueError("category case provenance is invalid")
    _validate_case_payload(category, case_kind, case)
    body: dict[str, object] = {
        "schema_version": CATEGORY_CASE_SCHEMA,
        "category": category,
        "case_kind": case_kind,
        "case": case,
        "provenance": provenance,
    }
    return {
        **body,
        "example_id": hashlib.sha256(_canonical(body)).hexdigest(),
    }


def _publish_records(
    output: str | Path,
    records: list[dict[str, object]],
    *,
    category: str,
    source_sha256: str,
    revision: str,
) -> FrozenCategoryCases:
    minimum = MINIMUM_CATEGORY_EXAMPLES[category]
    if not minimum <= len(records) <= MAXIMUM_CATEGORY_EXAMPLES:
        raise ValueError(f"{category} frozen source count is invalid")
    records.sort(key=lambda record: str(record["example_id"]))
    rendered = b"".join(_canonical(record) + b"\n" for record in records)
    destination = Path(output).resolve()
    publish_new_files(
        ((destination, rendered),),
        artifact_label=f"{category} frozen source cases",
    )
    return FrozenCategoryCases(
        output=destination,
        record_count=len(records),
        source_sha256=source_sha256,
        revision=revision,
    )


def freeze_ifeval_cases(
    source: str | Path,
    output: str | Path,
    *,
    revision: str,
    expected_source_sha256: str,
) -> FrozenCategoryCases:
    """Freeze the exact Google IFEval JSONL source as executable constraints."""
    content, source_sha256 = _read_verified_source(
        source,
        revision=revision,
        expected_source_sha256=expected_source_sha256,
    )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("IFEval source is not UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("IFEval source must be non-empty JSONL")
    provenance: dict[str, object] = {
        "source": "hf://datasets/google/IFEval/ifeval_input_data.jsonl",
        "revision": revision,
        "source_sha256": source_sha256,
        "license": "Apache-2.0",
        "attribution": "Google IFEval; Zhou et al., 2023",
    }
    records: list[dict[str, object]] = []
    source_keys: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise ValueError(f"IFEval line {line_number} is invalid JSON") from error
        if type(row) is not dict or set(row) != {
            "key",
            "prompt",
            "instruction_id_list",
            "kwargs",
        }:
            raise ValueError(f"IFEval line {line_number} fields are invalid")
        source_key = row.get("key")
        if type(source_key) is not int or source_key < 0:
            raise ValueError(f"IFEval line {line_number} key is invalid")
        if source_key in source_keys:
            raise ValueError("IFEval source contains a duplicate key")
        source_keys.add(source_key)
        case = {
            "source_key": source_key,
            "prompt": row.get("prompt"),
            "instruction_id_list": row.get("instruction_id_list"),
            "kwargs": row.get("kwargs"),
        }
        records.append(
            _build_record(
                category="ordinary_instruction",
                case_kind="ifeval_instruction",
                case=case,
                provenance=provenance,
            )
        )
    return _publish_records(
        output,
        records,
        category="ordinary_instruction",
        source_sha256=source_sha256,
        revision=revision,
    )


def freeze_aegis_test_cases(
    source: str | Path,
    output: str | Path,
    *,
    revision: str,
    expected_source_sha256: str,
) -> FrozenCategoryCases:
    """Freeze reconstructable NVIDIA Aegis 2.0 held-out prompt labels."""
    content, source_sha256 = _read_verified_source(
        source,
        revision=revision,
        expected_source_sha256=expected_source_sha256,
    )
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Aegis test source is invalid JSON") from error
    if type(payload) is not list or not payload:
        raise ValueError("Aegis test source must be a non-empty array")
    provenance: dict[str, object] = {
        "source": (
            "hf://datasets/nvidia/"
            "Aegis-AI-Content-Safety-Dataset-2.0/test.json"
        ),
        "revision": revision,
        "source_sha256": source_sha256,
        "license": "CC-BY-4.0",
        "attribution": "NVIDIA NeMo Guardrails team; AEGIS 2.0",
    }
    expected_fields = {
        "id",
        "reconstruction_id_if_redacted",
        "prompt",
        "response",
        "prompt_label",
        "response_label",
        "violated_categories",
        "prompt_label_source",
        "response_label_source",
    }
    records: list[dict[str, object]] = []
    source_ids: set[str] = set()
    for index, row in enumerate(payload):
        if type(row) is not dict or set(row) != expected_fields:
            raise ValueError(f"Aegis test row {index} fields are invalid")
        source_id = row.get("id")
        if type(source_id) is not str:
            raise ValueError(f"Aegis test row {index} ID is invalid")
        if source_id in source_ids:
            raise ValueError("Aegis test source contains a duplicate ID")
        source_ids.add(source_id)
        if row.get("reconstruction_id_if_redacted") is not None:
            continue
        categories = row.get("violated_categories")
        if type(categories) is not str:
            raise ValueError(f"Aegis test row {index} categories are invalid")
        violated_categories = (
            [category.strip() for category in categories.split(",")]
            if categories
            else []
        )
        case = {
            "source_id": source_id,
            "split": "test",
            "prompt": row.get("prompt"),
            "prompt_label": row.get("prompt_label"),
            "prompt_label_source": row.get("prompt_label_source"),
            "violated_categories": violated_categories,
        }
        records.append(
            _build_record(
                category="safety",
                case_kind="aegis_prompt_moderation",
                case=case,
                provenance=provenance,
            )
        )
    return _publish_records(
        output,
        records,
        category="safety",
        source_sha256=source_sha256,
        revision=revision,
    )
