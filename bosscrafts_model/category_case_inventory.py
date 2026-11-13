"""Validate content-addressed acceptance cases and freeze their ID inventory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .heldout_suite_builder import CATEGORY_CASE_INVENTORY_SCHEMA
from .immutable_artifacts import publish_new_files
from .model_acceptance import (
    ACCEPTANCE_CATEGORIES,
    EXPECTED_LANGUAGES,
    MAXIMUM_CATEGORY_EXAMPLES,
    MINIMUM_CATEGORY_EXAMPLES,
)

CATEGORY_CASE_SCHEMA = "bosscrafts-model-category-case-v1"
MAX_CATEGORY_CASE_FILE_BYTES = 128 * 1024 * 1024
_PACKED_CATEGORIES = {
    "tokenizer_fidelity",
    "semantic_fidelity",
    "raw_code",
    "execution_correctness",
}


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


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate category case key: {key}")
        payload[key] = value
    return payload


def _read_case_records(path: Path, category: str) -> list[dict[str, object]]:
    try:
        content = path.resolve().read_bytes()
        if not content or len(content) > MAX_CATEGORY_CASE_FILE_BYTES:
            raise ValueError(f"{category} category case file size is invalid")
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{category} category case file is unreadable") from error
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError(f"{category} category cases must be non-empty JSONL")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{category} category case line {line_number} is invalid JSON"
            ) from error
        if type(payload) is not dict:
            raise ValueError(f"{category} category case must be an object")
        records.append(payload)
    return records


def _validate_packed_case(category: str, case: dict[str, object]) -> str:
    required = {
        "task_database_identity_sha256",
        "task_example_id",
        "split",
        "language",
    }
    if category != "tokenizer_fidelity":
        required.add("family")
    if set(case) != required:
        raise ValueError(f"{category} packed case fields are invalid")
    if (
        not _is_sha256(case.get("task_database_identity_sha256"))
        or not _is_sha256(case.get("task_example_id"))
        or case.get("split") != "test"
        or case.get("language") not in EXPECTED_LANGUAGES
    ):
        raise ValueError(f"{category} packed case lineage is invalid")
    expected_family = {
        "semantic_fidelity": "code_to_semantic",
        "raw_code": "semantic_to_code",
        "execution_correctness": "semantic_to_code",
    }.get(category)
    if expected_family is not None and case.get("family") != expected_family:
        raise ValueError(f"{category} packed case family is invalid")
    return str(case["language"])


def _validate_case_payload(
    category: str,
    case_kind: object,
    case: object,
) -> str | None:
    if type(case) is not dict:
        raise ValueError(f"{category} case payload is invalid")
    if category in _PACKED_CATEGORIES:
        expected_kind = (
            "packed_tokenizer_roundtrip"
            if category == "tokenizer_fidelity"
            else "packed_model_task"
        )
        if case_kind != expected_kind:
            raise ValueError(f"{category} case kind is invalid")
        return _validate_packed_case(category, case)
    if category == "ordinary_instruction":
        if (
            case_kind != "ordinary_instruction"
            or set(case) != {"prompt", "expected_response"}
            or any(
                type(case.get(field)) is not str
                or not str(case[field]).strip()
                or len(str(case[field])) > 16_384
                for field in ("prompt", "expected_response")
            )
        ):
            raise ValueError("ordinary_instruction case payload is invalid")
        return None
    if category == "safety":
        if (
            case_kind != "safety_policy"
            or set(case) != {"prompt", "policy_id", "expected_action"}
            or type(case.get("prompt")) is not str
            or not str(case["prompt"]).strip()
            or len(str(case["prompt"])) > 16_384
            or type(case.get("policy_id")) is not str
            or not str(case["policy_id"]).strip()
            or case.get("expected_action") not in {"refuse", "safe_comply"}
        ):
            raise ValueError("safety case payload is invalid")
        return None
    if category == "regression":
        if (
            case_kind != "regression_suite"
            or set(case)
            != {"test_command", "minimum_test_count", "minimum_subtest_count"}
            or type(case.get("test_command")) is not str
            or not str(case["test_command"]).strip()
            or type(case.get("minimum_test_count")) is not int
            or int(case["minimum_test_count"]) < 1
            or type(case.get("minimum_subtest_count")) is not int
            or int(case["minimum_subtest_count"]) < 0
        ):
            raise ValueError("regression case payload is invalid")
        return None
    if category == "resources":
        if (
            case_kind != "resource_budget"
            or set(case)
            != {
                "vram_limit_bytes",
                "ram_limit_bytes",
                "minimum_tokens_per_second",
            }
            or type(case.get("vram_limit_bytes")) is not int
            or int(case["vram_limit_bytes"]) < 1
            or type(case.get("ram_limit_bytes")) is not int
            or int(case["ram_limit_bytes"]) < 1
            or type(case.get("minimum_tokens_per_second")) not in {int, float}
            or float(case["minimum_tokens_per_second"]) <= 0
        ):
            raise ValueError("resources case payload is invalid")
        return None
    raise AssertionError(f"unhandled acceptance category: {category}")


def build_category_case_inventory(
    case_files: Mapping[str, str | Path],
) -> dict[str, object]:
    """Validate every frozen case and return the self-hashed ID inventory."""
    if not isinstance(case_files, Mapping) or set(case_files) != set(
        ACCEPTANCE_CATEGORIES
    ):
        raise ValueError("category case files are incomplete")
    global_ids: set[str] = set()
    categories: dict[str, object] = {}
    for category in ACCEPTANCE_CATEGORIES:
        records = _read_case_records(Path(case_files[category]), category)
        if not MINIMUM_CATEGORY_EXAMPLES[category] <= len(
            records
        ) <= MAXIMUM_CATEGORY_EXAMPLES:
            raise ValueError(f"{category} category case count is invalid")
        example_ids: list[str] = []
        observed_languages: set[str] = set()
        for record in records:
            if set(record) != {
                "schema_version",
                "category",
                "case_kind",
                "case",
                "provenance",
                "example_id",
            }:
                raise ValueError(f"{category} category case fields are invalid")
            provenance = record.get("provenance")
            if (
                record.get("schema_version") != CATEGORY_CASE_SCHEMA
                or record.get("category") != category
                or type(provenance) is not dict
                or set(provenance) != {"source", "source_sha256"}
                or type(provenance.get("source")) is not str
                or not str(provenance["source"]).strip()
                or not _is_sha256(provenance.get("source_sha256"))
            ):
                raise ValueError(f"{category} category case provenance is invalid")
            body = {key: value for key, value in record.items() if key != "example_id"}
            example_id = record.get("example_id")
            if not _is_sha256(example_id) or example_id != _sha256(body):
                raise ValueError(f"{category} category case content identity is invalid")
            language = _validate_case_payload(
                category,
                record.get("case_kind"),
                record.get("case"),
            )
            if language is not None:
                observed_languages.add(language)
            if example_id in global_ids:
                raise ValueError("category case identity is duplicated")
            global_ids.add(example_id)
            example_ids.append(example_id)
        if example_ids != sorted(example_ids):
            raise ValueError(f"{category} category cases are not sorted by identity")
        if category in _PACKED_CATEGORIES and observed_languages != set(
            EXPECTED_LANGUAGES
        ):
            raise ValueError(f"{category} category cases do not cover every language")
        categories[category] = {
            "example_ids": example_ids,
            "example_ids_sha256": _sha256(example_ids),
        }
    unsigned: dict[str, object] = {
        "schema_version": CATEGORY_CASE_INVENTORY_SCHEMA,
        "categories": categories,
    }
    return {**unsigned, "inventory_sha256": _sha256(unsigned)}


def publish_category_case_inventory(
    output: str | Path,
    inventory: dict[str, object],
) -> Path:
    destination = Path(output).resolve()
    rendered = json.dumps(
        inventory,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    publish_new_files(
        ((destination, rendered.encode("utf-8")),),
        artifact_label="category case inventory",
    )
    return destination
