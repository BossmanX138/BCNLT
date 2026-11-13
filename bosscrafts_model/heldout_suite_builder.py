"""Freeze exact evaluation and acceptance case inventories before training."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from .artifact_schemas import MODEL_HELDOUT_SUITE_SCHEMA_V2
from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from .immutable_artifacts import publish_new_files
from .model_acceptance import (
    ACCEPTANCE_CATEGORIES,
    EXPECTED_HELDOUT_CELLS,
    EXPECTED_LANGUAGES,
    MAXIMUM_CATEGORY_EXAMPLES,
    MINIMUM_CATEGORY_EXAMPLES,
    STRUCTURED_TASKS,
    verify_heldout_suite_manifest,
)
from .resumable_task_packing import ResumableTaskDataset, audit_task_database
from .tokenizer import EXPECTED_TOKENIZER_SHA256, NativeTokenizerBinding

CATEGORY_CASE_INVENTORY_SCHEMA = "bosscrafts-model-category-case-inventory-v1"
MAX_CATEGORY_CASE_INVENTORY_BYTES = 16 * 1024 * 1024


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
            raise ValueError(f"duplicate category inventory key: {key}")
        payload[key] = value
    return payload


def load_category_case_inventory(
    path: str | Path,
) -> dict[str, tuple[str, ...]]:
    """Load the pre-training category case catalog and verify its self-hash."""
    try:
        content = Path(path).resolve().read_bytes()
        if not content or len(content) > MAX_CATEGORY_CASE_INVENTORY_BYTES:
            raise ValueError("category case inventory size is invalid")
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("category case inventory is unreadable") from error
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "categories",
        "inventory_sha256",
    }:
        raise ValueError("category case inventory fields are invalid")
    unsigned = {key: value for key, value in payload.items() if key != "inventory_sha256"}
    if (
        payload.get("schema_version") != CATEGORY_CASE_INVENTORY_SCHEMA
        or not _is_sha256(payload.get("inventory_sha256"))
        or payload.get("inventory_sha256") != _sha256(unsigned)
    ):
        raise ValueError("category case inventory integrity mismatch")
    categories = payload.get("categories")
    if type(categories) is not dict or set(categories) != set(ACCEPTANCE_CATEGORIES):
        raise ValueError("category case inventory is incomplete")
    result: dict[str, tuple[str, ...]] = {}
    for category in ACCEPTANCE_CATEGORIES:
        entry = categories.get(category)
        if type(entry) is not dict or set(entry) != {
            "example_ids",
            "example_ids_sha256",
        }:
            raise ValueError(f"{category} category case inventory is invalid")
        example_ids = entry.get("example_ids")
        if (
            type(example_ids) is not list
            or not MINIMUM_CATEGORY_EXAMPLES[category]
            <= len(example_ids)
            <= MAXIMUM_CATEGORY_EXAMPLES
            or any(not _is_sha256(example_id) for example_id in example_ids)
            or example_ids != sorted(example_ids)
            or len(set(example_ids)) != len(example_ids)
            or entry.get("example_ids_sha256") != _sha256(example_ids)
        ):
            raise ValueError(f"{category} category case inventory is invalid")
        result[category] = tuple(example_ids)
    return result


def assemble_heldout_suite_manifest(
    *,
    evaluation_task_database_identities_sha256: Mapping[str, str],
    evaluation_example_ids: Mapping[str, Mapping[str, Sequence[str]]],
    category_example_ids: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Assemble and independently verify a schema-v2 held-out suite manifest."""
    evaluation_cells: dict[str, object] = {}
    for split in ("validation", "test"):
        split_cells = evaluation_example_ids.get(split)
        if not isinstance(split_cells, Mapping):
            raise ValueError("evaluation inventory is missing a split")
        evaluation_cells[split] = {
            cell: {
                "example_ids": list(split_cells.get(cell, ())),
                "example_ids_sha256": _sha256(list(split_cells.get(cell, ()))),
            }
            for cell in EXPECTED_HELDOUT_CELLS
        }
    categories = {
        category: {
            "example_ids": list(category_example_ids.get(category, ())),
            "example_ids_sha256": _sha256(
                list(category_example_ids.get(category, ()))
            ),
        }
        for category in ACCEPTANCE_CATEGORIES
    }
    unsigned: dict[str, object] = {
        "schema_version": MODEL_HELDOUT_SUITE_SCHEMA_V2,
        "evaluation_task_database_identities_sha256": dict(
            evaluation_task_database_identities_sha256
        ),
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "expected_languages": list(EXPECTED_LANGUAGES),
        "structured_tasks": list(STRUCTURED_TASKS),
        "evaluation_cells": evaluation_cells,
        "categories": categories,
    }
    manifest = {**unsigned, "manifest_sha256": _sha256(unsigned)}
    verify_heldout_suite_manifest(
        manifest,
        task_database_identities_sha256=(
            evaluation_task_database_identities_sha256
        ),
    )
    return manifest


def build_heldout_suite_manifest(
    binding: NativeTokenizerBinding,
    *,
    validation_database: str | Path,
    test_database: str | Path,
    category_example_ids: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Audit both immutable databases and freeze deterministic per-cell IDs."""
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    validation_path = Path(validation_database).resolve()
    test_path = Path(test_database).resolve()
    if validation_path == test_path or (
        validation_path.exists()
        and test_path.exists()
        and os.path.samefile(validation_path, test_path)
    ):
        raise ValueError("validation and test databases must be distinct")
    paths = {"validation": validation_path, "test": test_path}
    audits = {
        split: audit_task_database(binding, path) for split, path in paths.items()
    }
    datasets = {
        split: ResumableTaskDataset(binding, path) for split, path in paths.items()
    }
    for split, dataset in datasets.items():
        if dataset.example_count() != dataset.example_count(split=split):
            raise ValueError(f"{split} database contains rows from another split")
    evaluation_example_ids = {
        split: dataset.select_frozen_example_ids_by_cell(
            split=split,
            expected_cells=EXPECTED_HELDOUT_CELLS,
        )
        for split, dataset in datasets.items()
    }
    return assemble_heldout_suite_manifest(
        evaluation_task_database_identities_sha256={
            split: audit.database_identity_sha256
            for split, audit in audits.items()
        },
        evaluation_example_ids=evaluation_example_ids,
        category_example_ids=category_example_ids,
    )


def publish_heldout_suite_manifest(
    output: str | Path,
    manifest: dict[str, object],
) -> Path:
    destination = Path(output).resolve()
    rendered = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    publish_new_files(
        ((destination, rendered.encode("utf-8")),),
        artifact_label="frozen held-out suite manifest",
    )
    return destination
