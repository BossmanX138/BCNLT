"""Freeze exact evaluation and acceptance case inventories before training."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

CATEGORY_CASE_INVENTORY_SCHEMA = "bosscrafts-model-category-case-inventory-v2"
MAX_CATEGORY_CASE_INVENTORY_BYTES = 16 * 1024 * 1024
PACKED_ACCEPTANCE_CATEGORIES = (
    "tokenizer_fidelity",
    "semantic_fidelity",
    "raw_code",
    "execution_correctness",
)


@dataclass(frozen=True)
class PackedTaskReference:
    category_example_id: str
    task_example_id: str
    language: str
    family: str | None


@dataclass(frozen=True)
class CategoryCaseInventory:
    inventory_sha256: str
    test_task_database_identity_sha256: str
    category_example_ids: dict[str, tuple[str, ...]]
    packed_task_references: dict[str, tuple[PackedTaskReference, ...]]


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
) -> CategoryCaseInventory:
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
        "test_task_database_identity_sha256",
        "categories",
        "packed_task_references",
        "inventory_sha256",
    }:
        raise ValueError("category case inventory fields are invalid")
    unsigned = {key: value for key, value in payload.items() if key != "inventory_sha256"}
    if (
        payload.get("schema_version") != CATEGORY_CASE_INVENTORY_SCHEMA
        or not _is_sha256(payload.get("test_task_database_identity_sha256"))
        or not _is_sha256(payload.get("inventory_sha256"))
        or payload.get("inventory_sha256") != _sha256(unsigned)
    ):
        raise ValueError("category case inventory integrity mismatch")
    categories = payload.get("categories")
    if type(categories) is not dict or set(categories) != set(ACCEPTANCE_CATEGORIES):
        raise ValueError("category case inventory is incomplete")
    result: dict[str, tuple[str, ...]] = {}
    global_example_ids: set[str] = set()
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
        category_ids = tuple(example_ids)
        if global_example_ids.intersection(category_ids):
            raise ValueError("category case inventory repeats an identity")
        global_example_ids.update(category_ids)
        result[category] = category_ids

    packed_payload = payload.get("packed_task_references")
    if type(packed_payload) is not dict or set(packed_payload) != set(
        PACKED_ACCEPTANCE_CATEGORIES
    ):
        raise ValueError("packed category task references are incomplete")
    packed_references: dict[str, tuple[PackedTaskReference, ...]] = {}
    for category in PACKED_ACCEPTANCE_CATEGORIES:
        entry = packed_payload.get(category)
        if type(entry) is not dict or set(entry) != {
            "references",
            "references_sha256",
        }:
            raise ValueError(f"{category} packed task references are invalid")
        references_payload = entry.get("references")
        if (
            type(references_payload) is not list
            or len(references_payload) != len(result[category])
            or entry.get("references_sha256") != _sha256(references_payload)
        ):
            raise ValueError(f"{category} packed task references are invalid")
        expected_family = {
            "tokenizer_fidelity": None,
            "semantic_fidelity": "code_to_semantic",
            "raw_code": "semantic_to_code",
            "execution_correctness": "semantic_to_code",
        }[category]
        parsed: list[PackedTaskReference] = []
        task_ids: set[str] = set()
        for index, reference in enumerate(references_payload):
            if (
                type(reference) is not dict
                or set(reference)
                != {
                    "category_example_id",
                    "task_example_id",
                    "language",
                    "family",
                }
                or reference.get("category_example_id") != result[category][index]
                or not _is_sha256(reference.get("task_example_id"))
                or reference.get("language") not in EXPECTED_LANGUAGES
                or reference.get("family") != expected_family
            ):
                raise ValueError(f"{category} packed task references are invalid")
            task_id = str(reference["task_example_id"])
            if task_id in task_ids:
                raise ValueError(f"{category} packed task references repeat a task")
            task_ids.add(task_id)
            parsed.append(
                PackedTaskReference(
                    category_example_id=str(reference["category_example_id"]),
                    task_example_id=task_id,
                    language=str(reference["language"]),
                    family=(
                        str(reference["family"])
                        if reference["family"] is not None
                        else None
                    ),
                )
            )
        packed_references[category] = tuple(parsed)
    return CategoryCaseInventory(
        inventory_sha256=str(payload["inventory_sha256"]),
        test_task_database_identity_sha256=str(
            payload["test_task_database_identity_sha256"]
        ),
        category_example_ids=result,
        packed_task_references=packed_references,
    )


def validate_category_case_inventory_against_test_database(
    inventory: CategoryCaseInventory,
    *,
    test_task_database_identity_sha256: str,
    task_rows: Mapping[str, Mapping[str, object]],
) -> None:
    """Ground every packed category case in the exact audited test database."""
    if type(inventory) is not CategoryCaseInventory:
        raise TypeError("inventory must be a loaded CategoryCaseInventory")
    if (
        not _is_sha256(test_task_database_identity_sha256)
        or inventory.test_task_database_identity_sha256
        != test_task_database_identity_sha256
    ):
        raise ValueError("category inventory test database identity mismatch")
    expected_task_ids = {
        reference.task_example_id
        for references in inventory.packed_task_references.values()
        for reference in references
    }
    if set(task_rows) != expected_task_ids:
        raise ValueError("category inventory packed task set is incomplete")
    for references in inventory.packed_task_references.values():
        for reference in references:
            row = task_rows.get(reference.task_example_id)
            if (
                not isinstance(row, Mapping)
                or row.get("example_id") != reference.task_example_id
                or row.get("split") != "test"
                or row.get("language") != reference.language
                or (
                    reference.family is not None
                    and row.get("family") != reference.family
                )
            ):
                raise ValueError("category inventory packed task metadata mismatch")


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
    category_case_inventory: CategoryCaseInventory,
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
    packed_task_ids = sorted(
        {
            reference.task_example_id
            for references in category_case_inventory.packed_task_references.values()
            for reference in references
        }
    )
    packed_task_rows = {
        row["example_id"]: row
        for row in datasets["test"].iter_examples_by_id(
            packed_task_ids,
            split="test",
        )
    }
    validate_category_case_inventory_against_test_database(
        category_case_inventory,
        test_task_database_identity_sha256=audits["test"].database_identity_sha256,
        task_rows=packed_task_rows,
    )
    return assemble_heldout_suite_manifest(
        evaluation_task_database_identities_sha256={
            split: audit.database_identity_sha256
            for split, audit in audits.items()
        },
        evaluation_example_ids=evaluation_example_ids,
        category_example_ids=category_case_inventory.category_example_ids,
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
