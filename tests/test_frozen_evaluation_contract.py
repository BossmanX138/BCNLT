from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest
from test_full_model_acceptance_categories import (
    BINDING,
    CATEGORIES,
    FAMILIES,
    HELDOUT_MANIFEST,
    _category,
    _evaluation,
    _heldout_cases,
    _inference,
)

from bosscrafts_model.model_acceptance import (
    EXPECTED_HELDOUT_CELLS,
    verify_heldout_suite_manifest,
    verify_model_acceptance,
)

VALIDATION_DATABASE_ID = "9" * 64
TEST_DATABASE_ID = "a" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _plural_manifest() -> dict[str, object]:
    legacy = copy.deepcopy(HELDOUT_MANIFEST)
    categories = legacy["categories"]
    evaluation_cells: dict[str, object] = {}
    for split in ("validation", "test"):
        split_cells: dict[str, object] = {}
        for cell, count in EXPECTED_HELDOUT_CELLS.items():
            example_ids = sorted(
                hashlib.sha256(f"{split}:{cell}:{index}".encode()).hexdigest()
                for index in range(count)
            )
            split_cells[cell] = {
                "example_ids": example_ids,
                "example_ids_sha256": hashlib.sha256(
                    _canonical(example_ids)
                ).hexdigest(),
            }
        evaluation_cells[split] = split_cells
    first_language = next(iter(sorted(cell.split("|", 1)[0] for cell in EXPECTED_HELDOUT_CELLS)))
    for family, case in _heldout_cases().items():
        entry = evaluation_cells["test"][f"{first_language}|{family}"]
        example_ids = entry["example_ids"]
        example_ids[0] = case.example_id
        example_ids.sort()
        entry["example_ids_sha256"] = hashlib.sha256(
            _canonical(example_ids)
        ).hexdigest()
    unsigned = {
        "schema_version": "bosscrafts-model-heldout-suite-v2",
        "evaluation_task_database_identities_sha256": {
            "validation": VALIDATION_DATABASE_ID,
            "test": TEST_DATABASE_ID,
        },
        "parser_runtime_identity_sha256": legacy[
            "parser_runtime_identity_sha256"
        ],
        "tokenizer_sha256": legacy["tokenizer_sha256"],
        "expected_languages": legacy["expected_languages"],
        "structured_tasks": legacy["structured_tasks"],
        "evaluation_cells": evaluation_cells,
        "categories": categories,
    }
    return {
        **unsigned,
        "manifest_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def test_v2_suite_binds_both_split_databases_and_exact_evaluation_cases() -> None:
    manifest = _plural_manifest()
    suite = verify_heldout_suite_manifest(
        manifest,
        task_database_identities_sha256={
            "validation": VALIDATION_DATABASE_ID,
            "test": TEST_DATABASE_ID,
        },
    )

    assert suite.evaluation_task_database_identities_sha256 == {
        "validation": VALIDATION_DATABASE_ID,
        "test": TEST_DATABASE_ID,
    }
    assert set(suite.evaluation_example_ids) == {"validation", "test"}
    assert sum(
        len(example_ids)
        for example_ids in suite.evaluation_example_ids["validation"].values()
    ) == 5_376


def test_v2_suite_rejects_split_database_or_case_substitution() -> None:
    manifest = _plural_manifest()
    with pytest.raises(ValueError, match="lineage"):
        verify_heldout_suite_manifest(
            manifest,
            task_database_identities_sha256={
                "validation": "f" * 64,
                "test": TEST_DATABASE_ID,
            },
        )

    substituted = copy.deepcopy(manifest)
    validation_cells = substituted["evaluation_cells"]["validation"]
    first_cell = next(iter(sorted(validation_cells)))
    validation_cells[first_cell]["example_ids"][0] = "f" * 64
    unsigned = {
        key: value for key, value in substituted.items() if key != "manifest_sha256"
    }
    substituted["manifest_sha256"] = hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()
    with pytest.raises(ValueError, match="evaluation inventory"):
        verify_heldout_suite_manifest(
            substituted,
            task_database_identities_sha256={
                "validation": VALIDATION_DATABASE_ID,
                "test": TEST_DATABASE_ID,
            },
        )


def test_model_acceptance_tracks_separate_validation_and_test_databases() -> None:
    manifest = _plural_manifest()
    manifest_id = manifest["manifest_sha256"]
    database_ids = {
        "validation": VALIDATION_DATABASE_ID,
        "test": TEST_DATABASE_ID,
    }
    evaluations = []
    for split in ("validation", "test"):
        payload = _evaluation(split)
        payload["task_database_identity_sha256"] = database_ids[split]
        payload["evaluation_task_database_identity_sha256"] = database_ids[split]
        payload["checkpoint_evaluation_task_database_identity_sha256"] = database_ids[
            split
        ]
        payload["checkpoint_heldout_suite_manifest_sha256"] = manifest_id
        evaluations.append(payload)
    inferences = []
    for task in FAMILIES:
        payload = _inference(task)
        payload["task_database_identity_sha256"] = TEST_DATABASE_ID
        payload["evaluation_task_database_identity_sha256"] = TEST_DATABASE_ID
        payload["checkpoint_evaluation_task_database_identity_sha256"] = (
            TEST_DATABASE_ID
        )
        payload["checkpoint_heldout_suite_manifest_sha256"] = manifest_id
        inferences.append(payload)
    categories = []
    for category in CATEGORIES:
        payload = _category(category)
        payload["task_database_identity_sha256"] = TEST_DATABASE_ID
        payload["heldout_manifest_sha256"] = manifest_id
        body = {
            key: value for key, value in payload.items() if key != "artifact_sha256"
        }
        payload["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        categories.append(payload)

    result = verify_model_acceptance(
        evaluations,
        inferences,
        categories,
        binding=BINDING,
        heldout_manifest=manifest,
        heldout_cases=_heldout_cases(),
        task_database_identities_sha256=database_ids,
    )

    assert result.acceptance_passed
    assert result.evaluation_task_database_identities_sha256 == database_ids

    substituted_cases = _heldout_cases()
    first_task = next(iter(substituted_cases))
    substituted_cases[first_task] = replace(
        substituted_cases[first_task],
        example_id="f" * 64,
    )
    with pytest.raises(ValueError, match="frozen test inventory"):
        verify_model_acceptance(
            evaluations,
            inferences,
            categories,
            binding=BINDING,
            heldout_manifest=manifest,
            heldout_cases=substituted_cases,
            task_database_identities_sha256=database_ids,
        )
