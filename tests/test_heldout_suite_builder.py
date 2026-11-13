from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_frozen_evaluation_contract import _plural_manifest

from bosscrafts_model.heldout_suite_builder import (
    CATEGORY_CASE_INVENTORY_SCHEMA,
    assemble_heldout_suite_manifest,
    load_category_case_inventory,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _category_inventory(manifest: dict[str, object]) -> dict[str, object]:
    unsigned = {
        "schema_version": CATEGORY_CASE_INVENTORY_SCHEMA,
        "categories": copy.deepcopy(manifest["categories"]),
    }
    return {
        **unsigned,
        "inventory_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def test_assembles_the_verified_plural_manifest_from_frozen_inputs() -> None:
    expected = _plural_manifest()
    evaluation_example_ids = {
        split: {
            cell: tuple(entry["example_ids"])
            for cell, entry in split_cells.items()
        }
        for split, split_cells in expected["evaluation_cells"].items()
    }
    category_example_ids = {
        category: tuple(entry["example_ids"])
        for category, entry in expected["categories"].items()
    }

    assembled = assemble_heldout_suite_manifest(
        evaluation_task_database_identities_sha256=expected[
            "evaluation_task_database_identities_sha256"
        ],
        evaluation_example_ids=evaluation_example_ids,
        category_example_ids=category_example_ids,
    )

    assert assembled == expected


def test_category_inventory_is_bounded_hashed_and_duplicate_safe(
    tmp_path: Path,
) -> None:
    inventory = _category_inventory(_plural_manifest())
    path = tmp_path / "category-inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    loaded = load_category_case_inventory(path)
    assert set(loaded) == set(inventory["categories"])

    forged = copy.deepcopy(inventory)
    first_category = next(iter(forged["categories"]))
    forged["categories"][first_category]["example_ids"][0] = "f" * 64
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_category_case_inventory(path)

    path.write_text('{"schema_version":"one","schema_version":"two"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_category_case_inventory(path)


def test_builder_script_exposes_both_split_databases(tmp_path: Path) -> None:
    del tmp_path
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_frozen_heldout_suite.py"),
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--validation-database" in completed.stdout
    assert "--test-database" in completed.stdout
    assert "--category-inventory" in completed.stdout
