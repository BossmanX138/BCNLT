from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bosscrafts_model.category_case_inventory import (
    CATEGORY_CASE_SCHEMA,
    build_category_case_inventory,
    publish_category_case_inventory,
)
from bosscrafts_model.heldout_suite_builder import load_category_case_inventory
from bosscrafts_model.model_acceptance import (
    ACCEPTANCE_CATEGORIES,
    EXPECTED_LANGUAGES,
    MINIMUM_CATEGORY_EXAMPLES,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _case_body(category: str, index: int) -> dict[str, object]:
    language = EXPECTED_LANGUAGES[index % len(EXPECTED_LANGUAGES)]
    database_id = "a" * 64
    task_id = hashlib.sha256(f"task:{category}:{index}".encode()).hexdigest()
    if category == "tokenizer_fidelity":
        case_kind = "packed_tokenizer_roundtrip"
        case = {
            "task_database_identity_sha256": database_id,
            "task_example_id": task_id,
            "split": "test",
            "language": language,
        }
    elif category in {"semantic_fidelity", "raw_code", "execution_correctness"}:
        case_kind = "packed_model_task"
        family = (
            "code_to_semantic"
            if category == "semantic_fidelity"
            else "semantic_to_code"
        )
        case = {
            "task_database_identity_sha256": database_id,
            "task_example_id": task_id,
            "split": "test",
            "language": language,
            "family": family,
        }
    elif category == "ordinary_instruction":
        case_kind = "ordinary_instruction"
        case = {
            "prompt": f"Return the exact decimal successor of {index}.",
            "expected_response": str(index + 1),
        }
    elif category == "safety":
        case_kind = "safety_policy"
        case = {
            "prompt": f"Safety scenario {index}",
            "policy_id": f"policy-{index % 8}",
            "expected_action": "refuse" if index % 2 else "safe_comply",
        }
    elif category == "regression":
        case_kind = "regression_suite"
        case = {
            "test_command": "python -W error -m pytest -q",
            "minimum_test_count": 551,
            "minimum_subtest_count": 195,
        }
    elif category == "resources":
        case_kind = "resource_budget"
        case = {
            "vram_limit_bytes": 12 * 1024**3,
            "ram_limit_bytes": 64 * 1024**3,
            "minimum_tokens_per_second": 1.0,
        }
    else:
        raise AssertionError(category)
    return {
        "schema_version": CATEGORY_CASE_SCHEMA,
        "category": category,
        "case_kind": case_kind,
        "case": case,
        "provenance": {
            "source": "test-fixture",
            "source_sha256": hashlib.sha256(
                f"source:{category}:{index}".encode()
            ).hexdigest(),
        },
    }


def _case(category: str, index: int) -> dict[str, object]:
    body = _case_body(category, index)
    return {**body, "example_id": hashlib.sha256(_canonical(body)).hexdigest()}


def _write_case_files(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for category in ACCEPTANCE_CATEGORIES:
        records = sorted(
            (
                _case(category, index)
                for index in range(MINIMUM_CATEGORY_EXAMPLES[category])
            ),
            key=lambda record: str(record["example_id"]),
        )
        path = root / f"{category}.jsonl"
        path.write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        paths[category] = path
    return paths


def test_builds_content_addressed_category_inventory(tmp_path: Path) -> None:
    case_files = _write_case_files(tmp_path)
    inventory = build_category_case_inventory(case_files)
    output = tmp_path / "category-inventory.json"
    publish_category_case_inventory(output, inventory)

    loaded = load_category_case_inventory(output)
    assert set(loaded) == set(ACCEPTANCE_CATEGORIES)
    assert all(
        len(loaded[category]) == MINIMUM_CATEGORY_EXAMPLES[category]
        for category in ACCEPTANCE_CATEGORIES
    )


def test_rejects_case_content_substitution_and_missing_language(
    tmp_path: Path,
) -> None:
    case_files = _write_case_files(tmp_path)
    semantic_path = case_files["semantic_fidelity"]
    records = [json.loads(line) for line in semantic_path.read_text().splitlines()]
    records[0]["case"]["family"] = "semantic_to_code"
    semantic_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content identity|family"):
        build_category_case_inventory(case_files)

    case_files = _write_case_files(tmp_path)
    records = [json.loads(line) for line in semantic_path.read_text().splitlines()]
    for record in records:
        record["case"]["language"] = EXPECTED_LANGUAGES[0]
        body = {key: value for key, value in record.items() if key != "example_id"}
        record["example_id"] = hashlib.sha256(_canonical(body)).hexdigest()
    records.sort(key=lambda record: record["example_id"])
    semantic_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="every language"):
        build_category_case_inventory(case_files)


def test_inventory_script_requires_every_category_file() -> None:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_category_case_inventory.py"),
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for category in ACCEPTANCE_CATEGORIES:
        assert f"--{category.replace('_', '-')}-cases" in completed.stdout
