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
    load_category_case_bundle,
    publish_category_case_bundle,
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
        case_kind = "ifeval_instruction"
        case = {
            "source_key": 1000 + index,
            "prompt": f"Return the decimal successor of {index} in lowercase.",
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{}],
        }
    elif category == "safety":
        case_kind = "aegis_prompt_moderation"
        case = {
            "source_id": hashlib.md5(
                f"aegis-test:{index}".encode(), usedforsecurity=False
            ).hexdigest(),
            "split": "test",
            "prompt": f"Safety scenario {index}",
            "prompt_label": "unsafe" if index % 2 else "safe",
            "prompt_label_source": "human",
            "violated_categories": (
                ["Violence"] if index % 2 else []
            ),
        }
    elif category == "regression":
        case_kind = "regression_suite"
        case = {
            "suite_id": "bosscrafts-pre-model-full-regression-v1",
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
            "revision": "b" * 40,
            "source_sha256": hashlib.sha256(
                f"source:{category}:{index}".encode()
            ).hexdigest(),
            "license": "BossCrafts-test-only",
            "attribution": "Synthetic BossCrafts test fixture",
        },
    }


def _case(category: str, index: int) -> dict[str, object]:
    body = _case_body(category, index)
    return {**body, "example_id": hashlib.sha256(_canonical(body)).hexdigest()}


def _write_case_files(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
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
    assert loaded.test_task_database_identity_sha256 == "a" * 64
    assert set(loaded.category_example_ids) == set(ACCEPTANCE_CATEGORIES)
    assert set(loaded.packed_task_references) == {
        "tokenizer_fidelity",
        "semantic_fidelity",
        "raw_code",
        "execution_correctness",
    }
    assert all(
        len(loaded.category_example_ids[category])
        == MINIMUM_CATEGORY_EXAMPLES[category]
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


def test_rejects_mixed_test_database_lineage_and_repeated_packed_tasks(
    tmp_path: Path,
) -> None:
    case_files = _write_case_files(tmp_path)
    tokenizer_path = case_files["tokenizer_fidelity"]
    records = [json.loads(line) for line in tokenizer_path.read_text().splitlines()]
    records[0]["case"]["task_database_identity_sha256"] = "b" * 64
    body = {key: value for key, value in records[0].items() if key != "example_id"}
    records[0]["example_id"] = hashlib.sha256(_canonical(body)).hexdigest()
    records.sort(key=lambda record: record["example_id"])
    tokenizer_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="one test task database"):
        build_category_case_inventory(case_files)

    case_files = _write_case_files(tmp_path)
    records = [json.loads(line) for line in tokenizer_path.read_text().splitlines()]
    records[1]["case"]["task_example_id"] = records[0]["case"]["task_example_id"]
    body = {key: value for key, value in records[1].items() if key != "example_id"}
    records[1]["example_id"] = hashlib.sha256(_canonical(body)).hexdigest()
    records.sort(key=lambda record: record["example_id"])
    tokenizer_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="repeats a packed task"):
        build_category_case_inventory(case_files)


@pytest.mark.parametrize(
    ("category", "case_kind", "case", "message"),
    [
        (
            "ordinary_instruction",
            "ordinary_instruction",
            {"prompt": "Say yes.", "expected_response": "yes"},
            "ordinary_instruction",
        ),
        (
            "ordinary_instruction",
            "ifeval_instruction",
            {
                "source_key": 1,
                "prompt": "Say yes in lowercase.",
                "instruction_id_list": ["change_case:english_lowercase"],
                "kwargs": [],
            },
            "ordinary_instruction",
        ),
        (
            "ordinary_instruction",
            "ifeval_instruction",
            {
                "source_key": 1,
                "prompt": "Invent a response.",
                "instruction_id_list": ["invented:constraint"],
                "kwargs": [{}],
            },
            "ordinary_instruction",
        ),
        (
            "safety",
            "aegis_prompt_moderation",
            {
                "source_id": "a" * 32,
                "split": "train",
                "prompt": "Unsafe fixture",
                "prompt_label": "unsafe",
                "prompt_label_source": "human",
                "violated_categories": ["Violence"],
            },
            "safety",
        ),
        (
            "regression",
            "regression_suite",
            {
                "test_command": "python -m pytest",
                "minimum_test_count": 1,
                "minimum_subtest_count": 0,
            },
            "regression",
        ),
    ],
)
def test_rejects_unexecutable_or_unpinned_category_case_shapes(
    tmp_path: Path,
    category: str,
    case_kind: str,
    case: dict[str, object],
    message: str,
) -> None:
    case_files = _write_case_files(tmp_path)
    path = case_files[category]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["case_kind"] = case_kind
    records[0]["case"] = case
    body = {key: value for key, value in records[0].items() if key != "example_id"}
    records[0]["example_id"] = hashlib.sha256(_canonical(body)).hexdigest()
    records.sort(key=lambda record: record["example_id"])
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        build_category_case_inventory(case_files)


def test_rejects_case_without_exact_revision_license_and_attribution(
    tmp_path: Path,
) -> None:
    case_files = _write_case_files(tmp_path)
    path = case_files["ordinary_instruction"]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["provenance"] = {
        "source": "https://huggingface.co/datasets/google/IFEval",
        "source_sha256": "a" * 64,
    }
    body = {key: value for key, value in records[0].items() if key != "example_id"}
    records[0]["example_id"] = hashlib.sha256(_canonical(body)).hexdigest()
    records.sort(key=lambda record: record["example_id"])
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance"):
        build_category_case_inventory(case_files)


def test_accepts_aegis_test_taxonomy_and_safe_needs_caution_rows(
    tmp_path: Path,
) -> None:
    case_files = _write_case_files(tmp_path)
    path = case_files["safety"]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["case"]["prompt_label"] = "safe"
    records[0]["case"]["violated_categories"] = [
        "Needs Caution",
        "Guns and Illegal Weapons",
        "Sexual (minor)",
        "High Risk Gov Decision Making",
    ]
    body = {key: value for key, value in records[0].items() if key != "example_id"}
    records[0]["example_id"] = hashlib.sha256(_canonical(body)).hexdigest()
    records.sort(key=lambda record: record["example_id"])
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    build_category_case_inventory(case_files)


def test_accepts_repeated_ifeval_instruction_types_with_distinct_arguments(
    tmp_path: Path,
) -> None:
    case_files = _write_case_files(tmp_path)
    path = case_files["ordinary_instruction"]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["case"]["instruction_id_list"] = [
        "keywords:frequency",
        "keywords:frequency",
    ]
    records[0]["case"]["kwargs"] = [
        {"relation": "at least", "keyword": "alpha", "frequency": 2},
        {"relation": "less than", "keyword": "beta", "frequency": 3},
    ]
    body = {key: value for key, value in records[0].items() if key != "example_id"}
    records[0]["example_id"] = hashlib.sha256(_canonical(body)).hexdigest()
    records.sort(key=lambda record: record["example_id"])
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    build_category_case_inventory(case_files)


def test_publishes_and_reloads_complete_immutable_category_case_bundle(
    tmp_path: Path,
) -> None:
    case_files = _write_case_files(tmp_path / "source")
    bundle_root = tmp_path / "bundle"
    published = publish_category_case_bundle(bundle_root, case_files)

    assert published.bundle_root == bundle_root.resolve()
    assert set(path.name for path in bundle_root.iterdir()) == {
        *(f"{category}.jsonl" for category in ACCEPTANCE_CATEGORIES),
        "inventory.json",
        "manifest.json",
    }
    loaded = load_category_case_bundle(bundle_root)
    assert loaded.manifest_sha256 == published.manifest_sha256
    assert loaded.inventory.inventory_sha256 == published.inventory.inventory_sha256
    assert set(loaded.case_files) == set(ACCEPTANCE_CATEGORIES)

    original_manifest = (bundle_root / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        publish_category_case_bundle(bundle_root, case_files)
    assert (bundle_root / "manifest.json").read_bytes() == original_manifest

    tokenizer_cases = bundle_root / "tokenizer_fidelity.jsonl"
    tokenizer_cases.write_bytes(tokenizer_cases.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="SHA-256|inventory"):
        load_category_case_bundle(bundle_root)


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
    assert "--output-directory" in completed.stdout
