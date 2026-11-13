from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bosscrafts_model.category_case_inventory import CATEGORY_CASE_SCHEMA
from bosscrafts_model.category_case_sources import (
    freeze_aegis_test_cases,
    freeze_ifeval_cases,
)
from bosscrafts_model.model_acceptance import MINIMUM_CATEGORY_EXAMPLES

IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
AEGIS_REVISION = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ifeval_source(path: Path) -> None:
    rows = [
        {
            "key": 1000 + index,
            "prompt": f"Write item {index} in lowercase.",
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{}],
        }
        for index in range(MINIMUM_CATEGORY_EXAMPLES["ordinary_instruction"])
    ]
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _aegis_source(path: Path, *, include_redacted: bool = False) -> None:
    count = MINIMUM_CATEGORY_EXAMPLES["safety"] + int(include_redacted)
    rows: list[dict[str, object]] = []
    for index in range(count):
        redacted = include_redacted and index == count - 1
        unsafe = bool(index % 2)
        rows.append(
            {
                "id": hashlib.md5(
                    f"aegis:{index}".encode(), usedforsecurity=False
                ).hexdigest(),
                "reconstruction_id_if_redacted": "external-row" if redacted else None,
                "prompt": "REDACTED" if redacted else f"Safety prompt {index}",
                "response": None,
                "prompt_label": "unsafe" if unsafe else "safe",
                "response_label": None,
                "violated_categories": "Violence" if unsafe else "",
                "prompt_label_source": "human",
                "response_label_source": None,
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_freezes_exact_ifeval_revision_as_immutable_content_addressed_cases(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ifeval_input_data.jsonl"
    output = tmp_path / "ordinary_instruction.jsonl"
    _ifeval_source(source)

    result = freeze_ifeval_cases(
        source,
        output,
        revision=IFEVAL_REVISION,
        expected_source_sha256=_sha256(source),
    )

    assert result.record_count == MINIMUM_CATEGORY_EXAMPLES["ordinary_instruction"]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["example_id"] for record in records] == sorted(
        record["example_id"] for record in records
    )
    assert {record["schema_version"] for record in records} == {
        CATEGORY_CASE_SCHEMA
    }
    assert {record["case_kind"] for record in records} == {"ifeval_instruction"}
    assert {record["provenance"]["revision"] for record in records} == {
        IFEVAL_REVISION
    }
    assert {record["provenance"]["license"] for record in records} == {
        "Apache-2.0"
    }
    original = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        freeze_ifeval_cases(
            source,
            output,
            revision=IFEVAL_REVISION,
            expected_source_sha256=_sha256(source),
        )
    assert output.read_bytes() == original


def test_freezes_only_reconstructable_aegis_test_prompts_with_human_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "test.json"
    output = tmp_path / "safety.jsonl"
    _aegis_source(source, include_redacted=True)

    result = freeze_aegis_test_cases(
        source,
        output,
        revision=AEGIS_REVISION,
        expected_source_sha256=_sha256(source),
    )

    assert result.record_count == MINIMUM_CATEGORY_EXAMPLES["safety"]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(record["case"]["split"] == "test" for record in records)
    assert all(
        record["case"]["prompt_label_source"] == "human" for record in records
    )
    assert all(record["case"]["prompt"] != "REDACTED" for record in records)
    assert {record["provenance"]["revision"] for record in records} == {
        AEGIS_REVISION
    }
    assert {record["provenance"]["license"] for record in records} == {
        "CC-BY-4.0"
    }


def test_source_freezers_reject_wrong_digest_duplicates_and_invalid_revision(
    tmp_path: Path,
) -> None:
    ifeval = tmp_path / "ifeval.jsonl"
    _ifeval_source(ifeval)
    with pytest.raises(ValueError, match="SHA-256"):
        freeze_ifeval_cases(
            ifeval,
            tmp_path / "wrong.jsonl",
            revision=IFEVAL_REVISION,
            expected_source_sha256="0" * 64,
        )

    first = json.loads(ifeval.read_text().splitlines()[0])
    ifeval.write_text(
        ifeval.read_text() + json.dumps(first) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        freeze_ifeval_cases(
            ifeval,
            tmp_path / "duplicate.jsonl",
            revision=IFEVAL_REVISION,
            expected_source_sha256=_sha256(ifeval),
        )

    aegis = tmp_path / "aegis.json"
    _aegis_source(aegis)
    with pytest.raises(ValueError, match="revision"):
        freeze_aegis_test_cases(
            aegis,
            tmp_path / "invalid-revision.jsonl",
            revision="main",
            expected_source_sha256=_sha256(aegis),
        )


def test_source_freezer_script_exposes_pinned_offline_imports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "freeze_category_case_source.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ifeval" in completed.stdout
    assert "aegis-test" in completed.stdout
    for subcommand in ("ifeval", "aegis-test"):
        help_result = subprocess.run(
            [sys.executable, str(script), subcommand, "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert help_result.returncode == 0, help_result.stderr
        assert "--source" in help_result.stdout
        assert "--revision" in help_result.stdout
        assert "--source-sha256" in help_result.stdout
        assert "--output" in help_result.stdout
