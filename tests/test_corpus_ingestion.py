from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bcnlt.corpus.database import CorpusDatabase
from bcnlt.corpus.filters import APPROVED_SPDX, evaluate_python_document
from bcnlt.corpus.sources import (
    SourceSpec,
    ingest_local_source,
    ingest_opencodeinstruct_stream,
    ingest_stack_stream,
)


def _code(statement: str = "answer = 42\n") -> str:
    padding = "# provenance-safe training fixture\n" * 20
    return statement + padding


def _stack_spec() -> SourceSpec:
    return SourceSpec(
        provider="huggingface",
        name="bigcode/the-stack-dedup",
        revision="17cad72c886a2858e08d4c349a00d6466f54df63",
        upstream_url="https://huggingface.co/datasets/bigcode/the-stack-dedup",
    )


def _nvidia_spec() -> SourceSpec:
    return SourceSpec(
        provider="huggingface",
        name="nvidia/OpenCodeInstruct",
        revision="8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
        upstream_url="https://huggingface.co/datasets/nvidia/OpenCodeInstruct",
    )


def _stack_row(**overrides):
    row = {
        "content": _code(),
        "path": "src/example.py",
        "repository": "owner/project",
        "license": "MIT",
    }
    row.update(overrides)
    return row


def test_source_revisions_must_be_immutable_pins():
    with pytest.raises(ValueError, match="revision"):
        SourceSpec(
            provider="huggingface",
            name="example/data",
            revision="main",
            upstream_url="https://example.invalid",
        )


def test_document_ids_include_source_revision_repository_path_and_content(tmp_path: Path):
    db = CorpusDatabase(tmp_path / "corpus.sqlite3")
    first = ingest_stack_stream(db, _stack_spec(), [_stack_row()])[0]
    changed = ingest_stack_stream(
        db,
        SourceSpec(
            provider="huggingface",
            name="bigcode/the-stack-dedup",
            revision="a" * 40,
            upstream_url="https://huggingface.co/datasets/bigcode/the-stack-dedup",
        ),
        [_stack_row()],
    )[0]

    assert first.document_id != changed.document_id
    assert first.content_sha256 == hashlib.sha256(_code().encode()).hexdigest()
    assert first.repository_identity == "owner/project"
    assert first.relative_path == "src/example.py"


def test_ingestion_is_idempotent_and_preserves_provenance(tmp_path: Path):
    db = CorpusDatabase(tmp_path / "corpus.sqlite3")
    rows = [_stack_row()]

    ingest_stack_stream(db, _stack_spec(), rows)
    ingest_stack_stream(db, _stack_spec(), rows)
    stored = db.documents()

    assert len(stored) == 1
    assert stored[0].license_spdx == "MIT"
    assert stored[0].ownership == "third_party"
    assert stored[0].upstream_url.endswith("bigcode/the-stack-dedup")


def test_transaction_rolls_back_and_restart_admits_cleanly(tmp_path: Path):
    db = CorpusDatabase(tmp_path / "corpus.sqlite3")
    record = ingest_stack_stream(db, _stack_spec(), [_stack_row()])[0]
    db.clear()

    with pytest.raises(RuntimeError):
        with db.transaction():
            db.admit(record)
            raise RuntimeError("interrupted")

    assert db.document_count() == 0
    with db.transaction():
        db.admit(record)
    assert db.document_count() == 1


@pytest.mark.parametrize("license_spdx", sorted(APPROVED_SPDX))
def test_stack_admits_each_approved_explicit_license(tmp_path: Path, license_spdx: str):
    db = CorpusDatabase(tmp_path / f"{license_spdx}.sqlite3")

    admitted = ingest_stack_stream(
        db, _stack_spec(), [_stack_row(license=license_spdx)]
    )

    assert len(admitted) == 1
    assert admitted[0].license_spdx == license_spdx


@pytest.mark.parametrize(
    "license_value",
    [None, "", "unknown", "MIT,GPL-3.0", "GPL-2.0", "GPL-3.0", "MPL-2.0"],
)
def test_stack_rejects_absent_unknown_mixed_or_copyleft_licenses(
    tmp_path: Path, license_value
):
    db = CorpusDatabase(tmp_path / "corpus.sqlite3")

    admitted = ingest_stack_stream(
        db, _stack_spec(), [_stack_row(license=license_value)]
    )

    assert admitted == []
    assert db.rejections()[0]["reason_code"] == "license_not_approved"


@pytest.mark.parametrize(
    ("path", "content", "reason"),
    [
        ("vendor/lib.py", _code(), "generated_or_vendor_path"),
        ("src/generated_client.py", _code(), "generated_or_vendor_path"),
        ("src/tiny.py", "x = 1\n", "size_out_of_range"),
        pytest.param(
            "src/huge.py",
            "#" + ("x" * (201 * 1024)),
            "size_out_of_range",
            id="oversize",
        ),
        pytest.param(
            "src/long_line.py",
            "value = '" + ("x" * 2100) + "'\n",
            "pathological_line_length",
            id="long-line",
        ),
        ("src/control.py", _code("value = '\u0001'\n"), "control_character"),
        ("src/broken.py", _code("def broken(:\n"), "python_parse_failure"),
        (
            "src/private.py",
            _code("SECRET = '''-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----'''\n"),
            "secret_detected",
        ),
        ("src/aws.py", _code("token = 'AKIAABCDEFGHIJKLMNOP'\n"), "secret_detected"),
        (
            "src/github.py",
            _code("token = 'ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456'\n"),
            "secret_detected",
        ),
        (
            "src/openai.py",
            _code("token = 'sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456'\n"),
            "secret_detected",
        ),
        (
            "src/entropy.py",
            _code("api_secret = 'QWxhZGRpbjpvcGVuIHNlc2FtZTEyMzQ1Njc4OTA='\n"),
            "secret_detected",
        ),
    ],
)
def test_quality_and_secret_filters_write_stable_rejections(
    tmp_path: Path, path: str, content: str, reason: str
):
    db = CorpusDatabase(tmp_path / "corpus.sqlite3")

    admitted = ingest_stack_stream(
        db, _stack_spec(), [_stack_row(path=path, content=content)]
    )

    assert admitted == []
    assert db.rejections()[0]["reason_code"] == reason


def test_holdout_hashes_are_rejected_before_persistence(tmp_path: Path):
    content = _code()
    decision = evaluate_python_document(
        relative_path="src/example.py",
        content=content,
        license_value="MIT",
        holdout_hashes={hashlib.sha256(content.encode()).hexdigest()},
    )

    assert decision.accepted is False
    assert decision.reason_code == "evaluation_holdout_collision"


def test_local_owner_source_records_explicit_ownership(tmp_path: Path):
    source = tmp_path / "owned"
    source.mkdir()
    (source / "module.py").write_text(_code(), encoding="utf-8")
    db = CorpusDatabase(tmp_path / "corpus.sqlite3")
    spec = SourceSpec(
        provider="local",
        name="BossCrafts-owned",
        revision="c" * 64,
        upstream_url="owner-provided",
    )

    admitted = ingest_local_source(
        db,
        spec,
        source,
        repository_identity="BossCrafts/owned",
        ownership="owner_provided",
        license_spdx="owner_provided",
    )

    assert len(admitted) == 1
    assert admitted[0].ownership == "owner_provided"
    assert admitted[0].license_spdx == "owner_provided"


def test_opencodeinstruct_requires_parseable_perfect_passing_python(tmp_path: Path):
    db = CorpusDatabase(tmp_path / "corpus.sqlite3")
    rows = [
        {
            "id": "pass",
            "output": f"```python\n{_code()}\n```",
            "average_test_score": 1.0,
            "test_execution_status": ["pass", "pass"],
        },
        {
            "id": "imperfect",
            "output": _code(),
            "average_test_score": 0.9,
            "test_execution_status": ["pass"],
        },
        {
            "id": "failed",
            "output": _code(),
            "average_test_score": 1.0,
            "test_execution_status": ["pass", "fail"],
        },
        {
            "id": "invalid",
            "output": _code("def nope(:\n"),
            "average_test_score": 1.0,
            "test_execution_status": ["pass"],
        },
    ]

    admitted = ingest_opencodeinstruct_stream(db, _nvidia_spec(), rows)

    assert [record.relative_path for record in admitted] == ["pass.py"]
    assert admitted[0].license_spdx == "CC-BY-4.0"
    assert admitted[0].attribution == "NVIDIA OpenCodeInstruct"
    assert {item["reason_code"] for item in db.rejections()} == {
        "instruction_tests_not_perfect",
        "instruction_test_failure",
        "python_parse_failure",
    }


def test_opencodeinstruct_accepts_pinned_plural_string_status_schema(tmp_path: Path):
    db = CorpusDatabase(tmp_path / "corpus.sqlite3")
    rows = [
        {
            "id": "pinned-schema",
            "output": f"```python\n{_code()}\n```",
            "average_test_score": "1",
            "tests_execution_status": '["pass", "pass"]',
        }
    ]

    admitted = ingest_opencodeinstruct_stream(db, _nvidia_spec(), rows)

    assert len(admitted) == 1
    assert admitted[0].relative_path == "pinned-schema.py"
    assert admitted[0].content == _code()
