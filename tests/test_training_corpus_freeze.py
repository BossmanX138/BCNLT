from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from bosscrafts_model.frozen_source_split import VerifiedFrozenSourceSplit
from bosscrafts_model.training_corpus_freeze import (
    TrainingCorpusFreezer,
    VerifiedTrainingSourceFreeze,
    assign_source_split,
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _key_for(split: str) -> str:
    for index in range(10_000):
        key = f"split-key-{split}-{index}"
        if assign_source_split(key) == split:
            return key
    raise AssertionError(f"could not find key for {split}")


def _document(
    document_id: str,
    *,
    repository: str,
    source_name: str = "bigcode/the-stack-dedup",
    language: str = "python",
) -> tuple[object, ...]:
    source = f"def {document_id.replace('-', '_')}(value):\n    return value + 1\n"
    return (
        document_id,
        "huggingface",
        source_name,
        "revision-1",
        language,
        repository,
        f"{document_id}.py",
        _sha256_bytes(source.encode("utf-8")),
        source,
        "MIT",
        "third_party",
        f"Attribution for {document_id}",
        "https://example.invalid/source",
        None,
        "registry",
        "parser",
        "grammar",
        "legacy-semantic-spec",
        "rules-v1",
    )


def _catalog(path: Path, rows: list[tuple[object, ...]]) -> str:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata(key,value) VALUES('schema_version','bcnlt-corpus-database-v3');
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                source_provider TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                canonical_language TEXT NOT NULL,
                repository_identity TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                content TEXT NOT NULL,
                license_spdx TEXT NOT NULL,
                ownership TEXT NOT NULL,
                attribution TEXT NOT NULL,
                upstream_url TEXT NOT NULL,
                dialect TEXT,
                registry_entry_sha256 TEXT NOT NULL,
                parser_bundle_sha256 TEXT NOT NULL,
                grammar_revision TEXT NOT NULL,
                semantic_spec_sha256 TEXT NOT NULL,
                extraction_rule_version TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()
    return _sha256_file(path)


def _heldout(root: Path, catalog_sha256: str, row: tuple[object, ...]) -> str:
    root.mkdir(parents=True)
    admitted = {
        "schema_version": "bosscrafts-current-corpus-record-v2",
        "document_id": row[0],
        "language": row[4],
        "source_sha256": row[7],
    }
    records_text = f"{_canonical(admitted)}\n"
    rejections_text = ""
    manifest = {
        "schema_version": "bosscrafts-current-corpus-manifest-v2",
        "admitted_count": 1,
        "rejected_count": 0,
        "records_file": "admitted.jsonl",
        "records_sha256": _sha256_bytes(records_text.encode("utf-8")),
        "rejections_file": "rejected.jsonl",
        "rejections_sha256": _sha256_bytes(b""),
        "source_catalog_sha256": catalog_sha256,
        "binding_sha256": "b" * 64,
        "tokenizer_sha256": "t" * 64,
        "parser_runtime_identity_sha256": "p" * 64,
        "training_ready": False,
    }
    manifest["manifest_sha256"] = _sha256_bytes(_canonical(manifest).encode("utf-8"))
    (root / "admitted.jsonl").write_bytes(records_text.encode("utf-8"))
    (root / "rejected.jsonl").write_bytes(rejections_text.encode("utf-8"))
    (root / "manifest.json").write_bytes(f"{_canonical(manifest)}\n".encode("utf-8"))
    return _sha256_file(root / "manifest.json")


def _qualification(path: Path) -> str:
    report = {
        "schema_version": "bosscrafts-combined-semantic-audit-qualification-v1",
        "audit_qualification_passed": True,
        "positive_control_count": 14,
        "positive_false_flag_count": 0,
        "negative_control_count": 14,
        "negative_detected_count": 14,
        "defect_metrics": {
            "event_ir_mismatch": {"detected": 2, "total": 2},
            "scope_mismatch": {"detected": 2, "total": 2},
            "span_mismatch": {"detected": 2, "total": 2},
            "surface_mismatch": {"detected": 2, "total": 2},
            "uncertainty_understated": {"detected": 2, "total": 2},
        },
        "training_ready": False,
    }
    report["report_sha256"] = _sha256_bytes(_canonical(report).encode("utf-8"))
    path.write_bytes(f"{_canonical(report)}\n".encode("utf-8"))
    return _sha256_file(path)


def test_freeze_is_deterministic_group_aware_and_never_training_ready(tmp_path: Path):
    heldout_id = _key_for("validation")
    train_repository = _key_for("train")
    validation_repository = _key_for("validation")
    synthetic_train_id = _key_for("train")
    rows = [
        _document(heldout_id, repository="reviewed-repository"),
        _document("same-reviewed-repository", repository="reviewed-repository"),
        _document("train-repository-document", repository=train_repository),
        _document("validation-document", repository=validation_repository),
        _document(
            synthetic_train_id,
            repository="nvidia/OpenCodeInstruct@revision-1",
            source_name="nvidia/OpenCodeInstruct",
        ),
    ]
    catalog = tmp_path / "catalog.sqlite"
    catalog_sha = _catalog(catalog, rows)
    heldout = tmp_path / "heldout"
    heldout_manifest_sha = _heldout(heldout, catalog_sha, rows[0])
    qualification = tmp_path / "qualification.json"
    qualification_sha = _qualification(qualification)

    first = TrainingCorpusFreezer().materialize(
        catalog,
        heldout,
        qualification,
        tmp_path / "freeze-a",
        expected_catalog_sha256=catalog_sha,
        expected_heldout_manifest_file_sha256=heldout_manifest_sha,
        expected_qualification_file_sha256=qualification_sha,
    )
    second = TrainingCorpusFreezer().materialize(
        catalog,
        heldout,
        qualification,
        tmp_path / "freeze-b",
        expected_catalog_sha256=catalog_sha,
        expected_heldout_manifest_file_sha256=heldout_manifest_sha,
        expected_qualification_file_sha256=qualification_sha,
    )

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.documents_sha256 == second.documents_sha256
    assert first.source_document_count == 5
    assert first.training_document_count == 2
    assert first.training_ready is False
    records = [
        json.loads(line)
        for line in (first.output_root / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["document_id"]: row for row in records}
    assert by_id[heldout_id]["split"] == "reserved_heldout"
    assert by_id["same-reviewed-repository"]["split"] == "reserved_heldout"
    assert by_id["train-repository-document"]["split"] == "train"
    assert by_id["validation-document"]["split"] == "validation"
    assert by_id[synthetic_train_id]["split"] == "train"
    assert by_id[synthetic_train_id]["split_group_kind"] == "document"

    manifest = json.loads((first.output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_freeze_complete"] is True
    assert manifest["repository_cross_split_count"] == 0
    assert manifest["training_ready"] is False
    assert "full current-runtime semantic admission is incomplete" in manifest["training_blockers"]

    verified = VerifiedTrainingSourceFreeze(
        first.output_root,
        catalog,
        expected_manifest_sha256=first.manifest_sha256,
    )
    assert verified.training_document_count == 2
    assert verified.source_catalog_sha256 == catalog_sha
    assert verified.qualification_file_sha256 == qualification_sha
    assert [candidate.document_id for candidate in verified.iter_training_candidates()] == [
        synthetic_train_id,
        "train-repository-document",
    ]
    validation = VerifiedFrozenSourceSplit(
        first.output_root,
        catalog,
        split="validation",
        expected_manifest_sha256=first.manifest_sha256,
    )
    assert validation.document_count == 1
    assert [candidate.document_id for candidate in validation.iter_candidates()] == [
        "validation-document"
    ]
    with pytest.raises(ValueError, match="frozen evaluation split"):
        VerifiedFrozenSourceSplit(
            first.output_root,
            catalog,
            split="train",
            expected_manifest_sha256=first.manifest_sha256,
        )


def test_freeze_rejects_tampered_qualification(tmp_path: Path):
    heldout_id = _key_for("test")
    row = _document(heldout_id, repository="heldout-repository")
    catalog = tmp_path / "catalog.sqlite"
    catalog_sha = _catalog(catalog, [row])
    heldout = tmp_path / "heldout"
    heldout_manifest_sha = _heldout(heldout, catalog_sha, row)
    qualification = tmp_path / "qualification.json"
    qualification_sha = _qualification(qualification)
    qualification.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="qualification file SHA-256 mismatch"):
        TrainingCorpusFreezer().materialize(
            catalog,
            heldout,
            qualification,
            tmp_path / "freeze",
            expected_catalog_sha256=catalog_sha,
            expected_heldout_manifest_file_sha256=heldout_manifest_sha,
            expected_qualification_file_sha256=qualification_sha,
        )


def test_freeze_rejects_catalog_content_hash_mismatch(tmp_path: Path):
    heldout_id = _key_for("validation")
    row = list(_document(heldout_id, repository="heldout-repository"))
    row[7] = "0" * 64
    catalog = tmp_path / "catalog.sqlite"
    catalog_sha = _catalog(catalog, [tuple(row)])
    heldout = tmp_path / "heldout"
    heldout_manifest_sha = _heldout(heldout, catalog_sha, tuple(row))
    qualification = tmp_path / "qualification.json"
    qualification_sha = _qualification(qualification)

    with pytest.raises(ValueError, match="catalog content SHA-256 mismatch"):
        TrainingCorpusFreezer().materialize(
            catalog,
            heldout,
            qualification,
            tmp_path / "freeze",
            expected_catalog_sha256=catalog_sha,
            expected_heldout_manifest_file_sha256=heldout_manifest_sha,
            expected_qualification_file_sha256=qualification_sha,
        )


def test_verified_freeze_rejects_tampered_document_inventory(tmp_path: Path):
    heldout_id = _key_for("validation")
    train_repository = _key_for("train")
    rows = [
        _document(heldout_id, repository="heldout-repository"),
        _document("train-document", repository=train_repository),
    ]
    catalog = tmp_path / "catalog.sqlite"
    catalog_sha = _catalog(catalog, rows)
    heldout = tmp_path / "heldout"
    heldout_manifest_sha = _heldout(heldout, catalog_sha, rows[0])
    qualification = tmp_path / "qualification.json"
    qualification_sha = _qualification(qualification)
    result = TrainingCorpusFreezer().materialize(
        catalog,
        heldout,
        qualification,
        tmp_path / "freeze",
        expected_catalog_sha256=catalog_sha,
        expected_heldout_manifest_file_sha256=heldout_manifest_sha,
        expected_qualification_file_sha256=qualification_sha,
    )
    with (result.output_root / "documents.jsonl").open("ab") as handle:
        handle.write(b"{}\n")

    with pytest.raises(ValueError, match="frozen document inventory SHA-256 mismatch"):
        VerifiedTrainingSourceFreeze(
            result.output_root,
            catalog,
            expected_manifest_sha256=result.manifest_sha256,
        )
