from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from bosscrafts_model.admission_runtime_equivalence import (
    compare_admission_runtime_equivalence,
    load_equivalence_receipt,
    publish_equivalence_receipt,
)
from bosscrafts_model.bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256

BASELINE_IDENTITY = "1" * 64


def _database(path: Path, parser_identity: str, *, payload: bytes = b"semantic") -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        connection.execute(
            """CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,content_sha256 TEXT NOT NULL,
                language TEXT NOT NULL,relative_path TEXT NOT NULL,
                status TEXT NOT NULL,reason_code TEXT,detail TEXT,
                semantic_spec_sha256 TEXT,parser_runtime_identity_sha256 TEXT,
                canonical_sha256 TEXT,canonical_jsonl_zlib BLOB,
                semantic_ids_zlib BLOB,semantic_record_count INTEGER NOT NULL DEFAULT 0,
                semantic_token_count INTEGER NOT NULL DEFAULT 0,
                raw_token_count INTEGER NOT NULL DEFAULT 0
            )"""
        )
        metadata = {
            "schema_version": "bosscrafts-current-corpus-resumable-v4",
            "parser_runtime_identity_sha256": parser_identity,
            "host_code_identity_sha256": hashlib.sha256(parser_identity.encode()).hexdigest(),
            "source_catalog_sha256": "2" * 64,
        }
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)", sorted(metadata.items())
        )
        connection.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "document-1",
                "3" * 64,
                "python",
                "example.py",
                "admitted",
                None,
                None,
                "4" * 64,
                parser_identity,
                "5" * 64,
                b"canonical",
                payload,
                2,
                3,
                4,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def test_exact_payload_equivalence_allows_only_runtime_provenance(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY)
    _database(candidate, PARSER_RUNTIME_IDENTITY_SHA256)

    result = compare_admission_runtime_equivalence(baseline, candidate)

    assert result.equivalent is True
    assert result.document_count == 1
    assert result.semantic_token_count == 3
    receipt = tmp_path / "receipt.json"
    receipt_sha256 = publish_equivalence_receipt(result, receipt)
    assert len(receipt_sha256) == 64
    assert load_equivalence_receipt(receipt)["manifest_sha256"] == receipt_sha256
    with pytest.raises(ValueError, match="already exists"):
        publish_equivalence_receipt(result, receipt)


def test_noncanonical_equivalence_receipt_is_rejected(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY)
    _database(candidate, PARSER_RUNTIME_IDENTITY_SHA256)
    receipt = tmp_path / "receipt.json"
    publish_equivalence_receipt(
        compare_admission_runtime_equivalence(baseline, candidate), receipt
    )
    receipt.write_text(receipt.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical"):
        load_equivalence_receipt(receipt)


def test_semantic_payload_difference_fails_equivalence(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY)
    _database(candidate, PARSER_RUNTIME_IDENTITY_SHA256, payload=b"changed")

    result = compare_admission_runtime_equivalence(baseline, candidate)

    assert result.equivalent is False
    assert result.mismatch_count == 1
    assert result.first_mismatch == {
        "scope": "documents",
        "ordinal": 1,
        "document_id": "document-1",
        "fields": ["semantic_ids_zlib"],
    }


def test_metadata_difference_outside_provenance_fails_equivalence(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY)
    _database(candidate, PARSER_RUNTIME_IDENTITY_SHA256)
    connection = sqlite3.connect(candidate)
    try:
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='source_catalog_sha256'", ("9" * 64,)
        )
        connection.commit()
    finally:
        connection.close()

    result = compare_admission_runtime_equivalence(baseline, candidate)

    assert result.equivalent is False
    assert result.first_mismatch == {
        "scope": "metadata",
        "key": "source_catalog_sha256",
        "baseline": "2" * 64,
        "candidate": "9" * 64,
    }
