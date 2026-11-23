from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from pathlib import Path

import pytest

from bosscrafts_model.admission_runtime_delta import (
    compare_admission_runtime_delta,
    load_delta_receipt,
    publish_delta_receipt,
)
from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)

BASELINE_IDENTITY = "1" * 64
BASELINE_PYTHON_SPEC = "2" * 64


def _database(
    path: Path,
    parser_identity: str,
    *,
    payload: bytes = b"\x01\x00\x02\x00\x03\x00",
    source_catalog_sha256: str = "3" * 64,
    complete: bool = True,
    status: str = "admitted",
    document_id: str = "document-1",
) -> None:
    specs = dict(CURRENT_RUNTIME_SPEC_SHA256)
    if parser_identity == BASELINE_IDENTITY:
        specs["python"] = BASELINE_PYTHON_SPEC
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
            "schema_version": "bosscrafts-current-corpus-resumable-v5",
            "catalog_exhausted": str(complete).lower(),
            "expected_source_document_count": "1",
            "parser_runtime_identity_sha256": parser_identity,
            "host_code_identity_sha256": hashlib.sha256(parser_identity.encode()).hexdigest(),
            "source_catalog_sha256": source_catalog_sha256,
            "source_freeze_manifest_sha256": "4" * 64,
            "review_manifest_sha256": "5" * 64,
            "tokenizer_sha256": "6" * 64,
            "vocabulary_size": "65536",
            "semantic_spec_sha256": json.dumps(
                specs, sort_keys=True, separators=(",", ":")
            ),
        }
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)", sorted(metadata.items())
        )
        admitted = status == "admitted"
        connection.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                document_id,
                "7" * 64,
                "python",
                "example.py",
                status,
                None if admitted else "SOURCE_REJECTED",
                None if admitted else "invalid source",
                specs["python"] if admitted else None,
                parser_identity if admitted else None,
                hashlib.sha256(b"canonical").hexdigest() if admitted else None,
                zlib.compress(b"canonical", 6) if admitted else None,
                zlib.compress(payload, 6) if admitted else None,
                2 if admitted else 0,
                3 if admitted else 0,
                4 if admitted else 0,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def test_spec_and_runtime_only_delta_is_reusable(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY)
    _database(candidate, PARSER_RUNTIME_IDENTITY_SHA256)

    result = compare_admission_runtime_delta(baseline, candidate)

    assert result.reusable_document_count == 1
    assert result.reusable_admitted_count == 1
    assert result.changed_document_count == 0
    assert result.documents[0].classification == "reusable"
    assert result.documents[0].differing_fields == ()
    receipt = tmp_path / "delta.json"
    receipt_sha256 = publish_delta_receipt(result, receipt)
    assert load_delta_receipt(receipt)["manifest_sha256"] == receipt_sha256
    with pytest.raises(ValueError, match="already exists"):
        publish_delta_receipt(result, receipt)


def test_payload_delta_is_changed(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY)
    _database(
        candidate,
        PARSER_RUNTIME_IDENTITY_SHA256,
        payload=b"\x01\x00\x02\x00\x04\x00",
    )

    result = compare_admission_runtime_delta(baseline, candidate)

    assert result.reusable_document_count == 0
    assert result.changed_document_count == 1
    assert result.documents[0].differing_fields == ("semantic_ids_zlib",)


def test_identical_rejection_is_reusable(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY, status="rejected")
    _database(candidate, PARSER_RUNTIME_IDENTITY_SHA256, status="rejected")

    result = compare_admission_runtime_delta(baseline, candidate)

    assert result.reusable_rejected_count == 1
    assert result.documents[0].classification == "reusable"


def test_source_lineage_mismatch_is_rejected(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY)
    _database(
        candidate,
        PARSER_RUNTIME_IDENTITY_SHA256,
        source_catalog_sha256="9" * 64,
    )

    with pytest.raises(ValueError, match="source_catalog_sha256"):
        compare_admission_runtime_delta(baseline, candidate)


def test_nonmatching_document_id_sets_are_not_reused(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY, document_id="baseline-document")
    _database(
        candidate,
        PARSER_RUNTIME_IDENTITY_SHA256,
        document_id="candidate-document",
    )

    result = compare_admission_runtime_delta(baseline, candidate)

    assert result.reusable_document_count == 0
    assert result.baseline_only_document_count == 1
    assert result.candidate_only_document_count == 1
    assert [item.classification for item in result.documents] == [
        "baseline_only",
        "candidate_only",
    ]


def test_incomplete_admission_is_rejected(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY, complete=False)
    _database(candidate, PARSER_RUNTIME_IDENTITY_SHA256)

    with pytest.raises(ValueError, match="incomplete"):
        compare_admission_runtime_delta(baseline, candidate)


def test_noncanonical_delta_receipt_is_rejected(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    _database(baseline, BASELINE_IDENTITY)
    _database(candidate, PARSER_RUNTIME_IDENTITY_SHA256)
    receipt = tmp_path / "delta.json"
    publish_delta_receipt(compare_admission_runtime_delta(baseline, candidate), receipt)
    receipt.write_text(receipt.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical"):
        load_delta_receipt(receipt)
