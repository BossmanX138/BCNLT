from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from bosscrafts_model.admission_runtime_equivalence import (
    compare_admission_runtime_equivalence,
    publish_equivalence_receipt,
)
from bosscrafts_model.bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from bosscrafts_model.eligibility_runtime_transfer import (
    TRANSFER_POLICY,
    transfer_runtime_equivalent_eligibility,
)
from bosscrafts_model.resumable_training_semantic_audit import (
    ResumableTrainingSemanticAudit,
)

BASELINE_RUNTIME = "1" * 64
CATALOG_SHA256 = "2" * 64
FREEZE_SHA256 = "3" * 64
CANONICAL_BLOB = b"canonical-zlib"
SEMANTIC_BLOB = b"semantic-zlib"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _database(path: Path, parser_runtime: str, *, source_split: str = "test") -> None:
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
            "parser_runtime_identity_sha256": parser_runtime,
            "host_code_identity_sha256": hashlib.sha256(
                parser_runtime.encode("utf-8")
            ).hexdigest(),
            "source_catalog_sha256": CATALOG_SHA256,
            "source_freeze_manifest_sha256": FREEZE_SHA256,
        }
        if source_split != "train":
            metadata["frozen_source_split"] = source_split
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)", sorted(metadata.items())
        )
        connection.executemany(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                (
                    "eligible",
                    "4" * 64,
                    "python",
                    "eligible.py",
                    "admitted",
                    None,
                    None,
                    "5" * 64,
                    parser_runtime,
                    "6" * 64,
                    CANONICAL_BLOB,
                    SEMANTIC_BLOB,
                    2,
                    3,
                    4,
                ),
                (
                    "excluded-audit",
                    "7" * 64,
                    "javascript",
                    "excluded.js",
                    "admitted",
                    None,
                    None,
                    "8" * 64,
                    parser_runtime,
                    "9" * 64,
                    b"canonical-two",
                    b"semantic-two",
                    5,
                    6,
                    7,
                ),
                (
                    "excluded-source",
                    "a" * 64,
                    "ruby",
                    "rejected.rb",
                    "rejected",
                    "parse_failure",
                    "fixture",
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    2,
                ),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _admission_identity(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        return ResumableTrainingSemanticAudit._admission_identity(connection, metadata)
    finally:
        connection.close()


def _eligibility(root: Path, baseline: Path, *, source_split: str = "test") -> None:
    root.mkdir()
    eligible = {
        "schema_version": "bosscrafts-training-eligible-document-v1",
        "document_id": "eligible",
        "language": "python",
        "canonical_sha256": "6" * 64,
        "canonical_payload_sha256": hashlib.sha256(CANONICAL_BLOB).hexdigest(),
        "semantic_payload_sha256": hashlib.sha256(SEMANTIC_BLOB).hexdigest(),
        "semantic_record_count": 2,
        "audit_receipt_sha256": "b" * 64,
    }
    excluded = (
        {
            "schema_version": "bosscrafts-training-eligibility-exclusion-v1",
            "document_id": "excluded-audit",
            "reason": "semantic_audit_findings",
            "semantic_record_count": 5,
            "finding_count": 1,
        },
        {
            "schema_version": "bosscrafts-training-eligibility-exclusion-v1",
            "document_id": "excluded-source",
            "reason": "source_rejected",
            "semantic_record_count": 0,
            "finding_count": 0,
        },
    )
    eligible_text = f"{_canonical(eligible)}\n"
    excluded_text = "".join(f"{_canonical(row)}\n" for row in excluded)
    unsigned = {
        "schema_version": "bosscrafts-training-eligibility-manifest-v1",
        "eligibility_complete": True,
        "source_audit_identity_sha256": "c" * 64,
        "source_admission_identity_sha256": _admission_identity(baseline),
        "source_freeze_manifest_sha256": FREEZE_SHA256,
        "source_catalog_sha256": CATALOG_SHA256,
        "semantic_audit_qualification_sha256": "d" * 64,
        "audit_host_code_identity_sha256": "e" * 64,
        "source_document_count": 3,
        "source_split": source_split,
        "eligible_document_count": 1,
        "excluded_document_count": 2,
        "eligible_file": "eligible.jsonl",
        "eligible_sha256": hashlib.sha256(eligible_text.encode()).hexdigest(),
        "excluded_file": "excluded.jsonl",
        "excluded_sha256": hashlib.sha256(excluded_text.encode()).hexdigest(),
        "training_ready": False,
        "training_blockers": ["evaluation task database audit has not passed"],
    }
    manifest = {
        **unsigned,
        "manifest_sha256": hashlib.sha256(_canonical(unsigned).encode()).hexdigest(),
    }
    (root / "eligible.jsonl").write_bytes(eligible_text.encode("utf-8"))
    (root / "excluded.jsonl").write_bytes(excluded_text.encode("utf-8"))
    (root / "manifest.json").write_bytes(
        f"{_canonical(manifest)}\n".encode("utf-8")
    )


def _fixture(
    tmp_path: Path, *, source_split: str = "test"
) -> tuple[Path, Path, Path, Path]:
    baseline = tmp_path / "baseline.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    source = tmp_path / "eligibility-source"
    receipt = tmp_path / "equivalence.json"
    _database(baseline, BASELINE_RUNTIME, source_split=source_split)
    _database(candidate, PARSER_RUNTIME_IDENTITY_SHA256, source_split=source_split)
    _eligibility(source, baseline, source_split=source_split)
    publish_equivalence_receipt(
        compare_admission_runtime_equivalence(baseline, candidate), receipt
    )
    return baseline, candidate, source, receipt


def test_transfer_binds_eligibility_to_current_runtime(tmp_path: Path) -> None:
    baseline, candidate, source, receipt = _fixture(tmp_path)
    destination = tmp_path / "transferred"

    result = transfer_runtime_equivalent_eligibility(
        baseline_admission_database=baseline,
        candidate_admission_database=candidate,
        source_eligibility_root=source,
        equivalence_receipt=receipt,
        output_root=destination,
    )

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert result.eligible_document_count == 1
    assert result.excluded_document_count == 2
    assert result.training_ready is False
    assert manifest["source_admission_identity_sha256"] == _admission_identity(candidate)
    assert manifest["source_eligibility_manifest_sha256"] == json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")
    )["manifest_sha256"]
    assert manifest["eligibility_transfer_policy"] == TRANSFER_POLICY
    assert (destination / "eligible.jsonl").read_bytes() == (
        source / "eligible.jsonl"
    ).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        transfer_runtime_equivalent_eligibility(
            baseline_admission_database=baseline,
            candidate_admission_database=candidate,
            source_eligibility_root=source,
            equivalence_receipt=receipt,
            output_root=destination,
        )


def test_transfer_accepts_training_admission_without_frozen_split_key(
    tmp_path: Path,
) -> None:
    baseline, candidate, source, receipt = _fixture(
        tmp_path, source_split="train"
    )
    destination = tmp_path / "transferred"

    result = transfer_runtime_equivalent_eligibility(
        baseline_admission_database=baseline,
        candidate_admission_database=candidate,
        source_eligibility_root=source,
        equivalence_receipt=receipt,
        output_root=destination,
    )

    assert result.eligible_document_count == 1
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_split"] == "train"


def test_transfer_rejects_candidate_changed_after_receipt(tmp_path: Path) -> None:
    baseline, candidate, source, receipt = _fixture(tmp_path)
    connection = sqlite3.connect(candidate)
    try:
        connection.execute(
            "UPDATE documents SET semantic_ids_zlib=? WHERE document_id='eligible'",
            (b"changed",),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="fresh comparison"):
        transfer_runtime_equivalent_eligibility(
            baseline_admission_database=baseline,
            candidate_admission_database=candidate,
            source_eligibility_root=source,
            equivalence_receipt=receipt,
            output_root=tmp_path / "transferred",
        )


def test_transfer_rejects_eligibility_payload_mismatch(tmp_path: Path) -> None:
    baseline, candidate, source, receipt = _fixture(tmp_path)
    row = json.loads((source / "eligible.jsonl").read_text(encoding="utf-8"))
    row["semantic_payload_sha256"] = "f" * 64
    eligible_text = f"{_canonical(row)}\n"
    (source / "eligible.jsonl").write_bytes(eligible_text.encode("utf-8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["eligible_sha256"] = hashlib.sha256(eligible_text.encode()).hexdigest()
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(unsigned).encode()).hexdigest()
    (source / "manifest.json").write_bytes(
        f"{_canonical(manifest)}\n".encode("utf-8")
    )

    with pytest.raises(ValueError, match="payload lineage"):
        transfer_runtime_equivalent_eligibility(
            baseline_admission_database=baseline,
            candidate_admission_database=candidate,
            source_eligibility_root=source,
            equivalence_receipt=receipt,
            output_root=tmp_path / "transferred",
        )
