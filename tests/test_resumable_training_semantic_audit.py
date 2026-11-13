from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

import bosscrafts_model.resumable_training_semantic_audit as audit_module
from bosscrafts_model.bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from bosscrafts_model.resumable_training_semantic_audit import (
    ResumableTrainingSemanticAudit,
)
from bosscrafts_model.training_eligibility import AuditedTrainingEligibilityBuilder


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _PassingWitnessClient:
    def inspect(self, source: str, *, language: str, path: str, dialect, queries):
        return {
            "source_sha256": _sha256(source.encode("utf-8")),
            "witnesses": [
                {
                    "query_id": query["query_id"],
                    "span": query["span"],
                    "ast_relation": query["ast_relation"],
                    "claimed_scope": query["scope"],
                    "derived_scope": query["scope"],
                    "exact_named_node_types": ["command_name"],
                    "boundary_start_named_node_types": [],
                    "boundary_end_named_node_types": [],
                    "relation_span_match": True,
                    "scope_match": True,
                }
                for query in queries
            ],
        }


class _FailingWitnessClient:
    def inspect(self, source: str, *, language: str, path: str, dialect, queries):
        raise ValueError("fixture sidecar failure")


def _fixture(
    root: Path, *, complete: bool = True, source_split: str = "train"
) -> tuple[Path, Path, str, str]:
    source = "set -e\n"
    source_sha = _sha256(source.encode("utf-8"))
    catalog = root / "catalog.sqlite"
    connection = sqlite3.connect(catalog)
    try:
        connection.executescript(
            """
            CREATE TABLE documents(
                document_id TEXT PRIMARY KEY,
                canonical_language TEXT NOT NULL,
                content TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                dialect TEXT
            );
            """
        )
        connection.executemany(
            "INSERT INTO documents VALUES(?,?,?,?,?,?)",
            (
                ("admitted-doc", "bash", source, "admitted.sh", source_sha, None),
                ("rejected-doc", "bash", "x", "rejected.sh", _sha256(b"x"), None),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    catalog_sha = _sha256(catalog.read_bytes())

    freeze = root / "freeze"
    freeze.mkdir()
    freeze_rows = [
        {
            "schema_version": "bosscrafts-training-source-document-v1",
            "document_id": document_id,
            "split": source_split,
        }
        for document_id in ("admitted-doc", "rejected-doc")
    ]
    freeze_text = "".join(f"{_canonical(row)}\n" for row in freeze_rows)
    freeze_manifest = {
        "schema_version": "bosscrafts-training-source-freeze-v1",
        "source_freeze_complete": True,
        "training_ready": False,
        "source_catalog_sha256": catalog_sha,
        "documents_file": "documents.jsonl",
        "documents_sha256": _sha256(freeze_text.encode("utf-8")),
        "source_document_count": 2,
        "training_document_count": 2 if source_split == "train" else 0,
    }
    freeze_sha = _sha256(_canonical(freeze_manifest).encode("utf-8"))
    freeze_manifest["manifest_sha256"] = freeze_sha
    (freeze / "documents.jsonl").write_bytes(freeze_text.encode("utf-8"))
    (freeze / "manifest.json").write_bytes(
        f"{_canonical(freeze_manifest)}\n".encode("utf-8")
    )

    record = {
        "ast_relation": "command.name",
        "confidence_band": "medium",
        "defines": None,
        "event": "CALL",
        "ir": {
            "children": [],
            "kind": "call",
            "metadata": {},
            "name": "set",
            "value": None,
        },
        "metadata": {
            "knowledge_state": "provisional",
            "uncertainty_signals": ["missing definition"],
            "uyr_rounds": 1,
        },
        "recent_history": [],
        "scope": "module",
        "span": [1, 0, 1, 3],
        "surface": "set",
        "symbol_id": None,
        "uncertainty_band": "medium",
    }
    canonical_jsonl = f"{_canonical(record)}\n"
    admission = root / "admission.sqlite"
    connection = sqlite3.connect(admission)
    try:
        connection.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents(
                document_id TEXT PRIMARY KEY,
                content_sha256 TEXT NOT NULL,
                language TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                status TEXT NOT NULL,
                reason_code TEXT,
                detail TEXT,
                semantic_spec_sha256 TEXT,
                parser_runtime_identity_sha256 TEXT,
                canonical_sha256 TEXT,
                canonical_jsonl_zlib BLOB,
                semantic_ids_zlib BLOB,
                semantic_record_count INTEGER NOT NULL,
                semantic_token_count INTEGER NOT NULL,
                raw_token_count INTEGER NOT NULL
            );
            """
        )
        metadata = {
            "schema_version": "bosscrafts-current-corpus-resumable-v4",
            "catalog_exhausted": str(complete).lower(),
            "expected_source_document_count": "2",
            "source_catalog_sha256": catalog_sha,
            "source_freeze_manifest_sha256": freeze_sha,
            "semantic_audit_qualification_sha256": "e" * 64,
            "host_code_identity_sha256": "f" * 64,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        }
        if source_split != "train":
            metadata.update(
                {
                    "frozen_source_split": source_split,
                    "frozen_source_split_identity_sha256": "d" * 64,
                }
            )
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)", sorted(metadata.items())
        )
        connection.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "admitted-doc",
                source_sha,
                "bash",
                "admitted.sh",
                "admitted",
                None,
                None,
                "semantic-spec",
                PARSER_RUNTIME_IDENTITY_SHA256,
                _sha256(canonical_jsonl.encode("utf-8")),
                zlib.compress(canonical_jsonl.encode("utf-8")),
                zlib.compress(b"ids"),
                1,
                3,
                3,
            ),
        )
        connection.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "rejected-doc",
                _sha256(b"x"),
                "bash",
                "rejected.sh",
                "rejected",
                "semantic_parse_failure",
                "fixture rejection",
                None,
                None,
                None,
                None,
                None,
                0,
                0,
                0,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return admission, catalog, catalog_sha, freeze_sha


def _audit(
    root: Path,
    admission: Path,
    catalog: Path,
    catalog_sha: str,
    freeze_sha: str,
    *,
    source_split: str = "train",
):
    return ResumableTrainingSemanticAudit(
        admission,
        catalog,
        root / "freeze",
        root / "audit.sqlite",
        expected_source_catalog_sha256=catalog_sha,
        expected_source_freeze_manifest_sha256=freeze_sha,
        expected_qualification_sha256="e" * 64,
        expected_admission_host_code_identity_sha256="f" * 64,
        audit_host_code_identity_sha256="a" * 64,
        source_split=source_split,
        witness_client=_PassingWitnessClient(),
    )


def _failing_audit(
    root: Path, admission: Path, catalog: Path, catalog_sha: str, freeze_sha: str
):
    return ResumableTrainingSemanticAudit(
        admission,
        catalog,
        root / "freeze",
        root / "audit.sqlite",
        expected_source_catalog_sha256=catalog_sha,
        expected_source_freeze_manifest_sha256=freeze_sha,
        expected_qualification_sha256="e" * 64,
        expected_admission_host_code_identity_sha256="f" * 64,
        audit_host_code_identity_sha256="a" * 64,
        witness_client=_FailingWitnessClient(),
    )


def _real_audit(
    root: Path, admission: Path, catalog: Path, catalog_sha: str, freeze_sha: str
):
    return ResumableTrainingSemanticAudit(
        admission,
        catalog,
        root / "freeze",
        root / "audit.sqlite",
        expected_source_catalog_sha256=catalog_sha,
        expected_source_freeze_manifest_sha256=freeze_sha,
        expected_qualification_sha256="e" * 64,
        expected_admission_host_code_identity_sha256="f" * 64,
        audit_host_code_identity_sha256="a" * 64,
    )


def test_audits_every_admitted_record_and_preserves_source_rejection(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    audit = _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha)
    status = audit.process()

    assert status.processed_total == 2
    assert status.admitted_document_count == 1
    assert status.source_rejected_document_count == 1
    assert status.semantic_record_count == 1
    assert status.failed_record_count == 0
    assert status.audit_complete is True
    assert status.training_ready is False
    resumed = _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha).process()
    assert resumed.processed_this_run == 0
    assert resumed == status.__class__(
        database_path=status.database_path,
        processed_this_run=0,
        processed_total=2,
        admitted_document_count=1,
        source_rejected_document_count=1,
        semantic_record_count=1,
        failed_record_count=0,
        finding_count=0,
        audit_complete=True,
        training_ready=False,
    )


def test_audits_exact_frozen_validation_split_with_split_lineage(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(
        tmp_path, source_split="validation"
    )
    status = _audit(
        tmp_path,
        admission,
        catalog,
        catalog_sha,
        freeze_sha,
        source_split="validation",
    ).process()
    assert status.audit_complete is True
    assert status.processed_total == 2
    connection = sqlite3.connect(tmp_path / "audit.sqlite")
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    finally:
        connection.close()
    assert metadata["source_split"] == "validation"
    assert metadata["frozen_source_split_identity_sha256"] == "d" * 64


def test_refuses_incomplete_source_admission(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path, complete=False)
    with pytest.raises(ValueError, match="source admission is not complete"):
        _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha)


def test_fails_closed_on_tampered_canonical_payload(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    connection = sqlite3.connect(admission)
    try:
        connection.execute(
            "UPDATE documents SET canonical_jsonl_zlib=? WHERE document_id='admitted-doc'",
            (zlib.compress(b"{}\n"),),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="canonical JSONL SHA-256 mismatch"):
        _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha).process()


@pytest.mark.parametrize(
    "payload_mutator",
    (
        lambda payload: payload[:-1],
        lambda payload: payload + b"trailing-data",
        lambda payload: payload + payload,
    ),
    ids=("truncated", "trailing-data", "concatenated"),
)
def test_fails_closed_on_non_exact_canonical_zlib_stream(
    tmp_path: Path,
    payload_mutator,
):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    connection = sqlite3.connect(admission)
    try:
        payload = connection.execute(
            "SELECT canonical_jsonl_zlib FROM documents WHERE document_id='admitted-doc'"
        ).fetchone()[0]
        connection.execute(
            "UPDATE documents SET canonical_jsonl_zlib=? WHERE document_id='admitted-doc'",
            (payload_mutator(payload),),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="canonical JSONL payload is invalid"):
        _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha).process()


def test_fails_closed_before_high_ratio_canonical_payload_exceeds_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    oversized = b"x" * 65
    monkeypatch.setattr(audit_module, "MAX_CANONICAL_OUTPUT_BYTES", 64)
    connection = sqlite3.connect(admission)
    try:
        connection.execute(
            """UPDATE documents
               SET canonical_sha256=?, canonical_jsonl_zlib=?
               WHERE document_id='admitted-doc'""",
            (_sha256(oversized), zlib.compress(oversized, 9)),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="canonical JSONL payload is invalid"):
        _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha).process()


def test_zero_record_admission_is_an_explicit_coverage_finding(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    connection = sqlite3.connect(admission)
    try:
        connection.execute(
            """UPDATE documents
               SET canonical_sha256=?, canonical_jsonl_zlib=?,
                   semantic_record_count=0, semantic_token_count=3
               WHERE document_id='admitted-doc'""",
            (_sha256(b""), zlib.compress(b"")),
        )
        connection.commit()
    finally:
        connection.close()
    status = _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha).process()
    assert status.semantic_record_count == 0
    assert status.failed_record_count == 0
    assert status.finding_count == 1
    connection = sqlite3.connect(tmp_path / "audit.sqlite")
    try:
        finding = connection.execute(
            "SELECT record_index,gate,codes_json FROM findings"
        ).fetchone()
    finally:
        connection.close()
    assert finding == (-1, "coverage", '["no_semantic_records"]')


def test_full_training_audit_script_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_v10_training_semantic_audit.py"),
            "--help",
        ],
        cwd=project_root,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--max-documents" in completed.stdout
    assert "--status-only" in completed.stdout


def test_audit_and_eligibility_scripts_share_frozen_code_identity(tmp_path: Path):
    from scripts import materialize_v10_training_eligibility as eligibility_script
    from scripts import run_v10_training_semantic_audit as audit_script

    inventory = audit_script._audit_host_code_inventory()
    identity = audit_script._audit_host_code_identity(inventory)
    assert identity == eligibility_script.audit_host_code_identity()
    manifest = tmp_path / "host-code-manifest.json"
    first_sha = audit_script._freeze_code_manifest(manifest, inventory, identity)
    assert first_sha == audit_script._freeze_code_manifest(
        manifest, inventory, identity
    )
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed across resume"):
        audit_script._freeze_code_manifest(manifest, inventory, identity)


def test_resume_detects_source_payload_changed_after_audit(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha).process()
    connection = sqlite3.connect(admission)
    try:
        connection.execute(
            "UPDATE documents SET canonical_jsonl_zlib=? WHERE document_id='admitted-doc'",
            (zlib.compress(b"{}\n"),),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="source payload changed after audit"):
        _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha).process()


def test_raw_ast_sidecar_failure_is_a_durable_blocking_finding(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    status = _failing_audit(
        tmp_path, admission, catalog, catalog_sha, freeze_sha
    ).process()
    assert status.audit_complete is True
    assert status.failed_record_count == 1
    assert status.finding_count == 1
    connection = sqlite3.connect(tmp_path / "audit.sqlite")
    try:
        finding = connection.execute(
            "SELECT record_index,gate,codes_json,evidence_json FROM findings"
        ).fetchone()
    finally:
        connection.close()
    assert finding[:3] == (-1, "raw_ast", '["raw_ast_witness_failure"]')
    assert "fixture sidecar failure" in finding[3]


def test_refuses_same_count_with_nonfrozen_document_id(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    connection = sqlite3.connect(admission)
    try:
        connection.execute(
            "UPDATE documents SET document_id='rogue-doc' WHERE document_id='rejected-doc'"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="source admission document count is incomplete"):
        _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha)


def test_materializes_only_zero_finding_audited_documents(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    audit = _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha)
    audit.process()
    result = AuditedTrainingEligibilityBuilder(
        expected_audit_host_code_identity_sha256="a" * 64
    ).materialize(
        tmp_path / "audit.sqlite",
        admission,
        tmp_path / "eligibility",
    )
    assert result.eligible_document_count == 1
    assert result.excluded_document_count == 1
    assert result.training_ready is False
    eligible = [
        json.loads(line)
        for line in (tmp_path / "eligibility" / "eligible.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    excluded = [
        json.loads(line)
        for line in (tmp_path / "eligibility" / "excluded.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["document_id"] for row in eligible] == ["admitted-doc"]
    assert excluded == [
        {
            "document_id": "rejected-doc",
            "finding_count": 0,
            "reason": "source_rejected",
            "schema_version": "bosscrafts-training-eligibility-exclusion-v1",
            "semantic_record_count": 0,
        }
    ]


def test_eligibility_refuses_admission_changed_after_audit(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    _audit(tmp_path, admission, catalog, catalog_sha, freeze_sha).process()
    connection = sqlite3.connect(admission)
    try:
        connection.execute(
            "UPDATE documents SET detail='changed' WHERE document_id='rejected-doc'"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        ValueError, match="training admission changed after semantic audit"
    ):
        AuditedTrainingEligibilityBuilder(
            expected_audit_host_code_identity_sha256="a" * 64
        ).materialize(
            tmp_path / "audit.sqlite",
            admission,
            tmp_path / "eligibility",
        )


def test_real_raw_ast_sidecar_audits_fixture_record(tmp_path: Path):
    admission, catalog, catalog_sha, freeze_sha = _fixture(tmp_path)
    status = _real_audit(
        tmp_path, admission, catalog, catalog_sha, freeze_sha
    ).process()
    assert status.audit_complete is True
    assert status.semantic_record_count == 1
    assert status.failed_record_count == 0
    assert status.finding_count == 0
