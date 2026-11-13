"""Freeze a committed resumable task prefix into an immutable candidate pool."""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    _canonical_json,
    audit_task_database,
)
from .tokenizer import NativeTokenizerBinding

CANDIDATE_SNAPSHOT_SCHEMA = "bosscrafts-task-candidate-snapshot-v1"


@dataclass(frozen=True)
class CandidateSnapshotStatus:
    database_path: Path
    database_identity_sha256: str
    snapshot_policy_sha256: str
    processed_document_count: int
    last_document_id: str
    task_count: int
    rejected_count: int
    total_sequence_tokens: int
    deterministic_prefix_verified: bool
    integrity_check: str


def freeze_candidate_snapshot(
    binding: NativeTokenizerBinding,
    resumable_database: str | Path,
    admission_database: str | Path,
    destination_database: str | Path,
) -> CandidateSnapshotStatus:
    """Copy a transactionally committed document-ID prefix and seal its scope."""
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    source_path = Path(resumable_database).resolve()
    admission_path = Path(admission_database).resolve()
    destination = Path(destination_database).resolve()
    if not source_path.is_file():
        raise ValueError("resumable task database does not exist")
    if not admission_path.is_file():
        raise ValueError("admission database does not exist")
    if destination.exists():
        raise ValueError("candidate snapshot database already exists")

    source = sqlite3.connect(f"{source_path.as_uri()}?mode=ro", uri=True, timeout=30)
    source.execute("PRAGMA query_only=ON")
    try:
        metadata = dict(source.execute("SELECT key,value FROM metadata"))
        if metadata.get("schema_version") != TASK_DATABASE_SCHEMA:
            raise ValueError("resumable task database schema mismatch")
        if (
            metadata.get("binding_sha256") != binding.binding_sha256
            or metadata.get("tokenizer_sha256") != binding.tokenizer_sha256
            or metadata.get("vocabulary_size") != str(binding.vocab_size)
        ):
            raise ValueError("resumable task tokenizer lineage mismatch")
        integrity = str(source.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise ValueError(f"resumable task database integrity failed: {integrity}")
        document_rows = source.execute(
            "SELECT document_id,task_count,rejected_count FROM documents ORDER BY document_id"
        ).fetchall()
        if not document_rows:
            raise ValueError("resumable task database has no committed documents")
        processed_ids = [str(row[0]) for row in document_rows]
        last_document_id = processed_ids[-1]
        task_count = int(source.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
        rejected_count = int(source.execute("SELECT COUNT(*) FROM rejections").fetchone()[0])
        total_sequence_tokens = int(
            source.execute("SELECT COALESCE(SUM(token_count),0) FROM tasks").fetchone()[0]
        )
        if sum(int(row[1]) for row in document_rows) != task_count:
            raise ValueError("committed document task counts do not match task rows")
        if sum(int(row[2]) for row in document_rows) != rejected_count:
            raise ValueError("committed document rejection counts do not match rejection rows")
        orphan_tasks = int(
            source.execute(
                """SELECT COUNT(*) FROM tasks AS t
                   LEFT JOIN documents AS d ON d.document_id=t.document_id
                   WHERE d.document_id IS NULL"""
            ).fetchone()[0]
        )
        if orphan_tasks:
            raise ValueError("candidate prefix contains orphan task rows")

        admission = sqlite3.connect(
            f"{admission_path.as_uri()}?mode=ro", uri=True, timeout=30
        )
        admission.execute("PRAGMA query_only=ON")
        try:
            admitted_prefix = [
                str(row[0])
                for row in admission.execute(
                    """SELECT document_id FROM documents
                       WHERE status='admitted' AND document_id<=?
                       ORDER BY document_id""",
                    (last_document_id,),
                )
            ]
        finally:
            admission.close()
        if admitted_prefix != processed_ids:
            raise ValueError("committed documents are not an exact admitted document-ID prefix")

        policy = {
            "schema_version": CANDIDATE_SNAPSHOT_SCHEMA,
            "selection": "complete committed admitted document_id prefix",
            "processed_document_count": len(processed_ids),
            "last_document_id": last_document_id,
            "task_count": task_count,
            "rejected_count": rejected_count,
            "source_packing_complete_at_freeze": metadata.get("packing_complete") == "true",
        }
        policy_sha256 = hashlib.sha256(
            _canonical_json(policy).encode("utf-8")
        ).hexdigest()

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        if temporary.exists():
            temporary.unlink()
        output = sqlite3.connect(temporary)
        try:
            source.backup(output)
            additions = {
                "packing_complete": "true",
                "candidate_snapshot_schema_version": CANDIDATE_SNAPSHOT_SCHEMA,
                "candidate_snapshot_policy_sha256": policy_sha256,
                "candidate_snapshot_processed_documents": str(len(processed_ids)),
                "candidate_snapshot_last_document_id": last_document_id,
                "candidate_snapshot_source_packing_complete": str(
                    metadata.get("packing_complete") == "true"
                ).lower(),
            }
            output.executemany(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                sorted(additions.items()),
            )
            output.commit()
            snapshot_integrity = str(output.execute("PRAGMA integrity_check").fetchone()[0])
            if snapshot_integrity != "ok":
                raise ValueError(
                    f"candidate snapshot database integrity failed: {snapshot_integrity}"
                )
        except BaseException:
            output.close()
            if temporary.exists():
                temporary.unlink()
            raise
        else:
            output.close()
            temporary.replace(destination)
    finally:
        source.close()

    audit = audit_task_database(binding, destination)
    return CandidateSnapshotStatus(
        database_path=destination,
        database_identity_sha256=audit.database_identity_sha256,
        snapshot_policy_sha256=policy_sha256,
        processed_document_count=len(processed_ids),
        last_document_id=last_document_id,
        task_count=task_count,
        rejected_count=rejected_count,
        total_sequence_tokens=total_sequence_tokens,
        deterministic_prefix_verified=True,
        integrity_check="ok",
    )
