"""Proof-bound reuse of unchanged task rows across parser runtimes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from .admission_runtime_delta import (
    AdmissionRuntimeDelta,
    compare_admission_runtime_delta,
    delta_receipt_payload,
    load_delta_receipt,
)
from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from .immutable_artifacts import publish_new_files
from .resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    ResumableCurrentTaskPacking,
)
from .task_database_audit_manifest import (
    load_verified_task_database_audit_manifest,
)
from .tokenizer import NativeTokenizerBinding

SCHEMA_VERSION = "bosscrafts-task-runtime-reuse-v1"
_REUSE_POLICY = (
    "copy only baseline-audited task rows whose admission payload is exactly "
    "reusable and whose document remains candidate-eligible; update only parser "
    "runtime and semantic-spec provenance"
)
_TASK_COLUMNS = (
    "example_id",
    "document_id",
    "language",
    "family",
    "record_index",
    "split",
    "input_ids_zlib",
    "token_count",
    "loss_start_index",
    "semantic_spec_sha256",
    "parser_runtime_identity_sha256",
    "source_sha256",
)
_REJECTION_COLUMNS = (
    "rejection_id",
    "document_id",
    "language",
    "record_index",
    "family",
    "reason_code",
)


@dataclass(frozen=True)
class TaskRuntimeReuseResult:
    baseline_task_database_sha256: str
    baseline_task_database_bytes: int
    baseline_task_audit_manifest_sha256: str
    admission_delta_manifest_sha256: str
    baseline_parser_runtime_identity_sha256: str
    candidate_parser_runtime_identity_sha256: str
    candidate_eligibility_manifest_sha256: str
    target_metadata_sha256: str
    eligible_document_count: int
    reusable_eligible_document_count: int
    copied_document_count: int
    copied_example_count: int
    copied_rejection_count: int
    repack_document_count: int
    copied_document_ids_sha256: str
    target_database: Path


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ro_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("SQLite input must be a regular file")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _metadata(connection: sqlite3.Connection, *, label: str) -> dict[str, str]:
    rows = connection.execute("SELECT key,value FROM metadata ORDER BY key").fetchall()
    if any(type(key) is not str or type(value) is not str for key, value in rows):
        raise ValueError(f"{label} metadata must contain strings")
    result = dict(rows)
    if len(result) != len(rows):
        raise ValueError(f"{label} metadata contains duplicate keys")
    return result


def _admission_identity(path: Path) -> tuple[dict[str, str], str]:
    connection = _ro_connection(path)
    try:
        metadata = _metadata(connection, label="baseline admission")
        counts = connection.execute(
            """SELECT SUM(CASE WHEN status='admitted' THEN 1 ELSE 0 END),
                      SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END)
               FROM documents"""
        ).fetchone()
    finally:
        connection.close()
    if counts is None:
        raise ValueError("baseline admission counts are unavailable")
    payload = {
        "metadata": metadata,
        "admitted_count": int(counts[0] or 0),
        "rejected_count": int(counts[1] or 0),
    }
    return metadata, hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _verify_table_columns(
    connection: sqlite3.Connection,
    table: str,
    expected: tuple[str, ...],
    *,
    label: str,
) -> None:
    columns = tuple(
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
    )
    if columns != expected:
        raise ValueError(f"{label} {table} schema mismatch")


def _verify_baseline_lineage(
    binding: NativeTokenizerBinding,
    baseline_task_database: Path,
    baseline_admission_database: Path,
    delta: AdmissionRuntimeDelta,
    *,
    expected_maximum_sequence: int,
    expected_semantic_wire: str,
) -> dict[str, str]:
    admission_metadata, admission_identity = _admission_identity(
        baseline_admission_database
    )
    if (
        admission_metadata.get("parser_runtime_identity_sha256")
        != delta.baseline_parser_runtime_identity_sha256
        or admission_metadata.get("source_catalog_sha256")
        != delta.source_catalog_sha256
        or admission_metadata.get("source_freeze_manifest_sha256")
        != delta.source_freeze_manifest_sha256
    ):
        raise ValueError("baseline admission does not match the delta lineage")
    connection = _ro_connection(baseline_task_database)
    try:
        if str(connection.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise ValueError("baseline task database integrity failed")
        metadata = _metadata(connection, label="baseline task database")
        _verify_table_columns(
            connection,
            "documents",
            ("document_id", "task_count", "rejected_count"),
            label="baseline task database",
        )
        _verify_table_columns(
            connection,
            "tasks",
            _TASK_COLUMNS,
            label="baseline task database",
        )
        _verify_table_columns(
            connection,
            "rejections",
            _REJECTION_COLUMNS,
            label="baseline task database",
        )
    finally:
        connection.close()
    expected = {
        "schema_version": TASK_DATABASE_SCHEMA,
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "vocabulary_size": str(binding.vocab_size),
        "maximum_sequence_tokens": str(expected_maximum_sequence),
        "source_admission_identity_sha256": admission_identity,
        "source_catalog_sha256": delta.source_catalog_sha256,
        "review_manifest_sha256": str(admission_metadata["review_manifest_sha256"]),
        "semantic_spec_sha256": str(admission_metadata["semantic_spec_sha256"]),
        "parser_runtime_identity_sha256": (
            delta.baseline_parser_runtime_identity_sha256
        ),
        "semantic_wire_schema_version": expected_semantic_wire,
        "packing_complete": "true",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"baseline task database lineage mismatch: {key}")
    return metadata


def _validated_delta(
    baseline_admission_database: Path,
    candidate_admission_database: Path,
    delta_receipt: Path,
) -> tuple[AdmissionRuntimeDelta, dict[str, object]]:
    receipt = load_delta_receipt(delta_receipt)
    fresh = compare_admission_runtime_delta(
        baseline_admission_database,
        candidate_admission_database,
    )
    if receipt != delta_receipt_payload(fresh):
        raise ValueError("admission runtime delta receipt is stale or mismatched")
    return fresh, receipt


def _attached_uri(path: Path) -> str:
    return f"{path.resolve(strict=True).as_uri()}?mode=ro&immutable=1"


def _copy_reusable_rows(
    target_database: Path,
    baseline_task_database: Path,
    baseline_admission_database: Path,
    candidate_admission_database: Path,
    delta: AdmissionRuntimeDelta,
    eligible_ids: frozenset[str],
    task_split: str,
) -> tuple[int, int, int, int, str]:
    reusable = {
        item.document_id: item
        for item in delta.documents
        if item.classification == "reusable"
        and item.status == "admitted"
        and item.document_id in eligible_ids
    }
    baseline = _ro_connection(baseline_task_database)
    try:
        baseline_document_ids = {
            str(row[0]) for row in baseline.execute("SELECT document_id FROM documents")
        }
    finally:
        baseline.close()
    copy_ids = sorted(set(reusable).intersection(baseline_document_ids))
    if not copy_ids:
        raise ValueError("no reusable eligible task documents are available")

    connection = sqlite3.connect(
        f"{target_database.resolve(strict=True).as_uri()}?mode=rw",
        uri=True,
        timeout=300,
    )
    attached = False
    try:
        connection.execute("PRAGMA busy_timeout=300000")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "ATTACH DATABASE ? AS baseline_tasks",
            (_attached_uri(baseline_task_database),),
        )
        connection.execute(
            "ATTACH DATABASE ? AS baseline_admission",
            (_attached_uri(baseline_admission_database),),
        )
        connection.execute(
            "ATTACH DATABASE ? AS candidate_admission",
            (_attached_uri(candidate_admission_database),),
        )
        attached = True
        connection.execute(
            """CREATE TEMP TABLE copy_documents(
                   document_id TEXT PRIMARY KEY,
                   baseline_semantic_spec_sha256 TEXT NOT NULL,
                   candidate_semantic_spec_sha256 TEXT NOT NULL,
                   source_sha256 TEXT NOT NULL,
                   language TEXT NOT NULL
               ) WITHOUT ROWID"""
        )
        connection.executemany(
            "INSERT INTO copy_documents VALUES(?,?,?,?,?)",
            (
                (
                    document_id,
                    str(reusable[document_id].baseline_semantic_spec_sha256),
                    str(reusable[document_id].candidate_semantic_spec_sha256),
                    str(reusable[document_id].content_sha256),
                    str(reusable[document_id].language),
                )
                for document_id in copy_ids
            ),
        )
        connection.commit()
        empty_target = connection.execute(
            """SELECT
                   (SELECT COUNT(*) FROM main.documents),
                   (SELECT COUNT(*) FROM main.tasks),
                   (SELECT COUNT(*) FROM main.rejections),
                   (SELECT value FROM main.metadata WHERE key='packing_complete')"""
        ).fetchone()
        if empty_target != (0, 0, 0, "false"):
            raise ValueError("target task database is not an empty bootstrap")
        admission_mismatch = connection.execute(
            """SELECT c.document_id
               FROM copy_documents AS c
               LEFT JOIN baseline_admission.documents AS b USING(document_id)
               LEFT JOIN candidate_admission.documents AS n USING(document_id)
               WHERE b.document_id IS NULL OR n.document_id IS NULL
                  OR b.status!='admitted' OR n.status!='admitted'
                  OR b.content_sha256!=c.source_sha256
                  OR n.content_sha256!=c.source_sha256
                  OR b.language!=c.language OR n.language!=c.language
                  OR b.semantic_spec_sha256!=c.baseline_semantic_spec_sha256
                  OR n.semantic_spec_sha256!=c.candidate_semantic_spec_sha256
                  OR b.parser_runtime_identity_sha256!=?
                  OR n.parser_runtime_identity_sha256!=?
               LIMIT 1""",
            (
                delta.baseline_parser_runtime_identity_sha256,
                delta.candidate_parser_runtime_identity_sha256,
            ),
        ).fetchone()
        if admission_mismatch is not None:
            raise ValueError("reusable admission provenance mismatch")
        task_mismatch = connection.execute(
            """SELECT t.example_id
               FROM baseline_tasks.tasks AS t
               JOIN copy_documents AS c USING(document_id)
               WHERE t.language!=c.language OR t.split!=?
                  OR t.semantic_spec_sha256!=c.baseline_semantic_spec_sha256
                  OR t.parser_runtime_identity_sha256!=?
                  OR t.source_sha256!=c.source_sha256
               LIMIT 1""",
            (task_split, delta.baseline_parser_runtime_identity_sha256),
        ).fetchone()
        if task_mismatch is not None:
            raise ValueError("baseline reusable task provenance mismatch")
        rejection_mismatch = connection.execute(
            """SELECT r.rejection_id
               FROM baseline_tasks.rejections AS r
               JOIN copy_documents AS c USING(document_id)
               WHERE r.language!=c.language
               LIMIT 1"""
        ).fetchone()
        if rejection_mismatch is not None:
            raise ValueError("baseline reusable rejection provenance mismatch")

        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT INTO main.documents(document_id,task_count,rejected_count)
               SELECT d.document_id,d.task_count,d.rejected_count
               FROM baseline_tasks.documents AS d
               JOIN copy_documents AS c USING(document_id)
               ORDER BY d.document_id"""
        )
        connection.execute(
            """INSERT INTO main.tasks(
                   example_id,document_id,language,family,record_index,split,
                   input_ids_zlib,token_count,loss_start_index,
                   semantic_spec_sha256,parser_runtime_identity_sha256,source_sha256
               )
               SELECT t.example_id,t.document_id,t.language,t.family,t.record_index,
                      t.split,t.input_ids_zlib,t.token_count,t.loss_start_index,
                      c.candidate_semantic_spec_sha256,?,t.source_sha256
               FROM baseline_tasks.tasks AS t
               JOIN copy_documents AS c USING(document_id)
               ORDER BY t.example_id""",
            (delta.candidate_parser_runtime_identity_sha256,),
        )
        connection.execute(
            """INSERT INTO main.rejections(
                   rejection_id,document_id,language,record_index,family,reason_code
               )
               SELECT r.rejection_id,r.document_id,r.language,r.record_index,
                      r.family,r.reason_code
               FROM baseline_tasks.rejections AS r
               JOIN copy_documents AS c USING(document_id)
               ORDER BY r.rejection_id"""
        )
        counts = connection.execute(
            """SELECT
                   (SELECT COUNT(*) FROM main.documents),
                   (SELECT COUNT(*) FROM main.tasks),
                   (SELECT COUNT(*) FROM main.rejections),
                   (SELECT COALESCE(SUM(task_count),0) FROM main.documents),
                   (SELECT COALESCE(SUM(rejected_count),0) FROM main.documents)"""
        ).fetchone()
        if (
            counts is None
            or int(counts[0]) != len(copy_ids)
            or int(counts[1]) != int(counts[3])
            or int(counts[2]) != int(counts[4])
        ):
            raise ValueError("copied task ledger count mismatch")
        connection.commit()
        copied_ids_sha256 = hashlib.sha256(
            "".join(f"{document_id}\n" for document_id in copy_ids).encode("utf-8")
        ).hexdigest()
        result = (
            len(copy_ids),
            int(counts[1]),
            int(counts[2]),
            len(reusable),
            copied_ids_sha256,
        )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return result
    except Exception:
        connection.rollback()
        raise
    finally:
        if attached:
            for name in ("candidate_admission", "baseline_admission", "baseline_tasks"):
                try:
                    connection.execute(f"DETACH DATABASE {name}")
                except sqlite3.Error:
                    pass
        connection.close()


def task_runtime_reuse_receipt_payload(
    result: TaskRuntimeReuseResult,
) -> dict[str, object]:
    raw = asdict(result)
    raw["target_database"] = str(result.target_database)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        **raw,
        "reuse_policy": _REUSE_POLICY,
        "training_ready": False,
    }
    manifest_sha256 = hashlib.sha256(
        _canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    return {**unsigned, "manifest_sha256": manifest_sha256}


def load_task_runtime_reuse_receipt(path: Path) -> dict[str, object]:
    try:
        raw = path.resolve(strict=True).read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("task runtime reuse receipt is invalid") from error
    if type(payload) is not dict or raw != f"{_canonical_json(payload)}\n":
        raise ValueError("task runtime reuse receipt is not canonical")
    unsigned = dict(payload)
    claimed = unsigned.pop("manifest_sha256", None)
    required = {
        "schema_version",
        "baseline_task_database_sha256",
        "baseline_task_database_bytes",
        "baseline_task_audit_manifest_sha256",
        "admission_delta_manifest_sha256",
        "baseline_parser_runtime_identity_sha256",
        "candidate_parser_runtime_identity_sha256",
        "candidate_eligibility_manifest_sha256",
        "target_metadata_sha256",
        "eligible_document_count",
        "reusable_eligible_document_count",
        "copied_document_count",
        "copied_example_count",
        "copied_rejection_count",
        "repack_document_count",
        "copied_document_ids_sha256",
        "target_database",
        "reuse_policy",
        "training_ready",
    }
    hash_fields = {
        "baseline_task_database_sha256",
        "baseline_task_audit_manifest_sha256",
        "admission_delta_manifest_sha256",
        "baseline_parser_runtime_identity_sha256",
        "candidate_parser_runtime_identity_sha256",
        "candidate_eligibility_manifest_sha256",
        "target_metadata_sha256",
        "copied_document_ids_sha256",
    }
    count_fields = {
        "baseline_task_database_bytes",
        "eligible_document_count",
        "reusable_eligible_document_count",
        "copied_document_count",
        "copied_example_count",
        "copied_rejection_count",
        "repack_document_count",
    }
    if (
        set(unsigned) != required
        or payload.get("schema_version") != SCHEMA_VERSION
        or not _is_sha256(claimed)
        or any(not _is_sha256(payload.get(key)) for key in hash_fields)
        or any(
            type(payload.get(key)) is not int or int(payload[key]) < 0
            for key in count_fields
        )
        or payload.get("reuse_policy") != _REUSE_POLICY
        or type(payload.get("target_database")) is not str
        or not Path(str(payload["target_database"])).is_absolute()
        or claimed
        != hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        or payload.get("training_ready") is not False
        or int(payload["copied_document_count"])
        > int(payload["reusable_eligible_document_count"])
        or int(payload["reusable_eligible_document_count"])
        > int(payload["eligible_document_count"])
        or int(payload["repack_document_count"])
        != int(payload["eligible_document_count"])
        - int(payload["copied_document_count"])
    ):
        raise ValueError("task runtime reuse receipt lineage is invalid")
    return payload


def migrate_reusable_tasks(
    binding: NativeTokenizerBinding,
    *,
    baseline_task_database: Path,
    baseline_task_audit_manifest: Path,
    baseline_admission_database: Path,
    candidate_admission_database: Path,
    admission_delta_receipt: Path,
    source_catalog: Path,
    candidate_eligibility_root: Path,
    candidate_eligibility_manifest_sha256: str,
    target_database: Path,
    output_receipt: Path,
    max_sequence: int = 2048,
    semantic_wire: str = "compact_v2",
    defer_cell_index: bool = True,
    forced_split: str | None = None,
) -> TaskRuntimeReuseResult:
    """Bootstrap a current task DB with only proof-reusable baseline rows."""

    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    baseline_task = baseline_task_database.resolve(strict=True)
    baseline_admission = baseline_admission_database.resolve(strict=True)
    candidate_admission = candidate_admission_database.resolve(strict=True)
    target = target_database.resolve()
    receipt_path = output_receipt.resolve()
    if target.exists() or receipt_path.exists():
        raise ValueError("target task database and receipt must be new paths")
    if target in {baseline_task, baseline_admission, candidate_admission}:
        raise ValueError("target task database must be a distinct path")

    verified_audit = load_verified_task_database_audit_manifest(
        binding,
        baseline_task,
        baseline_task_audit_manifest,
    )
    delta, delta_receipt = _validated_delta(
        baseline_admission,
        candidate_admission,
        admission_delta_receipt,
    )
    expected_wire = (
        "bosscrafts-compact-semantic-wire-v2"
        if semantic_wire == "compact_v2"
        else "bosscrafts-full-semantic-wire-v3"
    )
    _verify_baseline_lineage(
        binding,
        baseline_task,
        baseline_admission,
        delta,
        expected_maximum_sequence=max_sequence,
        expected_semantic_wire=expected_wire,
    )

    packing = ResumableCurrentTaskPacking(
        binding,
        candidate_admission,
        source_catalog,
        target,
        max_sequence=max_sequence,
        semantic_wire=semantic_wire,
        eligibility_root=candidate_eligibility_root,
        expected_eligibility_manifest_sha256=(
            candidate_eligibility_manifest_sha256
        ),
        defer_cell_index=defer_cell_index,
        forced_split=forced_split,
    )
    eligible_ids = packing.eligible_document_ids
    if eligible_ids is None:
        raise RuntimeError("candidate task bootstrap did not bind eligibility")
    copied_documents, copied_examples, copied_rejections, reusable_eligible, copied_sha = (
        _copy_reusable_rows(
            target,
            baseline_task,
            baseline_admission,
            candidate_admission,
            delta,
            eligible_ids,
            forced_split or "train",
        )
    )
    target_connection = _ro_connection(target)
    try:
        target_metadata = _metadata(target_connection, label="target task database")
    finally:
        target_connection.close()
    target_metadata_sha256 = hashlib.sha256(
        _canonical_json(target_metadata).encode("utf-8")
    ).hexdigest()
    result = TaskRuntimeReuseResult(
        baseline_task_database_sha256=verified_audit.database_file_sha256,
        baseline_task_database_bytes=verified_audit.database_file_size_bytes,
        baseline_task_audit_manifest_sha256=verified_audit.manifest_sha256,
        admission_delta_manifest_sha256=str(delta_receipt["manifest_sha256"]),
        baseline_parser_runtime_identity_sha256=(
            delta.baseline_parser_runtime_identity_sha256
        ),
        candidate_parser_runtime_identity_sha256=PARSER_RUNTIME_IDENTITY_SHA256,
        candidate_eligibility_manifest_sha256=(
            candidate_eligibility_manifest_sha256
        ),
        target_metadata_sha256=target_metadata_sha256,
        eligible_document_count=len(eligible_ids),
        reusable_eligible_document_count=reusable_eligible,
        copied_document_count=copied_documents,
        copied_example_count=copied_examples,
        copied_rejection_count=copied_rejections,
        repack_document_count=len(eligible_ids) - copied_documents,
        copied_document_ids_sha256=copied_sha,
        target_database=target,
    )
    payload = task_runtime_reuse_receipt_payload(result)
    publish_new_files(
        ((receipt_path, f"{_canonical_json(payload)}\n".encode("utf-8")),),
        artifact_label="task runtime reuse receipt",
    )
    return result
