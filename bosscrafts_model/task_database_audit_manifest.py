"""Immutable file-bound receipts for completed logical task database audits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from .immutable_artifacts import publish_new_files
from .resumable_task_packing import TaskDatabaseAuditResult, audit_task_database
from .tokenizer import NativeTokenizerBinding

TASK_DATABASE_AUDIT_MANIFEST_SCHEMA = "bosscrafts-task-database-audit-v2"


@dataclass(frozen=True)
class VerifiedTaskDatabaseAudit:
    audit: TaskDatabaseAuditResult
    database_file_sha256: str
    database_file_size_bytes: int
    manifest_sha256: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_quiescent(path: Path) -> None:
    for suffix in ("-wal", "-journal"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.is_file() and sidecar.stat().st_size:
            raise ValueError("task database has an uncommitted or live SQLite sidecar")


def _stable_file_identity(path: Path) -> tuple[str, int]:
    _assert_quiescent(path)
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    _assert_quiescent(path)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise ValueError("task database changed while its file identity was computed")
    return digest, after.st_size


def materialize_task_database_audit_manifest(
    binding: NativeTokenizerBinding,
    database_path: str | Path,
    output_path: str | Path,
) -> VerifiedTaskDatabaseAudit:
    """Run the full logical audit once and bind its receipt to exact file bytes."""
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    database = Path(database_path).resolve()
    output = Path(output_path).resolve()
    if not database.is_file():
        raise ValueError("task database does not exist")
    if output.exists():
        raise ValueError("task database audit manifest output already exists")
    _assert_quiescent(database)
    before = database.stat()
    audit = audit_task_database(binding, database)
    database_file_sha256, database_file_size_bytes = _stable_file_identity(database)
    after = database.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise ValueError("task database changed during its logical audit")
    return _publish_task_database_audit_manifest(
        binding,
        audit,
        database,
        output,
        database_file_sha256=database_file_sha256,
        database_file_size_bytes=database_file_size_bytes,
    )


def _publish_task_database_audit_manifest(
    binding: NativeTokenizerBinding,
    audit: TaskDatabaseAuditResult,
    database: Path,
    output: Path,
    *,
    database_file_sha256: str,
    database_file_size_bytes: int,
) -> VerifiedTaskDatabaseAudit:
    unsigned = {
        **asdict(audit),
        "database_path": str(database),
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "vocabulary_size": binding.vocab_size,
        "database_file_sha256": database_file_sha256,
        "database_file_size_bytes": database_file_size_bytes,
        "schema_version": TASK_DATABASE_AUDIT_MANIFEST_SCHEMA,
    }
    manifest_sha256 = hashlib.sha256(
        _canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    payload = {**unsigned, "manifest_sha256": manifest_sha256}
    publish_new_files(
        ((output, f"{_canonical_json(payload)}\n".encode("utf-8")),),
        artifact_label="task database audit manifest",
    )
    return VerifiedTaskDatabaseAudit(
        audit=audit,
        database_file_sha256=database_file_sha256,
        database_file_size_bytes=database_file_size_bytes,
        manifest_sha256=manifest_sha256,
    )


def publish_task_database_audit_manifest_from_result(
    binding: NativeTokenizerBinding,
    audit: TaskDatabaseAuditResult,
    database_path: str | Path,
    output_path: str | Path,
) -> VerifiedTaskDatabaseAudit:
    """Bind an audit just produced for the same immutable SQLite inode.

    This is for builders that audit a private temporary database before
    hard-linking that exact inode to its immutable destination.
    """
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if type(audit) is not TaskDatabaseAuditResult or audit.audit_passed is not True:
        raise TypeError("audit must be a passed TaskDatabaseAuditResult")
    database = Path(database_path).resolve()
    output = Path(output_path).resolve()
    if not database.is_file() or not audit.database_path.is_file():
        raise ValueError("audited task database does not exist")
    try:
        same_database = audit.database_path.samefile(database)
    except OSError as error:
        raise ValueError("audited task database identity cannot be resolved") from error
    if not same_database:
        raise ValueError("audit result belongs to another task database inode")
    if output.exists():
        raise ValueError("task database audit manifest output already exists")
    database_file_sha256, database_file_size_bytes = _stable_file_identity(database)
    return _publish_task_database_audit_manifest(
        binding,
        audit,
        database,
        output,
        database_file_sha256=database_file_sha256,
        database_file_size_bytes=database_file_size_bytes,
    )


def load_verified_task_database_audit_manifest(
    binding: NativeTokenizerBinding,
    database_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_database_identity_sha256: str | None = None,
) -> VerifiedTaskDatabaseAudit:
    """Verify a prior logical audit against the exact current database bytes."""
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if expected_database_identity_sha256 is not None and not _is_sha256(
        expected_database_identity_sha256
    ):
        raise ValueError("expected task database identity is invalid")
    database = Path(database_path).resolve()
    manifest = Path(manifest_path).resolve()
    if not database.is_file() or not manifest.is_file():
        raise ValueError("task database or audit manifest does not exist")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("task database audit manifest is invalid") from error
    if type(payload) is not dict:
        raise ValueError("task database audit manifest is invalid")
    unsigned = dict(payload)
    claimed_manifest_sha256 = unsigned.pop("manifest_sha256", None)
    required = {
        "audit_passed",
        "binding_sha256",
        "content_sha256",
        "database_file_sha256",
        "database_file_size_bytes",
        "database_identity_sha256",
        "database_path",
        "document_count",
        "example_count",
        "family_counts",
        "language_counts",
        "maximum_sequence_tokens",
        "schema_version",
        "split_counts",
        "tokenizer_sha256",
        "total_sequence_tokens",
        "vocabulary_size",
    }
    if (
        set(unsigned) != required
        or claimed_manifest_sha256
        != hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        or unsigned.get("schema_version") != TASK_DATABASE_AUDIT_MANIFEST_SCHEMA
        or unsigned.get("database_path") != str(database)
        or unsigned.get("binding_sha256") != binding.binding_sha256
        or unsigned.get("tokenizer_sha256") != binding.tokenizer_sha256
        or unsigned.get("vocabulary_size") != binding.vocab_size
        or unsigned.get("audit_passed") is not True
        or not _is_sha256(unsigned.get("content_sha256"))
        or not _is_sha256(unsigned.get("database_identity_sha256"))
        or not _is_sha256(unsigned.get("database_file_sha256"))
        or not _is_sha256(claimed_manifest_sha256)
    ):
        raise ValueError("task database audit manifest lineage is invalid")
    if (
        expected_database_identity_sha256 is not None
        and unsigned["database_identity_sha256"]
        != expected_database_identity_sha256
    ):
        raise ValueError("task database audit manifest identity mismatch")
    count_fields = (
        "document_count",
        "example_count",
        "maximum_sequence_tokens",
        "total_sequence_tokens",
        "database_file_size_bytes",
    )
    if any(type(unsigned[key]) is not int or unsigned[key] < 0 for key in count_fields):
        raise ValueError("task database audit manifest counts are invalid")
    count_maps: dict[str, dict[str, int]] = {}
    for key in ("family_counts", "language_counts", "split_counts"):
        raw = unsigned[key]
        if (
            type(raw) is not dict
            or not raw
            or any(
                type(name) is not str
                or not name
                or type(count) is not int
                or count < 1
                for name, count in raw.items()
            )
        ):
            raise ValueError("task database audit manifest counts are invalid")
        count_maps[key] = cast(dict[str, int], raw)
    database_file_sha256, database_file_size_bytes = _stable_file_identity(database)
    if (
        database_file_sha256 != unsigned["database_file_sha256"]
        or database_file_size_bytes != unsigned["database_file_size_bytes"]
    ):
        raise ValueError("task database bytes do not match the logical audit manifest")
    audit = TaskDatabaseAuditResult(
        database_path=database,
        content_sha256=str(unsigned["content_sha256"]),
        database_identity_sha256=str(unsigned["database_identity_sha256"]),
        example_count=int(unsigned["example_count"]),
        total_sequence_tokens=int(unsigned["total_sequence_tokens"]),
        maximum_sequence_tokens=int(unsigned["maximum_sequence_tokens"]),
        split_counts=count_maps["split_counts"],
        language_counts=count_maps["language_counts"],
        family_counts=count_maps["family_counts"],
        document_count=int(unsigned["document_count"]),
        audit_passed=True,
    )
    return VerifiedTaskDatabaseAudit(
        audit=audit,
        database_file_sha256=database_file_sha256,
        database_file_size_bytes=database_file_size_bytes,
        manifest_sha256=str(claimed_manifest_sha256),
    )
