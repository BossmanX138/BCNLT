"""Proof-bound eligibility transfer across exactly equivalent parser runtimes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .admission_runtime_equivalence import (
    ADMISSION_SCHEMA_VERSION,
    compare_admission_runtime_equivalence,
    equivalence_receipt_payload,
    load_equivalence_receipt,
)
from .immutable_artifacts import publish_new_files
from .resumable_training_semantic_audit import ResumableTrainingSemanticAudit
from .training_eligibility import (
    ELIGIBILITY_MANIFEST_SCHEMA,
    ELIGIBLE_DOCUMENT_SCHEMA,
    EXCLUDED_DOCUMENT_SCHEMA,
)

TRANSFER_POLICY = (
    "eligibility membership and audit receipts preserved only after a fresh exact logical "
    "comparison proves all admission metadata and document payloads identical except "
    "host-code and parser-runtime provenance"
)
_SHA256_KEYS = {
    "audit_receipt_sha256",
    "canonical_payload_sha256",
    "canonical_sha256",
    "semantic_payload_sha256",
}


@dataclass(frozen=True, slots=True)
class EligibilityRuntimeTransferResult:
    output_root: Path
    eligible_document_count: int
    excluded_document_count: int
    manifest_sha256: str
    candidate_admission_identity_sha256: str
    equivalence_manifest_sha256: str
    training_ready: bool


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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_canonical_manifest(path: Path) -> dict[str, object]:
    try:
        raw = path.resolve(strict=True).read_text(encoding="utf-8")
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source eligibility manifest is invalid") from error
    if type(manifest) is not dict or raw != f"{_canonical(manifest)}\n":
        raise ValueError("source eligibility manifest is not canonical")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != ELIGIBILITY_MANIFEST_SCHEMA
        or not _is_sha256(claimed)
        or claimed != _sha256(_canonical(unsigned).encode("utf-8"))
        or manifest.get("eligibility_complete") is not True
        or manifest.get("training_ready") is not False
        or manifest.get("eligible_file") != "eligible.jsonl"
        or manifest.get("excluded_file") != "excluded.jsonl"
    ):
        raise ValueError("source eligibility manifest lineage is invalid")
    return manifest


def _read_inventory(
    path: Path,
    *,
    expected_sha256: object,
    expected_count: object,
    expected_schema: str,
) -> tuple[bytes, list[dict[str, object]]]:
    try:
        content = path.resolve(strict=True).read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("source eligibility inventory is invalid") from error
    if not _is_sha256(expected_sha256) or _sha256(content) != expected_sha256:
        raise ValueError("source eligibility inventory SHA-256 mismatch")
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("source eligibility inventory count is invalid")
    rows: list[dict[str, object]] = []
    canonical_parts: list[str] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("source eligibility inventory row is invalid") from error
        if type(row) is not dict or row.get("schema_version") != expected_schema:
            raise ValueError("source eligibility inventory row schema is invalid")
        canonical = _canonical(row)
        if line != canonical:
            raise ValueError("source eligibility inventory row is not canonical")
        rows.append(row)
        canonical_parts.append(f"{canonical}\n")
    if text != "".join(canonical_parts) or len(rows) != expected_count:
        raise ValueError("source eligibility inventory count or framing mismatch")
    return content, rows


def _ro_connect(path: Path) -> sqlite3.Connection:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("admission database must be a regular file")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _load_admission(
    path: Path,
) -> tuple[dict[str, str], str, dict[str, tuple[Any, ...]]]:
    connection = _ro_connect(path)
    try:
        if str(connection.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise ValueError("source admission database integrity failed")
        metadata = dict(connection.execute("SELECT key,value FROM metadata ORDER BY key"))
        if metadata.get("schema_version") != ADMISSION_SCHEMA_VERSION:
            raise ValueError("source admission database schema mismatch")
        identity = ResumableTrainingSemanticAudit._admission_identity(
            connection, metadata
        )
        rows = {
            str(row[0]): row[1:]
            for row in connection.execute(
                """SELECT document_id,status,language,canonical_sha256,
                          canonical_jsonl_zlib,semantic_ids_zlib,
                          semantic_record_count
                   FROM documents ORDER BY document_id"""
            )
        }
    except sqlite3.Error as error:
        raise ValueError("source admission database is invalid") from error
    finally:
        connection.close()
    return metadata, identity, rows


def _validate_inventory_against_admission(
    manifest: dict[str, object],
    eligible: list[dict[str, object]],
    excluded: list[dict[str, object]],
    metadata: dict[str, str],
    admission_identity: str,
    admission_rows: dict[str, tuple[Any, ...]],
) -> None:
    if (
        manifest.get("source_admission_identity_sha256") != admission_identity
        or manifest.get("source_document_count") != len(admission_rows)
        or manifest.get("source_catalog_sha256")
        != metadata.get("source_catalog_sha256")
        or manifest.get("source_freeze_manifest_sha256")
        != metadata.get("source_freeze_manifest_sha256")
        or manifest.get("source_split") != metadata.get("frozen_source_split")
    ):
        raise ValueError("source eligibility admission lineage mismatch")

    eligible_ids: set[str] = set()
    for row in eligible:
        if set(row) != {
            "schema_version",
            "document_id",
            "language",
            "canonical_sha256",
            "canonical_payload_sha256",
            "semantic_payload_sha256",
            "semantic_record_count",
            "audit_receipt_sha256",
        } or any(not _is_sha256(row.get(key)) for key in _SHA256_KEYS):
            raise ValueError("eligible inventory row shape is invalid")
        document_id = row.get("document_id")
        if type(document_id) is not str or not document_id or document_id in eligible_ids:
            raise ValueError("eligible inventory contains an invalid document identity")
        eligible_ids.add(document_id)
        source = admission_rows.get(document_id)
        if source is None:
            raise ValueError("eligible document is absent from source admission")
        status, language, canonical_sha, canonical_blob, semantic_blob, record_count = source
        if (
            status != "admitted"
            or row.get("language") != language
            or row.get("canonical_sha256") != canonical_sha
            or row.get("semantic_record_count") != record_count
            or type(canonical_blob) is not bytes
            or type(semantic_blob) is not bytes
            or row.get("canonical_payload_sha256") != _sha256(canonical_blob)
            or row.get("semantic_payload_sha256") != _sha256(semantic_blob)
        ):
            raise ValueError("eligible inventory payload lineage mismatch")

    excluded_ids: set[str] = set()
    for row in excluded:
        if set(row) != {
            "schema_version",
            "document_id",
            "reason",
            "semantic_record_count",
            "finding_count",
        }:
            raise ValueError("excluded inventory row shape is invalid")
        document_id = row.get("document_id")
        finding_count = row.get("finding_count")
        if (
            type(document_id) is not str
            or not document_id
            or document_id in excluded_ids
            or type(finding_count) is not int
            or finding_count < 0
        ):
            raise ValueError("excluded inventory row is invalid")
        excluded_ids.add(document_id)
        source = admission_rows.get(document_id)
        if source is None:
            raise ValueError("excluded document is absent from source admission")
        status, _, _, _, _, record_count = source
        expected_reason = (
            "source_rejected" if status == "rejected" else "semantic_audit_findings"
        )
        if (
            row.get("reason") != expected_reason
            or row.get("semantic_record_count") != record_count
            or (status == "admitted" and finding_count == 0)
        ):
            raise ValueError("excluded inventory admission lineage mismatch")

    if eligible_ids & excluded_ids or eligible_ids | excluded_ids != set(admission_rows):
        raise ValueError("eligibility inventories do not partition the source admission")


def transfer_runtime_equivalent_eligibility(
    *,
    baseline_admission_database: str | Path,
    candidate_admission_database: str | Path,
    source_eligibility_root: str | Path,
    equivalence_receipt: str | Path,
    output_root: str | Path,
) -> EligibilityRuntimeTransferResult:
    """Transfer an audited eligibility partition to an exact-equivalent runtime."""

    baseline_path = Path(baseline_admission_database).resolve()
    candidate_path = Path(candidate_admission_database).resolve()
    source_root = Path(source_eligibility_root).resolve()
    output = Path(output_root).resolve()
    if baseline_path == candidate_path:
        raise ValueError("baseline and candidate admission databases must differ")
    if source_root == output:
        raise ValueError("eligibility transfer output must differ from its source")

    receipt = load_equivalence_receipt(Path(equivalence_receipt))
    fresh = compare_admission_runtime_equivalence(baseline_path, candidate_path)
    if not fresh.equivalent or receipt != equivalence_receipt_payload(fresh):
        raise ValueError("runtime equivalence receipt does not match a fresh comparison")

    source_manifest = _read_canonical_manifest(source_root / "manifest.json")
    eligible_bytes, eligible = _read_inventory(
        source_root / "eligible.jsonl",
        expected_sha256=source_manifest.get("eligible_sha256"),
        expected_count=source_manifest.get("eligible_document_count"),
        expected_schema=ELIGIBLE_DOCUMENT_SCHEMA,
    )
    excluded_bytes, excluded = _read_inventory(
        source_root / "excluded.jsonl",
        expected_sha256=source_manifest.get("excluded_sha256"),
        expected_count=source_manifest.get("excluded_document_count"),
        expected_schema=EXCLUDED_DOCUMENT_SCHEMA,
    )
    baseline_metadata, baseline_identity, baseline_rows = _load_admission(
        baseline_path
    )
    _validate_inventory_against_admission(
        source_manifest,
        eligible,
        excluded,
        baseline_metadata,
        baseline_identity,
        baseline_rows,
    )
    _, candidate_identity, _ = _load_admission(candidate_path)

    unsigned = {
        key: value
        for key, value in source_manifest.items()
        if key != "manifest_sha256"
    }
    unsigned.update(
        {
            "source_admission_identity_sha256": candidate_identity,
            "source_eligibility_manifest_sha256": source_manifest[
                "manifest_sha256"
            ],
            "source_runtime_equivalence_manifest_sha256": receipt[
                "manifest_sha256"
            ],
            "source_runtime_equivalence_normalized_documents_sha256": fresh.normalized_documents_sha256,
            "source_runtime_equivalence_baseline_parser_runtime_identity_sha256": fresh.baseline_parser_runtime_identity_sha256,
            "source_runtime_equivalence_candidate_parser_runtime_identity_sha256": fresh.candidate_parser_runtime_identity_sha256,
            "eligibility_transfer_policy": TRANSFER_POLICY,
            "training_ready": False,
        }
    )
    manifest_sha256 = _sha256(_canonical(unsigned).encode("utf-8"))
    manifest = {**unsigned, "manifest_sha256": manifest_sha256}
    publish_new_files(
        (
            (output / "eligible.jsonl", eligible_bytes),
            (output / "excluded.jsonl", excluded_bytes),
            (output / "manifest.json", f"{_canonical(manifest)}\n".encode("utf-8")),
        ),
        artifact_label="runtime-equivalent eligibility transfer",
    )
    return EligibilityRuntimeTransferResult(
        output_root=output,
        eligible_document_count=len(eligible),
        excluded_document_count=len(excluded),
        manifest_sha256=manifest_sha256,
        candidate_admission_identity_sha256=candidate_identity,
        equivalence_manifest_sha256=str(receipt["manifest_sha256"]),
        training_ready=False,
    )
