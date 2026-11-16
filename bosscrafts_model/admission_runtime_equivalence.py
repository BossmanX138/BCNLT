"""Exact logical equivalence audit for two resumable admission databases."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from .immutable_artifacts import publish_new_files

SCHEMA_VERSION = "bosscrafts-admission-runtime-equivalence-v1"
ADMISSION_SCHEMA_VERSION = "bosscrafts-current-corpus-resumable-v4"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_METADATA_DIFFERENCES = frozenset(
    {"host_code_identity_sha256", "parser_runtime_identity_sha256"}
)
_DOCUMENT_COLUMNS = (
    "document_id",
    "content_sha256",
    "language",
    "relative_path",
    "status",
    "reason_code",
    "detail",
    "semantic_spec_sha256",
    "parser_runtime_identity_sha256",
    "canonical_sha256",
    "canonical_jsonl_zlib",
    "semantic_ids_zlib",
    "semantic_record_count",
    "semantic_token_count",
    "raw_token_count",
)
_PARSER_IDENTITY_INDEX = _DOCUMENT_COLUMNS.index(
    "parser_runtime_identity_sha256"
)


@dataclass(frozen=True)
class AdmissionRuntimeEquivalence:
    equivalent: bool
    baseline_parser_runtime_identity_sha256: str
    candidate_parser_runtime_identity_sha256: str
    document_count: int
    admitted_count: int
    rejected_count: int
    semantic_record_count: int
    semantic_token_count: int
    normalized_documents_sha256: str
    mismatch_count: int
    first_mismatch: dict[str, object] | None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _readonly_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("admission database must be a regular file")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT key,value FROM metadata ORDER BY key").fetchall()
    if any(type(key) is not str or type(value) is not str for key, value in rows):
        raise ValueError("admission metadata must contain strings")
    result = dict(rows)
    if len(result) != len(rows):
        raise ValueError("admission metadata contains duplicate keys")
    if result.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        raise ValueError("admission database schema mismatch")
    return result


def _verify_document_schema(connection: sqlite3.Connection) -> None:
    columns = tuple(
        str(row[1]) for row in connection.execute("PRAGMA table_info(documents)")
    )
    if columns != _DOCUMENT_COLUMNS:
        raise ValueError("admission documents schema mismatch")


def _normalized_value(value: object) -> object:
    if isinstance(value, bytes):
        return {
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if value is None or type(value) in {str, int}:
        return value
    raise ValueError("admission document contains an unsupported SQLite value")


def _normalized_row(row: tuple[Any, ...]) -> dict[str, object]:
    return {
        column: _normalized_value(value)
        for index, (column, value) in enumerate(zip(_DOCUMENT_COLUMNS, row, strict=True))
        if index != _PARSER_IDENTITY_INDEX
    }


def _identity(metadata: dict[str, str], *, label: str) -> str:
    identity = metadata.get("parser_runtime_identity_sha256")
    if type(identity) is not str or _HEX_SHA256.fullmatch(identity) is None:
        raise ValueError(f"{label} parser runtime identity is invalid")
    return identity


def compare_admission_runtime_equivalence(
    baseline_database: Path,
    candidate_database: Path,
) -> AdmissionRuntimeEquivalence:
    """Compare every logical row, excluding only the parser identity column."""

    if baseline_database.resolve() == candidate_database.resolve():
        raise ValueError("baseline and candidate admission databases must differ")
    baseline = _readonly_connection(baseline_database)
    candidate = _readonly_connection(candidate_database)
    try:
        if str(baseline.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise ValueError("baseline admission database integrity failed")
        if str(candidate.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise ValueError("candidate admission database integrity failed")
        baseline_metadata = _metadata(baseline)
        candidate_metadata = _metadata(candidate)
        _verify_document_schema(baseline)
        _verify_document_schema(candidate)
        baseline_identity = _identity(baseline_metadata, label="baseline")
        candidate_identity = _identity(candidate_metadata, label="candidate")
        if baseline_identity == candidate_identity:
            raise ValueError("runtime equivalence requires distinct parser identities")
        if candidate_identity != PARSER_RUNTIME_IDENTITY_SHA256:
            raise ValueError("candidate admission does not use the current parser runtime")

        metadata_keys = set(baseline_metadata) | set(candidate_metadata)
        metadata_mismatches = sorted(
            key
            for key in metadata_keys
            if key not in _ALLOWED_METADATA_DIFFERENCES
            and baseline_metadata.get(key) != candidate_metadata.get(key)
        )
        first_mismatch: dict[str, object] | None = None
        mismatch_count = 0
        if metadata_mismatches:
            mismatch_count += len(metadata_mismatches)
            first_key = metadata_mismatches[0]
            first_mismatch = {
                "scope": "metadata",
                "key": first_key,
                "baseline": baseline_metadata.get(first_key),
                "candidate": candidate_metadata.get(first_key),
            }

        select = f"SELECT {','.join(_DOCUMENT_COLUMNS)} FROM documents ORDER BY document_id"
        baseline_cursor = baseline.execute(select)
        candidate_cursor = candidate.execute(select)
        digest = hashlib.sha256()
        document_count = 0
        admitted_count = 0
        rejected_count = 0
        semantic_record_count = 0
        semantic_token_count = 0
        while True:
            baseline_row = baseline_cursor.fetchone()
            candidate_row = candidate_cursor.fetchone()
            if baseline_row is None and candidate_row is None:
                break
            document_count += 1
            if baseline_row is None or candidate_row is None:
                mismatch_count += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "scope": "documents",
                        "ordinal": document_count,
                        "reason": "row_count_mismatch",
                    }
                continue
            baseline_parser_identity = baseline_row[_PARSER_IDENTITY_INDEX]
            candidate_parser_identity = candidate_row[_PARSER_IDENTITY_INDEX]
            baseline_status = baseline_row[_DOCUMENT_COLUMNS.index("status")]
            if baseline_status == "admitted":
                admitted_count += 1
                semantic_record_count += int(
                    baseline_row[_DOCUMENT_COLUMNS.index("semantic_record_count")]
                )
                semantic_token_count += int(
                    baseline_row[_DOCUMENT_COLUMNS.index("semantic_token_count")]
                )
                if baseline_parser_identity != baseline_identity:
                    raise ValueError("baseline admitted row parser identity mismatch")
                if candidate_parser_identity != candidate_identity:
                    raise ValueError("candidate admitted row parser identity mismatch")
            elif baseline_status == "rejected":
                rejected_count += 1
                if baseline_parser_identity is not None or candidate_parser_identity is not None:
                    raise ValueError("rejected row unexpectedly binds a parser identity")
            else:
                raise ValueError("admission row has an invalid status")

            normalized_baseline = _normalized_row(baseline_row)
            normalized_candidate = _normalized_row(candidate_row)
            digest.update(_canonical_json(normalized_candidate).encode("utf-8"))
            digest.update(b"\n")
            if normalized_baseline != normalized_candidate:
                mismatch_count += 1
                if first_mismatch is None:
                    differing_fields = sorted(
                        key
                        for key in normalized_baseline
                        if normalized_baseline[key] != normalized_candidate[key]
                    )
                    first_mismatch = {
                        "scope": "documents",
                        "ordinal": document_count,
                        "document_id": candidate_row[0],
                        "fields": differing_fields,
                    }

        return AdmissionRuntimeEquivalence(
            equivalent=mismatch_count == 0,
            baseline_parser_runtime_identity_sha256=baseline_identity,
            candidate_parser_runtime_identity_sha256=candidate_identity,
            document_count=document_count,
            admitted_count=admitted_count,
            rejected_count=rejected_count,
            semantic_record_count=semantic_record_count,
            semantic_token_count=semantic_token_count,
            normalized_documents_sha256=digest.hexdigest(),
            mismatch_count=mismatch_count,
            first_mismatch=first_mismatch,
        )
    finally:
        candidate.close()
        baseline.close()


def equivalence_receipt_payload(
    result: AdmissionRuntimeEquivalence,
) -> dict[str, object]:
    """Return the canonical, self-hashed receipt for an equivalence result."""

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        **result.__dict__,
        "comparison_policy": (
            "exact metadata and document equality, including compressed canonical JSONL "
            "and semantic token payloads; only host-code and parser-runtime identities may differ"
        ),
        "training_ready": False,
    }
    manifest_sha256 = hashlib.sha256(
        _canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    return {**unsigned, "manifest_sha256": manifest_sha256}


def load_equivalence_receipt(path: Path) -> dict[str, object]:
    """Load a canonical equivalence receipt and verify its self-hash and shape."""

    try:
        raw = path.resolve(strict=True).read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("admission runtime equivalence receipt is invalid") from error
    if type(payload) is not dict or raw != f"{_canonical_json(payload)}\n":
        raise ValueError("admission runtime equivalence receipt is not canonical")
    unsigned = dict(payload)
    claimed = unsigned.pop("manifest_sha256", None)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or type(claimed) is not str
        or _HEX_SHA256.fullmatch(claimed) is None
        or claimed
        != hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        or payload.get("training_ready") is not False
    ):
        raise ValueError("admission runtime equivalence receipt lineage is invalid")
    return payload


def publish_equivalence_receipt(
    result: AdmissionRuntimeEquivalence,
    output: Path,
) -> str:
    payload = equivalence_receipt_payload(result)
    publish_new_files(
        ((output, f"{_canonical_json(payload)}\n".encode("utf-8")),),
        artifact_label="admission runtime equivalence receipt",
    )
    return str(payload["manifest_sha256"])
