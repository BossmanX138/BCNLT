"""Deterministic per-document delta for completed admission runtimes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .admission_runtime_equivalence import (
    _DOCUMENT_COLUMNS,
    _HEX_SHA256,
    ADMISSION_SCHEMA_VERSIONS,
    _canonical_json,
    _readonly_connection,
    _verify_document_schema,
)
from .bcnlt_adapter import MAX_CANONICAL_OUTPUT_BYTES
from .bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from .compression import decompress_bounded
from .immutable_artifacts import publish_new_files

SCHEMA_VERSION = "bosscrafts-admission-runtime-delta-v1"
_REQUIRED_EQUAL_METADATA = (
    "schema_version",
    "source_catalog_sha256",
    "source_freeze_manifest_sha256",
    "review_manifest_sha256",
    "tokenizer_sha256",
    "vocabulary_size",
    "expected_source_document_count",
)
_PROVENANCE_COLUMNS = frozenset(
    {"semantic_spec_sha256", "parser_runtime_identity_sha256"}
)
_RUNTIME_METADATA_DIFFERENCES = frozenset(
    {
        "binding_sha256",
        "host_code_identity_sha256",
        "parser_runtime_identity_sha256",
        "semantic_audit_qualification_sha256",
        "semantic_spec_sha256",
    }
)
_ROW_INDEX = {column: index for index, column in enumerate(_DOCUMENT_COLUMNS)}


@dataclass(frozen=True)
class AdmissionDocumentDelta:
    document_id: str
    classification: Literal["reusable", "changed", "baseline_only", "candidate_only"]
    status: str | None
    language: str | None
    content_sha256: str | None
    baseline_semantic_spec_sha256: str | None
    candidate_semantic_spec_sha256: str | None
    differing_fields: tuple[str, ...]


@dataclass(frozen=True)
class AdmissionRuntimeDelta:
    baseline_database_sha256: str
    baseline_database_bytes: int
    candidate_database_sha256: str
    candidate_database_bytes: int
    baseline_parser_runtime_identity_sha256: str
    candidate_parser_runtime_identity_sha256: str
    source_catalog_sha256: str
    source_freeze_manifest_sha256: str
    expected_source_document_count: int
    baseline_document_count: int
    candidate_document_count: int
    reusable_document_count: int
    reusable_admitted_count: int
    reusable_rejected_count: int
    changed_document_count: int
    baseline_only_document_count: int
    candidate_only_document_count: int
    baseline_normalized_documents_sha256: str
    candidate_normalized_documents_sha256: str
    reusable_documents_sha256: str
    documents: tuple[AdmissionDocumentDelta, ...]


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _require_standalone_database(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} admission database must be a regular file")
    wal = Path(f"{path}-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"{label} admission database still has a write-ahead log")


def _metadata(connection: sqlite3.Connection, *, label: str) -> dict[str, str]:
    try:
        rows = connection.execute(
            "SELECT key,value FROM metadata ORDER BY key"
        ).fetchall()
    except sqlite3.DatabaseError as error:
        raise ValueError(f"{label} admission metadata is unavailable") from error
    if any(type(key) is not str or type(value) is not str for key, value in rows):
        raise ValueError(f"{label} admission metadata must contain strings")
    result = dict(rows)
    if len(result) != len(rows):
        raise ValueError(f"{label} admission metadata contains duplicate keys")
    if result.get("schema_version") not in ADMISSION_SCHEMA_VERSIONS:
        raise ValueError(f"{label} admission database schema mismatch")
    return result


def _identity(metadata: dict[str, str], *, label: str) -> str:
    identity = metadata.get("parser_runtime_identity_sha256")
    if type(identity) is not str or _HEX_SHA256.fullmatch(identity) is None:
        raise ValueError(f"{label} parser runtime identity is invalid")
    return identity


def _expected_count(metadata: dict[str, str], *, label: str) -> int:
    raw = metadata.get("expected_source_document_count")
    try:
        count = int(raw) if raw is not None else 0
    except ValueError as error:
        raise ValueError(f"{label} expected source document count is invalid") from error
    if count < 1 or str(count) != raw:
        raise ValueError(f"{label} expected source document count is invalid")
    if metadata.get("catalog_exhausted") != "true":
        raise ValueError(f"{label} admission database is incomplete")
    return count


def _runtime_specs(metadata: dict[str, str], *, label: str) -> dict[str, str]:
    try:
        value = json.loads(metadata["semantic_spec_sha256"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} semantic specification metadata is invalid") from error
    if (
        type(value) is not dict
        or any(
            type(language) is not str
            or type(spec) is not str
            or _HEX_SHA256.fullmatch(spec) is None
            for language, spec in value.items()
        )
    ):
        raise ValueError(f"{label} semantic specification metadata is invalid")
    return value


def _normalized_value(value: object) -> object:
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if value is None or type(value) in {str, int}:
        return value
    raise ValueError("admission document contains an unsupported SQLite value")


def _normalized_row(row: tuple[Any, ...]) -> dict[str, object]:
    return {
        column: _normalized_value(value)
        for column, value in zip(_DOCUMENT_COLUMNS, row, strict=True)
    }


def _validate_row(
    row: tuple[Any, ...],
    *,
    identity: str,
    specs: dict[str, str],
    label: str,
) -> None:
    document_id = row[_ROW_INDEX["document_id"]]
    language = row[_ROW_INDEX["language"]]
    status = row[_ROW_INDEX["status"]]
    spec = row[_ROW_INDEX["semantic_spec_sha256"]]
    parser_identity = row[_ROW_INDEX["parser_runtime_identity_sha256"]]
    if type(document_id) is not str or not document_id:
        raise ValueError(f"{label} admission document identity is invalid")
    if type(language) is not str or language not in specs:
        raise ValueError(f"{label} admission document language is invalid")
    if status == "admitted":
        if spec != specs[language] or parser_identity != identity:
            raise ValueError(f"{label} admitted row provenance mismatch")
        if any(
            row[_ROW_INDEX[column]] is None
            for column in ("canonical_sha256", "canonical_jsonl_zlib", "semantic_ids_zlib")
        ):
            raise ValueError(f"{label} admitted row payload is incomplete")
        canonical_sha256 = row[_ROW_INDEX["canonical_sha256"]]
        canonical_payload = row[_ROW_INDEX["canonical_jsonl_zlib"]]
        semantic_payload = row[_ROW_INDEX["semantic_ids_zlib"]]
        semantic_record_count = row[_ROW_INDEX["semantic_record_count"]]
        semantic_token_count = row[_ROW_INDEX["semantic_token_count"]]
        raw_token_count = row[_ROW_INDEX["raw_token_count"]]
        if (
            type(canonical_sha256) is not str
            or _HEX_SHA256.fullmatch(canonical_sha256) is None
            or type(canonical_payload) is not bytes
            or type(semantic_payload) is not bytes
            or type(semantic_record_count) is not int
            or semantic_record_count < 1
            or type(semantic_token_count) is not int
            or semantic_token_count < 3
            or type(raw_token_count) is not int
            or raw_token_count < 3
        ):
            raise ValueError(f"{label} admitted row payload metadata is invalid")
        canonical = decompress_bounded(
            canonical_payload,
            maximum_output_bytes=MAX_CANONICAL_OUTPUT_BYTES,
        )
        semantic_ids = decompress_bounded(
            semantic_payload,
            maximum_output_bytes=semantic_token_count * 2,
        )
        if (
            hashlib.sha256(canonical).hexdigest() != canonical_sha256
            or len(semantic_ids) != semantic_token_count * 2
        ):
            raise ValueError(f"{label} admitted row payload integrity failed")
    elif status == "rejected":
        if any(
            row[_ROW_INDEX[column]] is not None
            for column in (
                "semantic_spec_sha256",
                "parser_runtime_identity_sha256",
                "canonical_sha256",
                "canonical_jsonl_zlib",
                "semantic_ids_zlib",
            )
        ):
            raise ValueError(f"{label} rejected row unexpectedly contains a payload")
        if any(
            row[_ROW_INDEX[column]] != 0
            for column in (
                "semantic_record_count",
                "semantic_token_count",
                "raw_token_count",
            )
        ):
            raise ValueError(f"{label} rejected row counts are invalid")
    else:
        raise ValueError(f"{label} admission row has an invalid status")


def _differing_fields(
    baseline: tuple[Any, ...], candidate: tuple[Any, ...]
) -> tuple[str, ...]:
    return tuple(
        column
        for column, left, right in zip(
            _DOCUMENT_COLUMNS, baseline, candidate, strict=True
        )
        if column not in _PROVENANCE_COLUMNS
        and left != right
    )


def _entry(
    classification: Literal["reusable", "changed", "baseline_only", "candidate_only"],
    baseline: tuple[Any, ...] | None,
    candidate: tuple[Any, ...] | None,
    fields: tuple[str, ...] = (),
) -> AdmissionDocumentDelta:
    row = candidate if candidate is not None else baseline
    if row is None:
        raise RuntimeError("delta entry requires an admission row")
    return AdmissionDocumentDelta(
        document_id=str(row[_ROW_INDEX["document_id"]]),
        classification=classification,
        status=str(row[_ROW_INDEX["status"]]),
        language=str(row[_ROW_INDEX["language"]]),
        content_sha256=str(row[_ROW_INDEX["content_sha256"]]),
        baseline_semantic_spec_sha256=(
            None
            if baseline is None
            or baseline[_ROW_INDEX["semantic_spec_sha256"]] is None
            else str(baseline[_ROW_INDEX["semantic_spec_sha256"]])
        ),
        candidate_semantic_spec_sha256=(
            None
            if candidate is None
            or candidate[_ROW_INDEX["semantic_spec_sha256"]] is None
            else str(candidate[_ROW_INDEX["semantic_spec_sha256"]])
        ),
        differing_fields=fields,
    )


def compare_admission_runtime_delta(
    baseline_database: Path,
    candidate_database: Path,
) -> AdmissionRuntimeDelta:
    """Classify exact per-document reuse across two completed admissions."""

    baseline_path = baseline_database.resolve(strict=True)
    candidate_path = candidate_database.resolve(strict=True)
    if baseline_path == candidate_path:
        raise ValueError("baseline and candidate admission databases must differ")
    _require_standalone_database(baseline_path, label="baseline")
    _require_standalone_database(candidate_path, label="candidate")
    baseline_hash, baseline_bytes = _sha256_file(baseline_path)
    candidate_hash, candidate_bytes = _sha256_file(candidate_path)
    baseline = _readonly_connection(baseline_path)
    candidate = _readonly_connection(candidate_path)
    try:
        if str(baseline.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise ValueError("baseline admission database integrity failed")
        if str(candidate.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise ValueError("candidate admission database integrity failed")
        _verify_document_schema(baseline)
        _verify_document_schema(candidate)
        baseline_metadata = _metadata(baseline, label="baseline")
        candidate_metadata = _metadata(candidate, label="candidate")
        if baseline_metadata.keys() != candidate_metadata.keys():
            raise ValueError("admission metadata key set mismatch")
        expected_count = _expected_count(baseline_metadata, label="baseline")
        if _expected_count(candidate_metadata, label="candidate") != expected_count:
            raise ValueError("admission expected document count mismatch")
        for key in baseline_metadata:
            if (
                key not in _RUNTIME_METADATA_DIFFERENCES
                and baseline_metadata[key] != candidate_metadata[key]
            ):
                raise ValueError(f"admission source lineage mismatch: {key}")
        for key in _REQUIRED_EQUAL_METADATA:
            if baseline_metadata.get(key) != candidate_metadata.get(key):
                raise ValueError(f"admission source lineage mismatch: {key}")
        baseline_identity = _identity(baseline_metadata, label="baseline")
        candidate_identity = _identity(candidate_metadata, label="candidate")
        if baseline_identity == candidate_identity:
            raise ValueError("runtime delta requires distinct parser identities")
        if candidate_identity != PARSER_RUNTIME_IDENTITY_SHA256:
            raise ValueError("candidate admission does not use the current parser runtime")
        baseline_specs = _runtime_specs(baseline_metadata, label="baseline")
        candidate_specs = _runtime_specs(candidate_metadata, label="candidate")
        if candidate_specs != CURRENT_RUNTIME_SPEC_SHA256:
            raise ValueError("candidate semantic specifications are not current")

        select = f"SELECT {','.join(_DOCUMENT_COLUMNS)} FROM documents ORDER BY document_id"
        baseline_cursor = baseline.execute(select)
        candidate_cursor = candidate.execute(select)
        baseline_row = baseline_cursor.fetchone()
        candidate_row = candidate_cursor.fetchone()
        baseline_count = 0
        candidate_count = 0
        entries: list[AdmissionDocumentDelta] = []
        baseline_digest = hashlib.sha256()
        candidate_digest = hashlib.sha256()
        reusable_digest = hashlib.sha256()
        while baseline_row is not None or candidate_row is not None:
            if baseline_row is not None and (
                candidate_row is None
                or str(baseline_row[0]) < str(candidate_row[0])
            ):
                _validate_row(
                    baseline_row,
                    identity=baseline_identity,
                    specs=baseline_specs,
                    label="baseline",
                )
                normalized = _normalized_row(baseline_row)
                baseline_digest.update(f"{_canonical_json(normalized)}\n".encode())
                baseline_count += 1
                entries.append(_entry("baseline_only", baseline_row, None))
                baseline_row = baseline_cursor.fetchone()
                continue
            if candidate_row is not None and (
                baseline_row is None
                or str(candidate_row[0]) < str(baseline_row[0])
            ):
                _validate_row(
                    candidate_row,
                    identity=candidate_identity,
                    specs=candidate_specs,
                    label="candidate",
                )
                normalized = _normalized_row(candidate_row)
                candidate_digest.update(f"{_canonical_json(normalized)}\n".encode())
                candidate_count += 1
                entries.append(_entry("candidate_only", None, candidate_row))
                candidate_row = candidate_cursor.fetchone()
                continue

            if baseline_row is None or candidate_row is None:
                raise RuntimeError("unreachable admission delta merge state")
            _validate_row(
                baseline_row,
                identity=baseline_identity,
                specs=baseline_specs,
                label="baseline",
            )
            _validate_row(
                candidate_row,
                identity=candidate_identity,
                specs=candidate_specs,
                label="candidate",
            )
            normalized_baseline = _normalized_row(baseline_row)
            normalized_candidate = _normalized_row(candidate_row)
            baseline_digest.update(
                f"{_canonical_json(normalized_baseline)}\n".encode()
            )
            candidate_digest.update(
                f"{_canonical_json(normalized_candidate)}\n".encode()
            )
            baseline_count += 1
            candidate_count += 1
            fields = _differing_fields(baseline_row, candidate_row)
            classification: Literal["reusable", "changed"] = (
                "reusable" if not fields else "changed"
            )
            item = _entry(classification, baseline_row, candidate_row, fields)
            entries.append(item)
            if classification == "reusable":
                reusable_digest.update(
                    f"{_canonical_json(asdict(item))}\n".encode("utf-8")
                )
            baseline_row = baseline_cursor.fetchone()
            candidate_row = candidate_cursor.fetchone()

        if baseline_count != expected_count or candidate_count != expected_count:
            raise ValueError("completed admission document count mismatch")
        reusable = [item for item in entries if item.classification == "reusable"]
        if _sha256_file(baseline_path) != (baseline_hash, baseline_bytes):
            raise ValueError("baseline admission database changed during comparison")
        if _sha256_file(candidate_path) != (candidate_hash, candidate_bytes):
            raise ValueError("candidate admission database changed during comparison")
        return AdmissionRuntimeDelta(
            baseline_database_sha256=baseline_hash,
            baseline_database_bytes=baseline_bytes,
            candidate_database_sha256=candidate_hash,
            candidate_database_bytes=candidate_bytes,
            baseline_parser_runtime_identity_sha256=baseline_identity,
            candidate_parser_runtime_identity_sha256=candidate_identity,
            source_catalog_sha256=str(candidate_metadata["source_catalog_sha256"]),
            source_freeze_manifest_sha256=str(
                candidate_metadata["source_freeze_manifest_sha256"]
            ),
            expected_source_document_count=expected_count,
            baseline_document_count=baseline_count,
            candidate_document_count=candidate_count,
            reusable_document_count=len(reusable),
            reusable_admitted_count=sum(item.status == "admitted" for item in reusable),
            reusable_rejected_count=sum(item.status == "rejected" for item in reusable),
            changed_document_count=sum(
                item.classification == "changed" for item in entries
            ),
            baseline_only_document_count=sum(
                item.classification == "baseline_only" for item in entries
            ),
            candidate_only_document_count=sum(
                item.classification == "candidate_only" for item in entries
            ),
            baseline_normalized_documents_sha256=baseline_digest.hexdigest(),
            candidate_normalized_documents_sha256=candidate_digest.hexdigest(),
            reusable_documents_sha256=reusable_digest.hexdigest(),
            documents=tuple(entries),
        )
    finally:
        candidate.close()
        baseline.close()


def delta_receipt_payload(result: AdmissionRuntimeDelta) -> dict[str, object]:
    """Return the canonical, self-hashed delta receipt."""

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        **asdict(result),
        "comparison_policy": (
            "a document is reusable only when every admission field except "
            "semantic-spec and parser-runtime provenance is exactly equal"
        ),
        "training_ready": False,
    }
    manifest_sha256 = hashlib.sha256(
        _canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    return {**unsigned, "manifest_sha256": manifest_sha256}


def load_delta_receipt(path: Path) -> dict[str, object]:
    """Load and verify a canonical admission runtime delta receipt."""

    try:
        raw = path.resolve(strict=True).read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("admission runtime delta receipt is invalid") from error
    if type(payload) is not dict or raw != f"{_canonical_json(payload)}\n":
        raise ValueError("admission runtime delta receipt is not canonical")
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
        raise ValueError("admission runtime delta receipt lineage is invalid")
    return payload


def publish_delta_receipt(result: AdmissionRuntimeDelta, output: Path) -> str:
    """Publish a new immutable delta receipt and return its manifest hash."""

    payload = delta_receipt_payload(result)
    publish_new_files(
        ((output, f"{_canonical_json(payload)}\n".encode("utf-8")),),
        artifact_label="admission runtime delta receipt",
    )
    return str(payload["manifest_sha256"])
