"""Read-only committed admission progress; never an authorization to train."""

import sqlite3
from pathlib import Path

LEGACY_ADMISSION_SCHEMA = "bosscrafts-current-corpus-resumable-v1"
CURRENT_ADMISSION_SCHEMA = "bosscrafts-current-corpus-resumable-v4"
SUPPORTED_ADMISSION_SCHEMAS = frozenset(
    (LEGACY_ADMISSION_SCHEMA, CURRENT_ADMISSION_SCHEMA)
)
_REQUIRED_DOCUMENT_COLUMNS = frozenset(
    ("status", "language", "reason_code", "semantic_token_count", "raw_token_count")
)


def read_admission_progress(database: str | Path) -> dict[str, object]:
    path = Path(database).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        schema_version = metadata.get("schema_version")
        if schema_version not in SUPPORTED_ADMISSION_SCHEMAS:
            raise ValueError("unsupported admission schema")
        if metadata.get("catalog_exhausted") not in ("true", "false"):
            raise ValueError("invalid catalog exhaustion marker")
        document_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(documents)")
        }
        if not _REQUIRED_DOCUMENT_COLUMNS.issubset(document_columns):
            raise ValueError("unsupported admission documents table")
        rows = connection.execute(
            "SELECT status,COUNT(*),COALESCE(SUM(semantic_token_count),0),COALESCE(SUM(raw_token_count),0) FROM documents GROUP BY status"
        ).fetchall()
        if any(row[0] not in ("admitted", "rejected") for row in rows):
            raise ValueError("invalid document status")
        counts = {row[0]: row[1] for row in rows}
        return {
            "database_path": str(path),
            "schema_version": schema_version,
            "processed_total": sum(counts.values()),
            "admitted_count": counts.get("admitted", 0),
            "rejected_count": counts.get("rejected", 0),
            "semantic_token_count": sum(row[2] for row in rows),
            "raw_token_count": sum(row[3] for row in rows),
            "catalog_exhausted": metadata["catalog_exhausted"] == "true",
            "by_language": dict(
                connection.execute(
                    "SELECT language,COUNT(*) FROM documents GROUP BY language"
                )
            ),
            "rejection_reasons": dict(
                connection.execute(
                    "SELECT reason_code,COUNT(*) FROM documents WHERE status='rejected' GROUP BY reason_code"
                )
            ),
            "training_ready": False,
            "scope": "Committed snapshot only; no parser, provenance, packing or training readiness validation.",
        }
    finally:
        connection.close()
