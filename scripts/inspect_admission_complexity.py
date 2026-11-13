"""Estimate the downstream task-packing scale from committed admission rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-full-resumable-v1"
    / "admission.sqlite"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    arguments = parser.parse_args()
    database = arguments.database.resolve()
    if not database.is_file():
        parser.error("admission database does not exist")

    connection = sqlite3.connect(
        f"{database.as_uri()}?mode=ro",
        uri=True,
        timeout=30,
    )
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        aggregate = connection.execute(
            """SELECT COUNT(*),
                      COALESCE(SUM(semantic_record_count), 0),
                      COALESCE(AVG(semantic_record_count), 0),
                      COALESCE(MIN(semantic_record_count), 0),
                      COALESCE(MAX(semantic_record_count), 0),
                      COALESCE(SUM(semantic_token_count), 0)
               FROM documents
               WHERE status='admitted'"""
        ).fetchone()
        language_rows = connection.execute(
            """SELECT language, COUNT(*),
                      COALESCE(SUM(semantic_record_count), 0),
                      COALESCE(SUM(semantic_token_count), 0)
               FROM documents
               WHERE status='admitted'
               GROUP BY language
               ORDER BY language"""
        ).fetchall()
    finally:
        connection.close()

    if aggregate is None:
        raise RuntimeError("admission aggregate query returned no row")
    admitted_count = int(aggregate[0])
    semantic_record_count = int(aggregate[1])
    payload = {
        "admitted_document_count": admitted_count,
        "catalog_exhausted": metadata.get("catalog_exhausted") == "true",
        "database_path": str(database),
        "estimated_three_family_upper_bound": semantic_record_count * 3,
        "language_counts": {
            str(row[0]): {
                "admitted_document_count": int(row[1]),
                "semantic_record_count": int(row[2]),
                "semantic_token_count": int(row[3]),
            }
            for row in language_rows
        },
        "maximum_records_per_document": int(aggregate[4]),
        "mean_records_per_document": float(aggregate[2]),
        "minimum_records_per_document": int(aggregate[3]),
        "semantic_record_count": semantic_record_count,
        "semantic_token_count": int(aggregate[5]),
        "scope": (
            "Committed admitted rows only. The three-family value is an upper bound; "
            "packing may reject or combine records."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
