from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bcnlt.semantic_db import import_observations, open_database


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a completed head-to-head run without rebuilding semantic pairs."
    )
    parser.add_argument("observations", type=Path)
    parser.add_argument(
        "--database", type=Path, default=Path("data/bcnlt_semantic_pairs.sqlite3")
    )
    args = parser.parse_args()

    database = args.database.resolve()
    observations = args.observations.resolve()
    connection = open_database(database)
    try:
        imported = import_observations(connection, observations)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        models = [
            {"model_id": row[0], "observations": row[1]}
            for row in connection.execute(
                "SELECT model_id, COUNT(*) FROM model_observations "
                "GROUP BY model_id ORDER BY model_id"
            )
        ]
        languages = [
            {"language": row[0], "documents": row[1]}
            for row in connection.execute(
                "SELECT language, COUNT(*) FROM documents "
                "GROUP BY language ORDER BY language"
            )
        ]
        schema_version = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0]
    finally:
        connection.close()
    print(
        json.dumps(
            {
                "database": str(database),
                "schema_version": schema_version,
                "imported": imported,
                "languages": languages,
                "models": models,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
