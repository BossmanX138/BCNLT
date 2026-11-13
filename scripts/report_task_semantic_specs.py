"""Report task counts by source, language, family, and semantic spec hash."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.database.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """SELECT s.source_kind,t.language,t.family,t.semantic_spec_sha256,COUNT(*)
               FROM tasks AS t
               JOIN source_example_identity AS s USING(example_id)
               GROUP BY s.source_kind,t.language,t.family,t.semantic_spec_sha256
               ORDER BY s.source_kind,t.language,t.family,t.semantic_spec_sha256"""
        ).fetchall()
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    finally:
        connection.close()
    payload = {
        "metadata": {
            key: metadata[key]
            for key in sorted(metadata)
            if "replay" in key or "identity" in key or "weight" in key
        },
        "cells": [
            {
                "source_kind": source_kind,
                "language": language,
                "family": family,
                "semantic_spec_sha256": semantic_spec,
                "count": count,
            }
            for source_kind, language, family, semantic_spec, count in rows
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
