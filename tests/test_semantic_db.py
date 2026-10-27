from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from bcnlt.semantic_db import build_database, export_training_jsonl


def test_semantic_database_is_idempotent_and_marks_holdouts(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "sample.py"
    source.write_text("value = 1\n\ndef add(left, right):\n    return left + right\n", encoding="utf-8")
    database = tmp_path / "pairs.sqlite3"

    first = build_database(root, database, holdout_paths=[source])
    second = build_database(root, database, holdout_paths=[source])

    assert first.semantic_pairs == second.semantic_pairs
    assert first.semantic_pairs >= 2
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM semantic_pairs").fetchone()[0] == first.semantic_pairs
        split, status, eligible = connection.execute(
            "SELECT split, review_status, training_eligible FROM semantic_pairs LIMIT 1"
        ).fetchone()
    assert split == "evaluation_holdout"
    assert status == "unreviewed"
    assert eligible == 0


def test_training_export_contains_provenance(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "sample.py").write_text("answer = 42\n", encoding="utf-8")
    database = tmp_path / "pairs.sqlite3"
    export = tmp_path / "pairs.jsonl"
    stats = build_database(root, database)

    count = export_training_jsonl(database, export)

    rows = [json.loads(line) for line in export.read_text(encoding="utf-8").splitlines()]
    assert count == stats.semantic_pairs == 1
    assert rows[0]["output"] == "line 1"
    assert rows[0]["review_status"] == "unreviewed"
    assert rows[0]["training_eligible"] is False
    assert rows[0]["provenance"]["source_sha256"]
