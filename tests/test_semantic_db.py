from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from bcnlt.model_eval import OracleDefinitionLineModel, evaluate_model_file
from bcnlt.semantic_db import (
    build_database,
    export_training_jsonl,
    import_observations,
    open_database,
)


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
        language, dialect, parser_sha, spec_sha = connection.execute(
            "SELECT language, dialect, parser_bundle_sha256, semantic_spec_sha256 "
            "FROM documents LIMIT 1"
        ).fetchone()
    assert split == "evaluation_holdout"
    assert status == "unreviewed"
    assert eligible == 0
    assert language == "python"
    assert dialect == ""
    assert len(parser_sha) == 64
    assert len(spec_sha) == 64


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
    assert rows[0]["provenance"]["language"] == "python"
    assert len(rows[0]["provenance"]["parser_bundle_sha256"]) == 64
    assert len(rows[0]["provenance"]["semantic_spec_sha256"]) == 64


def test_database_keeps_language_specific_documents_and_ingestion_runs(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "sample.py").write_text("answer = 42\n", encoding="utf-8")
    (root / "sample.js").write_text("const answer = 42;\n", encoding="utf-8")
    database = tmp_path / "pairs.sqlite3"

    python = build_database(root, database, language="python")
    javascript = build_database(root, database, language="javascript")

    assert python.run_id != javascript.run_id
    with sqlite3.connect(database) as connection:
        documents = connection.execute(
            "SELECT language, source_identity FROM documents ORDER BY language"
        ).fetchall()
        run_count = connection.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0]
    assert [row[0] for row in documents] == ["javascript", "python"]
    assert documents[0][1] != documents[1][1]
    assert run_count == 2


def test_database_keeps_dialect_specific_documents_and_pairs_distinct(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "sample.py").write_text("answer = 42\n", encoding="utf-8")
    database = tmp_path / "pairs.sqlite3"

    default = build_database(root, database, language="python")
    python3 = build_database(root, database, language="python", dialect="python3")

    assert default.run_id != python3.run_id
    with sqlite3.connect(database) as connection:
        documents = connection.execute(
            "SELECT dialect, source_identity FROM documents ORDER BY dialect"
        ).fetchall()
        pair_ids = connection.execute(
            "SELECT pair_sha256 FROM semantic_pairs ORDER BY pair_sha256"
        ).fetchall()
    assert [row[0] for row in documents] == ["", "python3"]
    assert documents[0][1] != documents[1][1]
    assert len(pair_ids) == 2
    assert pair_ids[0] != pair_ids[1]


def test_open_database_migrates_legacy_python_documents_readably(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE documents (
                document_id INTEGER PRIMARY KEY,
                source_path TEXT NOT NULL UNIQUE,
                relative_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                language TEXT NOT NULL,
                byte_count INTEGER NOT NULL,
                line_count INTEGER NOT NULL,
                source_text TEXT NOT NULL,
                split TEXT NOT NULL,
                holdout_reason TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO documents(
                source_path, relative_path, source_sha256, language, byte_count,
                line_count, source_text, split, updated_at
            ) VALUES('legacy.py', 'legacy.py', 'abc', 'python', 4, 1, 'x=1', 'train_candidate', 'now')
            """
        )

    connection = open_database(database)
    try:
        row = connection.execute(
            "SELECT source_identity, language, dialect, parser_bundle_sha256, "
            "semantic_spec_sha256 FROM documents"
        ).fetchone()
        version = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0]
    finally:
        connection.close()

    with sqlite3.connect(database) as reopened:
        persisted = reopened.execute(
            "SELECT source_identity FROM documents"
        ).fetchone()[0]
        persisted_version = reopened.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0]

    assert row == ("legacy.py:python:", "python", "", "legacy-python", "legacy-python")
    assert version == "2"
    assert persisted == "legacy.py:python:"
    assert persisted_version == "2"


def test_observation_import_matches_the_report_language(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "sample.js"
    source.write_text(
        "function add(left, right) { return left + right; }\n",
        encoding="utf-8",
    )
    database = tmp_path / "pairs.sqlite3"
    build_database(root, database, language="javascript")
    report = evaluate_model_file(
        source,
        model=OracleDefinitionLineModel(),
        max_probes=100,
        language="javascript",
    )
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "language": "javascript",
                "models": {
                    "fixture": {
                        "model_id": "oracle",
                        "files": [report.to_dict()],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    connection = open_database(database)
    try:
        imported = import_observations(connection, results)
        count = connection.execute("SELECT COUNT(*) FROM model_observations").fetchone()[0]
    finally:
        connection.close()

    assert report.probe_count > 0
    assert imported == count == report.probe_count * 2
