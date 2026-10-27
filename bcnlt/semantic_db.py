from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .adapters.python import PythonAdapter
from .model_eval import _bcnlt_context, _generate_definition_probes, _standard_context
from .uncertainty import HeuristicUncertaintyScorer


SCHEMA_VERSION = "1"
GENERATOR_VERSION = "bcnlt-semantic-pairs-v1"
EXCLUDED_DIRS = {".git", ".pytest_cache", ".venv", "venv", "__pycache__", "outputs", "data"}


@dataclass(frozen=True)
class BuildStats:
    run_id: str
    documents: int
    semantic_units: int
    semantic_pairs: int
    observations: int


def discover_python_files(root: Path) -> list[Path]:
    return sorted(
        (path.resolve() for path in root.rglob("*.py") if not EXCLUDED_DIRS.intersection(path.parts)),
        key=lambda path: path.as_posix().lower(),
    )


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            root_path TEXT NOT NULL,
            generator_version TEXT NOT NULL,
            source_manifest_sha256 TEXT NOT NULL,
            document_count INTEGER NOT NULL,
            pair_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
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
        );
        CREATE TABLE IF NOT EXISTS semantic_units (
            unit_id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            event TEXT NOT NULL,
            surface TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            start_col INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            end_col INTEGER NOT NULL,
            scope TEXT NOT NULL,
            symbol_id TEXT,
            defines TEXT,
            ast_relation TEXT,
            confidence REAL NOT NULL,
            uncertainty REAL NOT NULL,
            uncertainty_signals_json TEXT NOT NULL,
            unit_json TEXT NOT NULL,
            UNIQUE(document_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS semantic_pairs (
            pair_id INTEGER PRIMARY KEY,
            pair_sha256 TEXT NOT NULL UNIQUE,
            document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            question TEXT NOT NULL,
            standard_context TEXT NOT NULL,
            bcnlt_context TEXT NOT NULL,
            expected_answer TEXT NOT NULL,
            expected_line INTEGER NOT NULL,
            surface TEXT NOT NULL,
            symbol_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            target_event TEXT NOT NULL,
            split TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'unreviewed',
            training_eligible INTEGER NOT NULL DEFAULT 0,
            provenance_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pairs_document ON semantic_pairs(document_id);
        CREATE INDEX IF NOT EXISTS idx_pairs_review ON semantic_pairs(split, review_status, training_eligible);
        CREATE TABLE IF NOT EXISTS model_observations (
            observation_id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            pair_id INTEGER NOT NULL REFERENCES semantic_pairs(pair_id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            context_kind TEXT NOT NULL,
            answer TEXT NOT NULL,
            correct INTEGER NOT NULL,
            observed_at TEXT NOT NULL,
            UNIQUE(run_id, pair_id, model_id, context_kind)
        );
        """
    )
    connection.execute(
        "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (SCHEMA_VERSION,),
    )
    return connection


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_database(
    root: Path,
    database_path: Path,
    holdout_paths: Iterable[Path] = (),
    observations_path: Path | None = None,
) -> BuildStats:
    root = root.resolve()
    files = discover_python_files(root)
    holdouts = {path.resolve() for path in holdout_paths}
    manifest = [
        {"path": _relative_path(path, root), "sha256": _sha256_bytes(path.read_bytes())}
        for path in files
    ]
    manifest_sha = _canonical_hash(manifest)
    run_id = f"ingest-{manifest_sha[:16]}"
    now = datetime.now(timezone.utc).isoformat()
    connection = open_database(database_path)
    adapter = PythonAdapter()
    scorer = HeuristicUncertaintyScorer()
    pair_total = 0
    unit_total = 0
    try:
        with connection:
            for path in files:
                source = path.read_text(encoding="utf-8-sig")
                source_sha = _sha256_bytes(source.encode("utf-8"))
                relative = _relative_path(path, root)
                split = "evaluation_holdout" if path in holdouts else "train_candidate"
                holdout_reason = "wizard-v02-vs-runeforge-mk2-head-to-head" if path in holdouts else None
                connection.execute(
                    """
                    INSERT INTO documents(
                        source_path, relative_path, source_sha256, language, byte_count,
                        line_count, source_text, split, holdout_reason, updated_at
                    ) VALUES(?, ?, ?, 'python', ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_path) DO UPDATE SET
                        relative_path=excluded.relative_path,
                        source_sha256=excluded.source_sha256,
                        byte_count=excluded.byte_count,
                        line_count=excluded.line_count,
                        source_text=excluded.source_text,
                        split=excluded.split,
                        holdout_reason=excluded.holdout_reason,
                        updated_at=excluded.updated_at
                    """,
                    (
                        str(path), relative, source_sha, len(source.encode("utf-8")),
                        len(source.splitlines()), source, split, holdout_reason, now,
                    ),
                )
                document_id = connection.execute(
                    "SELECT document_id FROM documents WHERE source_path = ?", (str(path),)
                ).fetchone()[0]
                connection.execute("DELETE FROM semantic_units WHERE document_id = ?", (document_id,))
                connection.execute("DELETE FROM semantic_pairs WHERE document_id = ?", (document_id,))
                try:
                    document = adapter.parse(source, path=str(path))
                except SyntaxError:
                    continue
                units = list(adapter.to_units(document))
                for ordinal, unit in enumerate(units):
                    score = scorer.score(unit, document.index)
                    connection.execute(
                        """
                        INSERT INTO semantic_units(
                            document_id, ordinal, event, surface, start_line, start_col,
                            end_line, end_col, scope, symbol_id, defines, ast_relation,
                            confidence, uncertainty, uncertainty_signals_json, unit_json
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id, ordinal, unit.event, unit.surface, *unit.span,
                            unit.scope, unit.symbol_id, unit.defines, unit.ast_relation,
                            score.confidence, score.uncertainty,
                            json.dumps(score.signals), json.dumps(unit.to_dict(), sort_keys=True),
                        ),
                    )
                probes = _generate_definition_probes(units, max_probes=1_000_000)
                for ordinal, probe in enumerate(probes):
                    standard = _standard_context(source, probe, 1)
                    bcnlt = _bcnlt_context(units, probe, 2)
                    identity = {
                        "source_sha256": source_sha,
                        "question": probe.question,
                        "standard_context": standard,
                        "bcnlt_context": bcnlt,
                        "expected_line": probe.target_line,
                    }
                    provenance = {
                        "generator": GENERATOR_VERSION,
                        "source_path": str(path),
                        "relative_path": relative,
                        "source_sha256": source_sha,
                        "split": split,
                        "holdout_reason": holdout_reason,
                    }
                    connection.execute(
                        """
                        INSERT INTO semantic_pairs(
                            pair_sha256, document_id, ordinal, question, standard_context,
                            bcnlt_context, expected_answer, expected_line, surface, symbol_id,
                            scope, target_event, split, review_status, training_eligible,
                            provenance_json, updated_at
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unreviewed', 0, ?, ?)
                        """,
                        (
                            _canonical_hash(identity), document_id, ordinal, probe.question,
                            standard, bcnlt, f"line {probe.target_line}", probe.target_line,
                            probe.surface, probe.symbol_id, probe.scope, probe.target_event,
                            split, json.dumps(provenance, sort_keys=True), now,
                        ),
                    )
                pair_total += len(probes)
                unit_total += len(units)
            connection.execute(
                """
                INSERT INTO ingestion_runs(
                    run_id, created_at, root_path, generator_version,
                    source_manifest_sha256, document_count, pair_count
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    document_count=excluded.document_count,
                    pair_count=excluded.pair_count
                """,
                (run_id, now, str(root), GENERATOR_VERSION, manifest_sha, len(files), pair_total),
            )
        observations = import_observations(connection, observations_path) if observations_path else 0
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    return BuildStats(run_id, len(files), unit_total, pair_total, observations)


def import_observations(connection: sqlite3.Connection, results_path: Path) -> int:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    run_hash = _sha256_bytes(results_path.read_bytes())
    run_id = f"head-to-head-{run_hash[:16]}"
    observed_at = str(payload.get("created_at") or datetime.now(timezone.utc).isoformat())
    inserted = 0
    with connection:
        for model in payload.get("models", {}).values():
            model_id = str(model["model_id"])
            for report in model.get("files", []):
                source_path = str(Path(report["path"]).resolve())
                for result in report.get("results", []):
                    probe = result["probe"]
                    row = connection.execute(
                        """
                        SELECT p.pair_id FROM semantic_pairs p
                        JOIN documents d ON d.document_id = p.document_id
                        WHERE d.source_path = ? AND p.question = ? AND p.expected_line = ?
                          AND p.standard_context = ? AND p.bcnlt_context = ?
                        """,
                        (
                            source_path, probe["question"], result["expected_line"],
                            result["standard_context"], result["bcnlt_context"],
                        ),
                    ).fetchone()
                    if row is None:
                        continue
                    for context_kind in ("standard", "bcnlt"):
                        connection.execute(
                            """
                            INSERT INTO model_observations(
                                run_id, pair_id, model_id, context_kind, answer, correct, observed_at
                            ) VALUES(?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(run_id, pair_id, model_id, context_kind) DO UPDATE SET
                                answer=excluded.answer, correct=excluded.correct,
                                observed_at=excluded.observed_at
                            """,
                            (
                                run_id, row[0], model_id, context_kind,
                                result[f"{context_kind}_answer"],
                                int(bool(result[f"{context_kind}_correct"])), observed_at,
                            ),
                        )
                        inserted += 1
    return inserted


def export_training_jsonl(database_path: Path, export_path: Path) -> int:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """
            SELECT pair_sha256, question, bcnlt_context, expected_answer, split,
                   review_status, training_eligible, provenance_json
            FROM semantic_pairs ORDER BY pair_sha256
            """
        ).fetchall()
    finally:
        connection.close()
    with export_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    {
                        "pair_id": row[0],
                        "instruction": row[1],
                        "input": row[2],
                        "output": row[3],
                        "split": row[4],
                        "review_status": row[5],
                        "training_eligible": bool(row[6]),
                        "provenance": json.loads(row[7]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return len(rows)
