"""Derive a lineage-bound curriculum containing only newly added train examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PINNED_TOKENIZERS_SITE = PROJECT_ROOT / "vendor" / "tokenizers_runtime_0_23"
if not (PINNED_TOKENIZERS_SITE / "tokenizers" / "tokenizers.pyd").is_file():
    raise RuntimeError("project-contained tokenizers 0.23 runtime is missing")
if "tokenizers" in sys.modules:
    raise RuntimeError("tokenizers was imported before the pinned runtime was selected")
sys.path.insert(0, str(PINNED_TOKENIZERS_SITE))

import tokenizers

if tokenizers.__version__ != "0.23.2":
    raise RuntimeError("project-contained tokenizers runtime version mismatch")

from bosscrafts_model.balanced_curriculum import CURRICULUM_SCHEMA
from bosscrafts_model.resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    _canonical_json,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_BASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-balanced-curriculum-v1"
    / "tasks.sqlite"
)
DEFAULT_EXPANDED = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-balanced-expanded-v2"
    / "tasks.sqlite"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-novel-train-v2"
    / "tasks.sqlite"
)


def _ro_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _cell_counts(connection: sqlite3.Connection) -> dict[tuple[str, str, str], int]:
    return {
        (str(split), str(language), str(family)): int(count)
        for split, language, family, count in connection.execute(
            "SELECT split, language, family, COUNT(*) FROM tasks GROUP BY split, language, family"
        )
    }


def _heldout_ids(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT example_id FROM tasks WHERE split IN ('validation','test')"
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--expanded", type=Path, default=DEFAULT_EXPANDED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    base_path = arguments.base.resolve()
    expanded_path = arguments.expanded.resolve()
    destination = arguments.output.resolve()
    if destination.exists():
        raise ValueError("novel training curriculum already exists")

    binding = NativeTokenizerBinding.load()
    base_audit = audit_task_database(binding, base_path)
    expanded_audit = audit_task_database(binding, expanded_path)
    base = _ro_connect(base_path)
    expanded = _ro_connect(expanded_path)
    try:
        base_metadata = dict(base.execute("SELECT key, value FROM metadata"))
        expanded_metadata = dict(expanded.execute("SELECT key, value FROM metadata"))
        if base_metadata.get("schema_version") != TASK_DATABASE_SCHEMA:
            raise ValueError("base curriculum schema mismatch")
        if expanded_metadata.get("schema_version") != TASK_DATABASE_SCHEMA:
            raise ValueError("expanded curriculum schema mismatch")
        lineage_keys = (
            "binding_sha256",
            "tokenizer_sha256",
            "vocabulary_size",
            "maximum_sequence_tokens",
            "source_admission_identity_sha256",
            "source_catalog_sha256",
            "review_manifest_sha256",
            "semantic_spec_sha256",
        )
        if any(base_metadata.get(key) != expanded_metadata.get(key) for key in lineage_keys):
            raise ValueError("base and expanded curriculum lineage mismatch")
        if _heldout_ids(base) != _heldout_ids(expanded):
            raise ValueError("expanded curriculum changed the frozen held-out examples")
        base_counts = _cell_counts(base)
        expanded_counts = _cell_counts(expanded)
        for cell, base_count in base_counts.items():
            expected = base_count * 2 if cell[0] == "train" else base_count
            if expanded_counts.get(cell) != expected:
                raise ValueError(f"expanded curriculum cell count mismatch: {cell}")
        base_train_ids = {
            str(row[0])
            for row in base.execute("SELECT example_id FROM tasks WHERE split='train'")
        }
        expanded_train_ids = {
            str(row[0])
            for row in expanded.execute("SELECT example_id FROM tasks WHERE split='train'")
        }
        if not base_train_ids.issubset(expanded_train_ids):
            raise ValueError("expanded curriculum does not contain every base train example")
    finally:
        base.close()
        expanded.close()

    policy = {
        "schema_version": "bosscrafts-novel-training-curriculum-v1",
        "base_database_identity_sha256": base_audit.database_identity_sha256,
        "expanded_database_identity_sha256": expanded_audit.database_identity_sha256,
        "selection": "expanded heldout unchanged; train excludes every base train example_id",
    }
    policy_sha256 = hashlib.sha256(
        _canonical_json(policy).encode("utf-8")
    ).hexdigest()
    output_metadata = dict(expanded_metadata)
    output_metadata.update(
        {
            "curriculum_schema_version": CURRICULUM_SCHEMA,
            "curriculum_policy_sha256": policy_sha256,
            "novel_training_policy_sha256": policy_sha256,
            "parent_base_database_identity_sha256": base_audit.database_identity_sha256,
            "parent_expanded_database_identity_sha256": expanded_audit.database_identity_sha256,
            "packing_complete": "true",
        }
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    output = sqlite3.connect(temporary)
    try:
        output.execute("PRAGMA journal_mode=DELETE")
        output.execute("PRAGMA synchronous=FULL")
        output.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        output.execute(
            """CREATE TABLE tasks (
                   example_id TEXT PRIMARY KEY,
                   document_id TEXT NOT NULL,
                   language TEXT NOT NULL,
                   family TEXT NOT NULL,
                   record_index INTEGER NOT NULL,
                   split TEXT NOT NULL,
                   input_ids_zlib BLOB NOT NULL,
                   token_count INTEGER NOT NULL,
                   loss_start_index INTEGER NOT NULL,
                   semantic_spec_sha256 TEXT NOT NULL,
                   source_sha256 TEXT NOT NULL
               )"""
        )
        output.execute("CREATE INDEX tasks_cell ON tasks(split, language, family)")
        output.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)",
            sorted(output_metadata.items()),
        )
        output.execute("ATTACH DATABASE ? AS expanded", (str(expanded_path),))
        output.execute("ATTACH DATABASE ? AS base", (str(base_path),))
        output.execute(
            """INSERT INTO tasks
               SELECT e.* FROM expanded.tasks AS e
               WHERE e.split != 'train'
                  OR NOT EXISTS (
                      SELECT 1 FROM base.tasks AS b
                      WHERE b.split='train' AND b.example_id=e.example_id
                  )"""
        )
        output.commit()
        derived_counts = _cell_counts(output)
        if derived_counts != base_counts:
            raise ValueError("derived novel curriculum is not exactly cell-balanced")
    except BaseException:
        output.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        output.close()

    try:
        result = audit_task_database(binding, temporary)
        if result.split_counts != base_audit.split_counts:
            raise ValueError("derived novel curriculum split counts changed")
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.replace(destination)
    payload = asdict(result)
    payload["database_path"] = str(destination)
    payload["base_database_identity_sha256"] = base_audit.database_identity_sha256
    payload["expanded_database_identity_sha256"] = expanded_audit.database_identity_sha256
    payload["novel_training_policy_sha256"] = policy_sha256
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
