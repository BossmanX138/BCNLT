"""Derive a balanced short-target generation-repair curriculum from clean v5."""
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

from bosscrafts_model.balanced_curriculum import CURRICULUM_SCHEMA, TASK_FAMILIES
from bosscrafts_model.resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    _canonical_json,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-deterministic-clean-v5"
    / "tasks.sqlite"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-generation-repair-v6"
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
            "SELECT split,language,family,COUNT(*) FROM tasks "
            "GROUP BY split,language,family"
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-per-cell", type=int, default=256)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    arguments = parser.parse_args()
    if not 1 <= arguments.train_per_cell <= 2048:
        parser.error("train-per-cell must be between 1 and 2048")
    if not 16 <= arguments.max_target_tokens <= 2048:
        parser.error("max-target-tokens must be between 16 and 2048")

    source_path = arguments.source.resolve()
    destination = arguments.output.resolve()
    if destination.exists():
        raise ValueError("generation repair curriculum already exists")
    binding = NativeTokenizerBinding.load()
    source_audit = audit_task_database(binding, source_path)
    source = _ro_connect(source_path)
    try:
        source_metadata = dict(source.execute("SELECT key,value FROM metadata"))
        if source_metadata.get("schema_version") != TASK_DATABASE_SCHEMA:
            raise ValueError("source curriculum schema mismatch")
        heldout_counts = {
            cell: count
            for cell, count in _cell_counts(source).items()
            if cell[0] in {"validation", "test"}
        }
        languages = tuple(
            row[0]
            for row in source.execute(
                "SELECT DISTINCT language FROM tasks ORDER BY language"
            )
        )
        availability = {
            (str(language), str(family)): int(count)
            for language, family, count in source.execute(
                """SELECT language,family,COUNT(*) FROM tasks
                   WHERE split='train'
                     AND token_count-loss_start_index<=?
                   GROUP BY language,family""",
                (arguments.max_target_tokens,),
            )
        }
    finally:
        source.close()

    required_cells = {
        (language, family)
        for language in languages
        for family in TASK_FAMILIES
    }
    missing = {
        f"{language}|{family}": availability.get((language, family), 0)
        for language, family in sorted(required_cells)
        if availability.get((language, family), 0) < arguments.train_per_cell
    }
    if missing:
        raise ValueError(
            "insufficient short-target repair examples: "
            + json.dumps(missing, sort_keys=True)
        )

    policy = {
        "schema_version": "bosscrafts-generation-repair-curriculum-v1",
        "source_database_identity_sha256": source_audit.database_identity_sha256,
        "families": list(TASK_FAMILIES),
        "languages": list(languages),
        "train_per_cell": arguments.train_per_cell,
        "maximum_target_tokens": arguments.max_target_tokens,
        "selection": "lowest example_id per language-family after target cap",
        "heldout_policy": "copy validation and test rows byte-for-byte from source",
    }
    policy_sha256 = hashlib.sha256(
        _canonical_json(policy).encode("utf-8")
    ).hexdigest()
    output_metadata = dict(source_metadata)
    output_metadata.update(
        {
            "curriculum_schema_version": CURRICULUM_SCHEMA,
            "curriculum_policy_sha256": policy_sha256,
            "generation_repair_policy_sha256": policy_sha256,
            "repair_parent_database_identity_sha256": (
                source_audit.database_identity_sha256
            ),
            "repair_train_per_cell": str(arguments.train_per_cell),
            "repair_maximum_target_tokens": str(arguments.max_target_tokens),
            "repair_families": ",".join(TASK_FAMILIES),
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
        output.execute("CREATE INDEX tasks_cell ON tasks(split,language,family)")
        output.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)",
            sorted(output_metadata.items()),
        )
        output.execute("ATTACH DATABASE ? AS source", (str(source_path),))
        output.execute(
            "INSERT INTO tasks SELECT * FROM source.tasks "
            "WHERE split IN ('validation','test')"
        )
        output.execute(
            """INSERT INTO tasks
               SELECT example_id,document_id,language,family,record_index,split,
                      input_ids_zlib,token_count,loss_start_index,
                      semantic_spec_sha256,source_sha256
               FROM (
                   SELECT t.*,
                          ROW_NUMBER() OVER (
                              PARTITION BY language,family ORDER BY example_id
                          ) AS repair_rank
                   FROM source.tasks AS t
                   WHERE split='train'
                     AND token_count-loss_start_index<=?
               )
               WHERE repair_rank<=?""",
            (arguments.max_target_tokens, arguments.train_per_cell),
        )
        output.commit()
        observed = _cell_counts(output)
        expected = dict(heldout_counts)
        expected.update(
            {
                ("train", language, family): arguments.train_per_cell
                for language in languages
                for family in TASK_FAMILIES
            }
        )
        if observed != expected:
            raise ValueError("derived repair curriculum cell counts are invalid")
        leaked = output.execute(
            """SELECT COUNT(*) FROM tasks AS train
               WHERE train.split='train' AND EXISTS (
                   SELECT 1 FROM tasks AS heldout
                   WHERE heldout.split IN ('validation','test')
                     AND heldout.example_id=train.example_id
               )"""
        ).fetchone()[0]
        if int(leaked) != 0:
            raise ValueError("held-out example leaked into repair training")
    except BaseException:
        output.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        output.close()

    try:
        result = audit_task_database(binding, temporary)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.replace(destination)
    payload = asdict(result)
    payload.update(
        {
            "database_path": str(destination),
            "source_database_identity_sha256": (
                source_audit.database_identity_sha256
            ),
            "generation_repair_policy_sha256": policy_sha256,
            "train_per_cell": arguments.train_per_cell,
            "maximum_target_tokens": arguments.max_target_tokens,
            "repair_families": list(TASK_FAMILIES),
            "available_short_examples_by_cell": {
                f"{language}|{family}": availability[(language, family)]
                for language, family in sorted(required_cells)
            },
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
