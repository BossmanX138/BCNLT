"""Build the held-out-safe v10 instruction curriculum with complete v9 replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys


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

from bosscrafts_model.instruction_curriculum import INSTRUCTION_FAMILY
from bosscrafts_model.resumable_task_packing import (
    CELL_WEIGHT_MULTIPLIERS_METADATA_KEY,
    TASK_DATABASE_SCHEMA,
    ResumableTaskDataset,
    _canonical_json,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


CURRICULUM_SCHEMA = "bosscrafts-instruction-replay-curriculum-v1"
EXPECTED_REPLAY_IDENTITY = (
    "677925e6500d03c546dfc198e911398d58e5791d9995b46f02d821919378127b"
)
EXPECTED_CLEAN_REPLAY_IDENTITY = (
    "626186864cf4444db997bea5bd8ea4a18f18b5042509033a0d6939d0bd35afd2"
)
EXPECTED_INSTRUCTION_IDENTITY = (
    "d1d94a21744cc8857ce32d3b98ea2459741e7e9cda9d62548e7e362f5da80e5d"
)
DEFAULT_REPLAY = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-long-generation-repair-v9"
    / "tasks.sqlite"
)
DEFAULT_INSTRUCTION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-instruction-completion-python-v10"
    / "tasks.sqlite"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-instruction-replay-v10"
    / "tasks.sqlite"
)


def _ro_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _metadata(path: Path) -> dict[str, str]:
    connection = _ro_connect(path)
    try:
        return dict(connection.execute("SELECT key,value FROM metadata"))
    finally:
        connection.close()


def _cell_counts(path: Path) -> dict[tuple[str, str, str], int]:
    connection = _ro_connect(path)
    try:
        return {
            (str(split), str(language), str(family)): int(count)
            for split, language, family, count in connection.execute(
                "SELECT split,language,family,COUNT(*) FROM tasks "
                "GROUP BY split,language,family"
            )
        }
    finally:
        connection.close()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--instruction", type=Path, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--replay-mode",
        choices=("all", "stratified"),
        default="all",
    )
    parser.add_argument("--replay-per-cell", type=int, default=256)
    arguments = parser.parse_args()
    if arguments.replay_mode == "stratified" and (
        not 8 <= arguments.replay_per_cell <= 2048
        or arguments.replay_per_cell % 8
    ):
        parser.error("stratified replay-per-cell must be 8..2048 and divisible by 8")

    replay_path = arguments.replay.resolve()
    instruction_path = arguments.instruction.resolve()
    destination = arguments.output.resolve()
    if destination.exists():
        raise ValueError("v10 instruction replay curriculum already exists")

    binding = NativeTokenizerBinding.load()
    replay_audit = audit_task_database(binding, replay_path)
    instruction_audit = audit_task_database(binding, instruction_path)
    expected_replay_identity = (
        EXPECTED_REPLAY_IDENTITY
        if arguments.replay_mode == "all"
        else EXPECTED_CLEAN_REPLAY_IDENTITY
    )
    if replay_audit.database_identity_sha256 != expected_replay_identity:
        raise ValueError("v10 replay source identity does not match its selection mode")
    if instruction_audit.database_identity_sha256 != EXPECTED_INSTRUCTION_IDENTITY:
        raise ValueError("v10 instruction source identity is not the audited curriculum")

    replay_metadata = _metadata(replay_path)
    instruction_metadata = _metadata(instruction_path)
    if (
        replay_metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or instruction_metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or replay_metadata.get("packing_complete") != "true"
        or instruction_metadata.get("packing_complete") != "true"
    ):
        raise ValueError("v10 source curriculum metadata is invalid")

    replay_counts = _cell_counts(replay_path)
    instruction_counts = _cell_counts(instruction_path)
    expected_instruction_counts = {
        ("train", "python", INSTRUCTION_FAMILY): 4096,
        ("validation", "python", INSTRUCTION_FAMILY): 256,
        ("test", "python", INSTRUCTION_FAMILY): 256,
    }
    if instruction_counts != expected_instruction_counts:
        raise ValueError("v10 instruction source cell counts are invalid")
    replay_train_cells = sum(split == "train" for split, _, _ in replay_counts)
    if replay_train_cells < 1:
        raise ValueError("v10 replay source contains no training cells")
    if arguments.replay_mode == "stratified":
        insufficient = {
            f"{language}|{family}": count
            for (split, language, family), count in replay_counts.items()
            if split == "train" and count < arguments.replay_per_cell
        }
        if insufficient:
            raise ValueError(
                "v10 replay source has insufficient stratified cells: "
                + json.dumps(insufficient, sort_keys=True)
            )

    instruction_multiplier = float(replay_train_cells)
    multiplier_key = f"train|python|{INSTRUCTION_FAMILY}"
    multipliers_json = _canonical_json({multiplier_key: instruction_multiplier})
    policy = {
        "schema_version": CURRICULUM_SCHEMA,
        "replay_source_database_identity_sha256": replay_audit.database_identity_sha256,
        "instruction_source_database_identity_sha256": (
            instruction_audit.database_identity_sha256
        ),
        "row_policy": (
            "copy every replay and instruction row byte-for-byte"
            if arguments.replay_mode == "all"
            else (
                "copy all heldout rows; select equal train rows from eight target-length "
                "quantiles per language-family cell; copy all instruction rows"
            )
        ),
        "replay_mode": arguments.replay_mode,
        "replay_per_cell": (
            arguments.replay_per_cell
            if arguments.replay_mode == "stratified"
            else None
        ),
        "heldout_policy": "preserve source validation and test rows and document splits",
        "training_weight_policy": (
            "equalize source cells, multiply instruction cell by replay cell count, "
            "then normalize mean training weight to one"
        ),
        "training_cell_weight_multipliers": {multiplier_key: instruction_multiplier},
    }
    policy_sha256 = hashlib.sha256(
        _canonical_json(policy).encode("utf-8")
    ).hexdigest()
    output_metadata = dict(replay_metadata)
    output_metadata.update(
        {
            "curriculum_schema_version": CURRICULUM_SCHEMA,
            "curriculum_policy_sha256": policy_sha256,
            "instruction_replay_policy_sha256": policy_sha256,
            "replay_source_database_identity_sha256": (
                replay_audit.database_identity_sha256
            ),
            "instruction_source_database_identity_sha256": (
                instruction_audit.database_identity_sha256
            ),
            "instruction_replay_target_train_weight_fraction": "0.5",
            "instruction_replay_replay_mode": arguments.replay_mode,
            "instruction_replay_replay_per_cell": (
                str(arguments.replay_per_cell)
                if arguments.replay_mode == "stratified"
                else "all"
            ),
            CELL_WEIGHT_MULTIPLIERS_METADATA_KEY: multipliers_json,
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
        output.execute("ATTACH DATABASE ? AS replay", (str(replay_path),))
        output.execute("ATTACH DATABASE ? AS instruction", (str(instruction_path),))
        if arguments.replay_mode == "all":
            output.execute("INSERT INTO tasks SELECT * FROM replay.tasks")
            selected_replay_counts = dict(replay_counts)
        else:
            output.execute(
                "INSERT INTO tasks SELECT * FROM replay.tasks "
                "WHERE split IN ('validation','test')"
            )
            per_quantile = arguments.replay_per_cell // 8
            output.execute(
                """WITH bucketed AS (
                       SELECT t.*,
                              NTILE(8) OVER (
                                  PARTITION BY language,family
                                  ORDER BY token_count-loss_start_index,example_id
                              ) AS target_quantile
                       FROM replay.tasks AS t WHERE split='train'
                   ), ranked AS (
                       SELECT b.*,
                              ROW_NUMBER() OVER (
                                  PARTITION BY language,family,target_quantile
                                  ORDER BY example_id
                              ) AS quantile_rank
                       FROM bucketed AS b
                   )
                   INSERT INTO tasks(
                       example_id,document_id,language,family,record_index,split,
                       input_ids_zlib,token_count,loss_start_index,
                       semantic_spec_sha256,source_sha256
                   )
                   SELECT example_id,document_id,language,family,record_index,split,
                          input_ids_zlib,token_count,loss_start_index,
                          semantic_spec_sha256,source_sha256
                   FROM ranked WHERE quantile_rank<=?""",
                (per_quantile,),
            )
            selected_replay_counts = {
                cell: (
                    arguments.replay_per_cell if cell[0] == "train" else count
                )
                for cell, count in replay_counts.items()
            }
        output.execute("INSERT INTO tasks SELECT * FROM instruction.tasks")
        output.commit()
        observed = {
            (str(split), str(language), str(family)): int(count)
            for split, language, family, count in output.execute(
                "SELECT split,language,family,COUNT(*) FROM tasks "
                "GROUP BY split,language,family"
            )
        }
        expected = dict(selected_replay_counts)
        expected.update(instruction_counts)
        if observed != expected:
            raise ValueError("v10 merged curriculum cell counts are invalid")
    except BaseException:
        output.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        output.close()

    try:
        result = audit_task_database(binding, temporary)
        dataset = ResumableTaskDataset(binding, temporary)
        instruction_weight = 0.0
        replay_weight = 0.0
        for row in dataset.iter_examples(split="train"):
            weight = float(row["sampling_weight"])
            if row["family"] == INSTRUCTION_FAMILY:
                instruction_weight += weight
            else:
                replay_weight += weight
        total_weight = instruction_weight + replay_weight
        instruction_fraction = instruction_weight / total_weight
        if not math.isclose(instruction_fraction, 0.5, abs_tol=1e-12):
            raise ValueError("v10 instruction training weight is not one half")
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.replace(destination)

    payload = asdict(result)
    payload.update(
        {
            "database_path": str(destination),
            "curriculum_schema_version": CURRICULUM_SCHEMA,
            "instruction_replay_policy_sha256": policy_sha256,
            "replay_source_database_identity_sha256": (
                replay_audit.database_identity_sha256
            ),
            "instruction_source_database_identity_sha256": (
                instruction_audit.database_identity_sha256
            ),
            "replay_train_cells": replay_train_cells,
            "replay_mode": arguments.replay_mode,
            "replay_per_cell": (
                arguments.replay_per_cell
                if arguments.replay_mode == "stratified"
                else None
            ),
            "instruction_train_multiplier": instruction_multiplier,
            "instruction_train_weight_fraction": instruction_fraction,
            "replay_train_weight_fraction": replay_weight / total_weight,
        }
    )
    if arguments.report is not None:
        _atomic_json(arguments.report.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
