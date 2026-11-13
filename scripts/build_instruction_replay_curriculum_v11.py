"""Build a held-out-safe, order-mixed v11 instruction/replay curriculum."""
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
    SALTED_SOURCE_EXAMPLE_ID_POLICY,
    TASK_DATABASE_SCHEMA,
    ResumableTaskDataset,
    _canonical_json,
    audit_task_database,
    salted_source_example_id,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


CURRICULUM_SCHEMA = "bosscrafts-instruction-replay-curriculum-v2"
ORDER_AUDIT_SCHEMA = "bosscrafts-instruction-replay-order-audit-v1"
EXAMPLE_ID_POLICY = SALTED_SOURCE_EXAMPLE_ID_POLICY
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
    / "bcnlt-current-spec-65536-instruction-replay-balanced-v11"
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


def _v11_example_id(source_kind: str, source_example_id: str) -> str:
    return salted_source_example_id(source_kind, source_example_id)


def _source_identity_map_sha256(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    count = 0
    for example_id, source_example_id, source_kind in connection.execute(
        "SELECT example_id,source_example_id,source_kind "
        "FROM source_example_identity ORDER BY example_id"
    ):
        digest.update(
            _canonical_json(
                {
                    "example_id": str(example_id),
                    "source_example_id": str(source_example_id),
                    "source_kind": str(source_kind),
                }
            ).encode("utf-8")
        )
        count += 1
    if count < 1:
        raise ValueError("v11 source identity map is empty")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--instruction", type=Path, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--replay-mode",
        choices=("stratified",),
        default="stratified",
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
        raise ValueError("v11 instruction replay curriculum already exists")

    binding = NativeTokenizerBinding.load()
    replay_audit = audit_task_database(binding, replay_path)
    instruction_audit = audit_task_database(binding, instruction_path)
    expected_replay_identity = (
        EXPECTED_REPLAY_IDENTITY
        if arguments.replay_mode == "all"
        else EXPECTED_CLEAN_REPLAY_IDENTITY
    )
    if replay_audit.database_identity_sha256 != expected_replay_identity:
        raise ValueError("v11 replay source identity does not match its selection mode")
    if instruction_audit.database_identity_sha256 != EXPECTED_INSTRUCTION_IDENTITY:
        raise ValueError("v11 instruction source identity is not the audited curriculum")

    replay_metadata = _metadata(replay_path)
    instruction_metadata = _metadata(instruction_path)
    if (
        replay_metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or instruction_metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or replay_metadata.get("packing_complete") != "true"
        or instruction_metadata.get("packing_complete") != "true"
    ):
        raise ValueError("v11 source curriculum metadata is invalid")

    replay_counts = _cell_counts(replay_path)
    instruction_counts = _cell_counts(instruction_path)
    expected_instruction_counts = {
        ("train", "python", INSTRUCTION_FAMILY): 4096,
        ("validation", "python", INSTRUCTION_FAMILY): 256,
        ("test", "python", INSTRUCTION_FAMILY): 256,
    }
    if instruction_counts != expected_instruction_counts:
        raise ValueError("v11 instruction source cell counts are invalid")
    replay_train_cells = sum(split == "train" for split, _, _ in replay_counts)
    if replay_train_cells < 1:
        raise ValueError("v11 replay source contains no training cells")
    if arguments.replay_mode == "stratified":
        insufficient = {
            f"{language}|{family}": count
            for (split, language, family), count in replay_counts.items()
            if split == "train" and count < arguments.replay_per_cell
        }
        if insufficient:
            raise ValueError(
                "v11 replay source has insufficient stratified cells: "
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
                "copy all heldout rows; within each of eight target-length quantiles "
                "select one replay row from each equal SHA-ID interval; copy all "
                "instruction rows"
            )
        ),
        "example_id_policy": EXAMPLE_ID_POLICY,
        "source_identity_policy": (
            "preserve every original source example ID in source_example_identity; "
            "order tasks by a source-namespaced salted SHA-256 ID"
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
            "instruction_replay_order_audit_schema": ORDER_AUDIT_SCHEMA,
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
        output.create_function(
            "v11_example_id", 2, _v11_example_id, deterministic=True
        )
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
                   ), spread AS (
                       SELECT b.*,
                              NTILE(?) OVER (
                                  PARTITION BY language,family,target_quantile
                                  ORDER BY example_id
                              ) AS hash_interval
                       FROM bucketed AS b
                   ), ranked AS (
                       SELECT s.*,
                              ROW_NUMBER() OVER (
                                  PARTITION BY language,family,target_quantile,hash_interval
                                  ORDER BY example_id
                              ) AS interval_rank
                       FROM spread AS s
                   )
                   INSERT INTO tasks(
                       example_id,document_id,language,family,record_index,split,
                       input_ids_zlib,token_count,loss_start_index,
                       semantic_spec_sha256,source_sha256
                   )
                   SELECT example_id,document_id,language,family,record_index,split,
                          input_ids_zlib,token_count,loss_start_index,
                          semantic_spec_sha256,source_sha256
                   FROM ranked WHERE interval_rank=1""",
                (per_quantile,),
            )
            selected_replay_counts = {
                cell: (
                    arguments.replay_per_cell if cell[0] == "train" else count
                )
                for cell, count in replay_counts.items()
            }
        output.execute("INSERT INTO tasks SELECT * FROM instruction.tasks")
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
            raise ValueError("v11 merged curriculum cell counts are invalid")
        output.execute("ALTER TABLE tasks RENAME TO tasks_source_ids")
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
        output.execute(
            """CREATE TABLE source_example_identity (
                   example_id TEXT PRIMARY KEY,
                   source_example_id TEXT NOT NULL,
                   source_kind TEXT NOT NULL CHECK(source_kind IN ('replay','instruction')),
                   UNIQUE(source_kind,source_example_id)
               )"""
        )
        source_kind_sql = (
            "CASE WHEN family='instruction_to_code' "
            "THEN 'instruction' ELSE 'replay' END"
        )
        output.execute(
            f"""INSERT INTO tasks(
                       example_id,document_id,language,family,record_index,split,
                       input_ids_zlib,token_count,loss_start_index,
                       semantic_spec_sha256,source_sha256
                   )
                   SELECT v11_example_id({source_kind_sql},example_id),
                          document_id,language,family,record_index,split,
                          input_ids_zlib,token_count,loss_start_index,
                          semantic_spec_sha256,source_sha256
                   FROM tasks_source_ids"""
        )
        output.execute(
            f"""INSERT INTO source_example_identity(
                       example_id,source_example_id,source_kind
                   )
                   SELECT v11_example_id({source_kind_sql},example_id),
                          example_id,{source_kind_sql}
                   FROM tasks_source_ids"""
        )
        output.execute("DROP TABLE tasks_source_ids")
        output.execute("CREATE INDEX tasks_cell ON tasks(split,language,family)")
        source_identity_map_sha256 = _source_identity_map_sha256(output)
        output.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?)",
            ("source_example_identity_map_sha256", source_identity_map_sha256),
        )
        output.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?)",
            ("training_example_id_policy", EXAMPLE_ID_POLICY),
        )
        output.commit()
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
        ordered_families: list[str] = []
        for row in dataset.iter_examples(split="train"):
            weight = float(row["sampling_weight"])
            ordered_families.append(str(row["family"]))
            if row["family"] == INSTRUCTION_FAMILY:
                instruction_weight += weight
            else:
                replay_weight += weight
        total_weight = instruction_weight + replay_weight
        instruction_fraction = instruction_weight / total_weight
        if not math.isclose(instruction_fraction, 0.5, abs_tol=1e-12):
            raise ValueError("v11 instruction training weight is not one half")
        order_window_size = 800
        order_windows = []
        for offset in range(0, len(ordered_families), order_window_size):
            families = ordered_families[offset : offset + order_window_size]
            instruction_count = sum(
                family == INSTRUCTION_FAMILY for family in families
            )
            fraction = instruction_count / len(families)
            order_windows.append(
                {
                    "window_index": offset // order_window_size + 1,
                    "example_count": len(families),
                    "instruction_count": instruction_count,
                    "instruction_fraction": fraction,
                }
            )
        if any(
            not 0.20 <= float(window["instruction_fraction"]) <= 0.35
            for window in order_windows
        ):
            raise ValueError(
                "v11 curriculum is not mixed across training windows: "
                + json.dumps(order_windows, sort_keys=True)
            )
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
            "order_audit_schema": ORDER_AUDIT_SCHEMA,
            "order_window_size": order_window_size,
            "order_windows": order_windows,
            "example_id_policy": EXAMPLE_ID_POLICY,
            "source_example_identity_map_sha256": source_identity_map_sha256,
        }
    )
    if arguments.report is not None:
        _atomic_json(arguments.report.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
