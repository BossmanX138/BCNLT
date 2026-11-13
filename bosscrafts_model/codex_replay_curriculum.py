"""Build a provenance-preserving CodeX plus BCNLT replay curriculum."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from .codex_code_curriculum import INSTRUCTION_FAMILY
from .resumable_task_packing import (
    CELL_WEIGHT_MULTIPLIERS_METADATA_KEY,
    SALTED_SOURCE_EXAMPLE_ID_POLICY,
    TASK_DATABASE_SCHEMA,
    ResumableTaskDataset,
    _canonical_json,
    _sha256_file,
    audit_task_database,
    salted_source_example_id,
)
from .tokenizer import NativeTokenizerBinding


CURRICULUM_SCHEMA = "bosscrafts-codex-bcnlt-replay-curriculum-v1"
ORDER_AUDIT_SCHEMA = "bosscrafts-codex-bcnlt-order-audit-v1"
EXAMPLE_ID_POLICY = SALTED_SOURCE_EXAMPLE_ID_POLICY


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
        raise ValueError("mixed curriculum source identity map is empty")
    return digest.hexdigest()


def _training_multipliers(
    codex_counts: dict[tuple[str, str, str], int],
    replay_counts: dict[tuple[str, str, str], int],
    *,
    codex_weight_fraction: float,
) -> dict[str, float]:
    codex_train = {cell: count for cell, count in codex_counts.items() if cell[0] == "train"}
    replay_train = {cell: count for cell, count in replay_counts.items() if cell[0] == "train"}
    if not codex_train or not replay_train:
        raise ValueError("both curriculum sources must contain training cells")
    total_codex = sum(codex_train.values())
    replay_fraction = 1.0 - codex_weight_fraction
    multipliers: dict[str, float] = {}
    for (split, language, family), count in sorted(codex_train.items()):
        multipliers[f"{split}|{language}|{family}"] = (
            codex_weight_fraction * count / total_codex
        )
    per_replay_cell = replay_fraction / len(replay_train)
    for split, language, family in sorted(replay_train):
        key = f"{split}|{language}|{family}"
        if key in multipliers:
            raise ValueError("CodeX and replay training cells overlap")
        multipliers[key] = per_replay_cell
    if not math.isclose(sum(multipliers.values()), 1.0, abs_tol=1e-12):
        raise ValueError("training cell multiplier fractions do not sum to one")
    return multipliers


def build_codex_replay_curriculum(
    binding: NativeTokenizerBinding,
    *,
    codex_database: str | Path,
    replay_database: str | Path,
    destination: str | Path,
    replay_per_train_cell: int = 512,
    codex_weight_fraction: float = 0.80,
    order_window_size: int = 800,
    minimum_replay_order_fraction: float = 0.10,
    maximum_replay_order_fraction: float = 0.20,
    expected_codex_identity: str | None = None,
    expected_replay_identity: str | None = None,
) -> dict[str, object]:
    """Build and audit a deterministic mixed curriculum.

    All CodeX rows and all replay heldout rows are retained. Training replay is
    selected uniformly over target-length quantiles and source-ID intervals.
    Explicit cell multipliers make the aggregate optimization weight independent
    of sparse language cells.
    """
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if (
        replay_per_train_cell < 8
        or replay_per_train_cell % 8
        or order_window_size < 8
    ):
        raise ValueError("replay count must be >=8 and divisible by 8; window must be >=8")
    if not 0.5 < codex_weight_fraction < 1.0:
        raise ValueError("CodeX weight fraction must be between one half and one")
    if not 0.0 <= minimum_replay_order_fraction < maximum_replay_order_fraction <= 1.0:
        raise ValueError("replay order fraction bounds are invalid")

    codex_path = Path(codex_database).resolve()
    replay_path = Path(replay_database).resolve()
    output_path = Path(destination).resolve()
    if output_path.exists():
        raise ValueError("mixed curriculum output database already exists")
    codex_audit = audit_task_database(binding, codex_path)
    replay_audit = audit_task_database(binding, replay_path)
    if expected_codex_identity and codex_audit.database_identity_sha256 != expected_codex_identity:
        raise ValueError("CodeX source identity is not the frozen phase-one curriculum")
    if expected_replay_identity and replay_audit.database_identity_sha256 != expected_replay_identity:
        raise ValueError("replay source identity is not the frozen clean curriculum")

    codex_metadata = _metadata(codex_path)
    replay_metadata = _metadata(replay_path)
    if (
        codex_metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or replay_metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or codex_metadata.get("packing_complete") != "true"
        or replay_metadata.get("packing_complete") != "true"
        or codex_metadata.get("parser_runtime_identity_sha256")
        != PARSER_RUNTIME_IDENTITY_SHA256
        or replay_metadata.get("parser_runtime_identity_sha256")
        != PARSER_RUNTIME_IDENTITY_SHA256
        or codex_metadata.get("maximum_sequence_tokens")
        != replay_metadata.get("maximum_sequence_tokens")
    ):
        raise ValueError("mixed curriculum source metadata is incompatible")

    codex_counts = _cell_counts(codex_path)
    replay_source_counts = _cell_counts(replay_path)
    if any(cell[2] != INSTRUCTION_FAMILY for cell in codex_counts):
        raise ValueError("CodeX source contains a non-instruction family")
    insufficient = {
        f"{language}|{family}": count
        for (split, language, family), count in replay_source_counts.items()
        if split == "train" and count < replay_per_train_cell
    }
    if insufficient:
        raise ValueError("replay source has insufficient training cells: " + json.dumps(insufficient, sort_keys=True))
    selected_replay_counts = {
        cell: replay_per_train_cell if cell[0] == "train" else count
        for cell, count in replay_source_counts.items()
    }
    multipliers = _training_multipliers(
        codex_counts,
        selected_replay_counts,
        codex_weight_fraction=codex_weight_fraction,
    )
    multiplier_json = _canonical_json(multipliers)
    policy = {
        "schema_version": CURRICULUM_SCHEMA,
        "codex_source_database_identity_sha256": codex_audit.database_identity_sha256,
        "replay_source_database_identity_sha256": replay_audit.database_identity_sha256,
        "row_policy": (
            "copy all CodeX rows and all replay validation/test rows; for every replay "
            "training cell select one row from each equal source-ID interval within "
            "each of eight target-length quantiles"
        ),
        "example_id_policy": EXAMPLE_ID_POLICY,
        "source_identity_policy": (
            "preserve original source example IDs and order by source-kind-salted SHA-256 IDs"
        ),
        "replay_per_train_cell": replay_per_train_cell,
        "codex_weight_fraction": codex_weight_fraction,
        "replay_weight_fraction": 1.0 - codex_weight_fraction,
        "training_cell_weight_multipliers": multipliers,
        "heldout_policy": "preserve every source validation and test row and document split",
        "order_window_size": order_window_size,
        "minimum_replay_order_fraction": minimum_replay_order_fraction,
        "maximum_replay_order_fraction": maximum_replay_order_fraction,
    }
    policy_sha256 = hashlib.sha256(_canonical_json(policy).encode("utf-8")).hexdigest()
    output_metadata = dict(codex_metadata)
    output_metadata.update(
        {
            "curriculum_schema_version": CURRICULUM_SCHEMA,
            "curriculum_policy_sha256": policy_sha256,
            "codex_replay_policy_sha256": policy_sha256,
            "codex_source_database_identity_sha256": codex_audit.database_identity_sha256,
            "replay_source_database_identity_sha256": replay_audit.database_identity_sha256,
            "codex_target_train_weight_fraction": format(codex_weight_fraction, ".17g"),
            "replay_target_train_weight_fraction": format(1.0 - codex_weight_fraction, ".17g"),
            "replay_per_train_cell": str(replay_per_train_cell),
            "codex_replay_order_audit_schema": ORDER_AUDIT_SCHEMA,
            CELL_WEIGHT_MULTIPLIERS_METADATA_KEY: multiplier_json,
            "packing_complete": "true",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    output = sqlite3.connect(temporary)
    try:
        output.create_function("mixed_example_id", 2, salted_source_example_id, deterministic=True)
        output.execute("PRAGMA journal_mode=DELETE")
        output.execute("PRAGMA synchronous=FULL")
        output.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        output.execute(
            """CREATE TABLE tasks_source_ids(
                   source_kind TEXT NOT NULL CHECK(source_kind IN ('instruction','replay')),
                   source_example_id TEXT NOT NULL,
                   document_id TEXT NOT NULL,language TEXT NOT NULL,family TEXT NOT NULL,
                   record_index INTEGER NOT NULL,split TEXT NOT NULL,
                   input_ids_zlib BLOB NOT NULL,token_count INTEGER NOT NULL,
                   loss_start_index INTEGER NOT NULL,semantic_spec_sha256 TEXT NOT NULL,
                   parser_runtime_identity_sha256 TEXT NOT NULL,
                   source_sha256 TEXT NOT NULL,
                   PRIMARY KEY(source_kind,source_example_id)
               )"""
        )
        output.executemany("INSERT INTO metadata VALUES(?,?)", sorted(output_metadata.items()))
        output.execute("ATTACH DATABASE ? AS codex", (str(codex_path),))
        output.execute("ATTACH DATABASE ? AS replay", (str(replay_path),))
        output.execute(
            """INSERT INTO tasks_source_ids
               SELECT 'instruction',example_id,document_id,language,family,record_index,
                      split,input_ids_zlib,token_count,loss_start_index,
                      semantic_spec_sha256,parser_runtime_identity_sha256,
                      source_sha256 FROM codex.tasks"""
        )
        output.execute(
            """INSERT INTO tasks_source_ids
               SELECT 'replay',example_id,document_id,language,family,record_index,
                      split,input_ids_zlib,token_count,loss_start_index,
                      semantic_spec_sha256,parser_runtime_identity_sha256,source_sha256
               FROM replay.tasks WHERE split IN ('validation','test')"""
        )
        per_quantile = replay_per_train_cell // 8
        output.execute(
            """WITH bucketed AS (
                   SELECT t.*,
                          NTILE(8) OVER(
                              PARTITION BY language,family
                              ORDER BY token_count-loss_start_index,example_id
                          ) AS target_quantile
                   FROM replay.tasks AS t WHERE split='train'
               ), spread AS (
                   SELECT b.*,
                          NTILE(?) OVER(
                              PARTITION BY language,family,target_quantile
                              ORDER BY example_id
                          ) AS hash_interval
                   FROM bucketed AS b
               ), ranked AS (
                   SELECT s.*,
                          ROW_NUMBER() OVER(
                              PARTITION BY language,family,target_quantile,hash_interval
                              ORDER BY example_id
                          ) AS interval_rank
                   FROM spread AS s
               )
               INSERT INTO tasks_source_ids
               SELECT 'replay',example_id,document_id,language,family,record_index,
                      split,input_ids_zlib,token_count,loss_start_index,
                      semantic_spec_sha256,parser_runtime_identity_sha256,source_sha256
               FROM ranked WHERE interval_rank=1""",
            (per_quantile,),
        )
        observed_source_counts = {
            (str(kind), str(split), str(language), str(family)): int(count)
            for kind, split, language, family, count in output.execute(
                "SELECT source_kind,split,language,family,COUNT(*) FROM tasks_source_ids "
                "GROUP BY source_kind,split,language,family"
            )
        }
        expected_source_counts = {
            ("instruction", *cell): count for cell, count in codex_counts.items()
        }
        expected_source_counts.update(
            {("replay", *cell): count for cell, count in selected_replay_counts.items()}
        )
        if observed_source_counts != expected_source_counts:
            raise ValueError("mixed curriculum source cell counts are invalid")

        output.execute(
            """CREATE TABLE tasks(
                   example_id TEXT PRIMARY KEY,document_id TEXT NOT NULL,
                   language TEXT NOT NULL,family TEXT NOT NULL,
                   record_index INTEGER NOT NULL,split TEXT NOT NULL,
                   input_ids_zlib BLOB NOT NULL,token_count INTEGER NOT NULL,
                   loss_start_index INTEGER NOT NULL,
                   semantic_spec_sha256 TEXT NOT NULL,
                   parser_runtime_identity_sha256 TEXT NOT NULL,
                   source_sha256 TEXT NOT NULL
               )"""
        )
        output.execute(
            """CREATE TABLE source_example_identity(
                   example_id TEXT PRIMARY KEY,source_example_id TEXT NOT NULL,
                   source_kind TEXT NOT NULL CHECK(source_kind IN ('instruction','replay')),
                   UNIQUE(source_kind,source_example_id)
               )"""
        )
        output.execute(
            """INSERT INTO tasks
               SELECT mixed_example_id(source_kind,source_example_id),document_id,
                      language,family,record_index,split,input_ids_zlib,token_count,
                      loss_start_index,semantic_spec_sha256,
                      parser_runtime_identity_sha256,source_sha256
               FROM tasks_source_ids"""
        )
        output.execute(
            """INSERT INTO source_example_identity
               SELECT mixed_example_id(source_kind,source_example_id),source_example_id,
                      source_kind FROM tasks_source_ids"""
        )
        output.execute("DROP TABLE tasks_source_ids")
        output.execute("CREATE INDEX tasks_cell ON tasks(split,language,family)")

        output.execute(
            """CREATE TABLE codex_provenance(
                   example_id TEXT PRIMARY KEY,content_sha256 TEXT NOT NULL UNIQUE,
                   source_parquet TEXT NOT NULL,source_parquet_sha256 TEXT NOT NULL,
                   source_row_index INTEGER NOT NULL,input_sha256 TEXT NOT NULL,
                   full_output_sha256 TEXT NOT NULL,final_code_sha256 TEXT NOT NULL,
                   fence_label TEXT NOT NULL,license_spdx TEXT NOT NULL
               )"""
        )
        output.execute(
            """INSERT INTO codex_provenance
               SELECT mixed_example_id('instruction',example_id),content_sha256,
                      source_parquet,source_parquet_sha256,source_row_index,input_sha256,
                      full_output_sha256,final_code_sha256,fence_label,license_spdx
               FROM codex.codex_provenance"""
        )
        output.execute(
            "CREATE TABLE codex_source_shards AS SELECT * FROM codex.source_shards"
        )
        output.execute(
            "CREATE TABLE codex_rejection_counts AS SELECT * FROM codex.rejection_counts"
        )
        source_identity_sha256 = _source_identity_map_sha256(output)
        output.execute(
            "INSERT INTO metadata VALUES(?,?)",
            ("source_example_identity_map_sha256", source_identity_sha256),
        )
        output.execute(
            "INSERT INTO metadata VALUES(?,?)",
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
        weighted: Counter[str] = Counter()
        order_kinds: list[str] = []
        for row in dataset.iter_examples(split="train"):
            kind = "instruction" if row["family"] == INSTRUCTION_FAMILY else "replay"
            weighted[kind] += float(row["sampling_weight"])
            order_kinds.append(kind)
        total_weight = weighted["instruction"] + weighted["replay"]
        observed_codex_weight_fraction = weighted["instruction"] / total_weight
        if not math.isclose(observed_codex_weight_fraction, codex_weight_fraction, abs_tol=1e-12):
            raise ValueError("observed CodeX training weight fraction is invalid")
        windows: list[dict[str, object]] = []
        for offset in range(0, len(order_kinds), order_window_size):
            kinds = order_kinds[offset : offset + order_window_size]
            replay_count = kinds.count("replay")
            windows.append(
                {
                    "window_index": offset // order_window_size + 1,
                    "example_count": len(kinds),
                    "replay_count": replay_count,
                    "replay_fraction": replay_count / len(kinds),
                }
            )
        full_windows = [window for window in windows if window["example_count"] == order_window_size]
        if any(
            not minimum_replay_order_fraction
            <= float(window["replay_fraction"])
            <= maximum_replay_order_fraction
            for window in full_windows
        ):
            raise ValueError("mixed curriculum is not sufficiently interleaved in full training windows")
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    temporary.replace(output_path)
    report = asdict(result)
    report.update(
        {
            "database_path": str(output_path),
            "database_file_sha256": _sha256_file(output_path),
            "curriculum_schema_version": CURRICULUM_SCHEMA,
            "curriculum_policy_sha256": policy_sha256,
            "codex_source_database_identity_sha256": codex_audit.database_identity_sha256,
            "replay_source_database_identity_sha256": replay_audit.database_identity_sha256,
            "replay_per_train_cell": replay_per_train_cell,
            "codex_train_weight_fraction": observed_codex_weight_fraction,
            "replay_train_weight_fraction": weighted["replay"] / total_weight,
            "raw_replay_train_fraction": order_kinds.count("replay") / len(order_kinds),
            "order_audit_schema": ORDER_AUDIT_SCHEMA,
            "order_window_size": order_window_size,
            "order_windows": windows,
            "source_example_identity_map_sha256": source_identity_sha256,
            "training_cell_weight_multipliers": multipliers,
        }
    )
    return report
