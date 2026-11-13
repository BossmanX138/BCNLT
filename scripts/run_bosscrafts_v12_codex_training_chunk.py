"""Run one guarded V12 CodeX plus BCNLT replay training chunk from V11."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.balanced_curriculum import TASK_FAMILIES
from bosscrafts_model.bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from bosscrafts_model.codex_code_curriculum import INSTRUCTION_FAMILY
from bosscrafts_model.resumable_task_packing import (
    CELL_WEIGHT_MULTIPLIERS_METADATA_KEY,
    SALTED_SOURCE_EXAMPLE_ID_POLICY,
    TASK_DATABASE_SCHEMA,
    _sha256_file,
)


DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-codex-2m-bcnlt-replay-weighted-v12"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v12-codex-replay"
    / "checkpoint.pt"
)
DEFAULT_PARENT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-instruction-replay-balanced-v11"
    / "tasks.sqlite"
)
DEFAULT_PARENT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v11-instruction-mixed"
    / "checkpoint.pt"
)

EXPECTED_DATABASE_IDENTITY = "390a9d450f4e005051e4dccefa1287b42d6c51f8f5fed0f7ec06541b09faf03f"
EXPECTED_DATABASE_SHA256 = "dda3519750297cd7c62641ad4c7e74d72ac7720fc36b0e71c20f6bcb754c16a8"
EXPECTED_POLICY_SHA256 = "943bddc8c82cf5c8d72ee8e255325e2158b837de491504bd1817e4cf1c7a5ea9"
EXPECTED_SOURCE_IDENTITY_MAP_SHA256 = "529220184476abbf76862999e129e8fba9aa215cf0bf6e0f96600133475c3b00"
EXPECTED_CODEX_SOURCE_IDENTITY = "249c068873764d4594427ac06bf840a1729545a2f83a416c8706a38d916783c9"
EXPECTED_REPLAY_SOURCE_IDENTITY = "626186864cf4444db997bea5bd8ea4a18f18b5042509033a0d6939d0bd35afd2"
EXPECTED_MODEL_FINGERPRINT = "383bc3acb9bea01ed9a40fb6293a5ac81ee7ffb27ffbc4fdcfeafd96358b3bca"
PARENT_CHECKPOINT_SHA256 = "2ae40b745b07103bdd6165745d58330f634f239953caeb6cec2e814ad5119495"
PARENT_DATABASE_IDENTITY_SHA256 = "55d0f66e477dbc78b1d354fa98cacf609c1ed5ff83977e93e2a52c62bc3f0812"
PROMOTION_PLAN = (
    PROJECT_ROOT / "artifacts" / "plans" / "bosscrafts-engineering-v12-promotion-plan.json"
)
PROMOTION_PLAN_SHA256 = "2eaec8974122a2773eef840debd622ebf676ed1fb665cb511cc513550dbe2946"

REPLAY_PER_CELL = 512
REPLAY_TRAIN_CELLS = 42
CODEX_TRAIN_EXAMPLES = 121_868
REPLAY_TRAIN_EXAMPLES = REPLAY_PER_CELL * REPLAY_TRAIN_CELLS
TRAIN_EXAMPLES = CODEX_TRAIN_EXAMPLES + REPLAY_TRAIN_EXAMPLES
GRADIENT_ACCUMULATION = 8
PLANNED_OPTIMIZER_STEPS = math.ceil(TRAIN_EXAMPLES / GRADIENT_ACCUMULATION)
LEARNING_RATE = 0.00002
WARMUP_RATIO = 0.01
MINIMUM_LEARNING_RATE_RATIO = 0.1
MAXIMUM_STEPS_PER_CHUNK = 100
CODEX_WEIGHT_FRACTION = 0.80
REPLAY_WEIGHT_FRACTION = 0.20


def expected_initialization() -> dict[str, object]:
    return {
        "type": "verified_completed_checkpoint",
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "task_database_identity_sha256": PARENT_DATABASE_IDENTITY_SHA256,
    }


def expected_training_plan_sha256() -> str:
    warmup_steps = min(
        PLANNED_OPTIMIZER_STEPS - 1,
        int(PLANNED_OPTIMIZER_STEPS * WARMUP_RATIO),
    )
    plan = {
        "dtype": "bfloat16",
        "epochs": 1,
        "full_model": True,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "gradient_checkpointing": True,
        "optimizer": {
            "amsgrad": False,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "learning_rate": LEARNING_RATE,
            "name": "AdamW",
            "weight_decay": 0.01,
        },
        "schedule": {
            "minimum_learning_rate_ratio": MINIMUM_LEARNING_RATE_RATIO,
            "name": "linear_warmup_cosine_decay",
            "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
            "warmup_ratio": WARMUP_RATIO,
            "warmup_steps": warmup_steps,
        },
        "schema_version": "bosscrafts-training-plan-v2",
        "seed": 7,
        "split": "train",
        "initialization": expected_initialization(),
    }
    return hashlib.sha256(
        json.dumps(plan, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _verify_database(database: Path) -> None:
    if not database.is_file():
        raise ValueError("V12 CodeX replay curriculum does not exist")
    if _sha256_file(database) != EXPECTED_DATABASE_SHA256:
        raise ValueError("V12 CodeX replay curriculum file hash mismatch")
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        counts = {
            (str(split), str(language), str(family)): int(count)
            for split, language, family, count in connection.execute(
                "SELECT split,language,family,COUNT(*) FROM tasks GROUP BY split,language,family"
            )
        }
        split_counts = dict(
            connection.execute("SELECT split,COUNT(*) FROM tasks GROUP BY split")
        )
        family_split_counts = {
            (str(split), str(family)): int(count)
            for split, family, count in connection.execute(
                "SELECT split,family,COUNT(*) FROM tasks GROUP BY split,family"
            )
        }
        maximum_sequence = int(
            connection.execute("SELECT MAX(token_count) FROM tasks").fetchone()[0]
        )
    finally:
        connection.close()
    try:
        multipliers = json.loads(metadata[CELL_WEIGHT_MULTIPLIERS_METADATA_KEY])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("V12 training cell multipliers are invalid") from error
    if (
        metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or metadata.get("packing_complete") != "true"
        or metadata.get("curriculum_schema_version")
        != "bosscrafts-codex-bcnlt-replay-curriculum-v1"
        or metadata.get("maximum_sequence_tokens") != "2048"
        or metadata.get("codex_replay_policy_sha256") != EXPECTED_POLICY_SHA256
        or metadata.get("codex_source_database_identity_sha256")
        != EXPECTED_CODEX_SOURCE_IDENTITY
        or metadata.get("replay_source_database_identity_sha256")
        != EXPECTED_REPLAY_SOURCE_IDENTITY
        or metadata.get("replay_per_train_cell") != str(REPLAY_PER_CELL)
        or metadata.get("codex_replay_order_audit_schema")
        != "bosscrafts-codex-bcnlt-order-audit-v1"
        or metadata.get("training_example_id_policy") != SALTED_SOURCE_EXAMPLE_ID_POLICY
        or metadata.get("source_example_identity_map_sha256")
        != EXPECTED_SOURCE_IDENTITY_MAP_SHA256
        or maximum_sequence > 2048
    ):
        raise ValueError("V12 CodeX replay curriculum metadata is invalid")
    if split_counts != {"test": 7971, "train": TRAIN_EXAMPLES, "validation": 7890}:
        raise ValueError("V12 CodeX replay split counts are invalid")
    expected_instruction = {
        ("train", INSTRUCTION_FAMILY): CODEX_TRAIN_EXAMPLES,
        ("validation", INSTRUCTION_FAMILY): 2514,
        ("test", INSTRUCTION_FAMILY): 2595,
    }
    if any(family_split_counts.get(cell) != count for cell, count in expected_instruction.items()):
        raise ValueError("V12 CodeX instruction counts are invalid")
    languages = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))
    for language in languages:
        for family in TASK_FAMILIES:
            if (
                counts.get(("train", language, family)) != REPLAY_PER_CELL
                or counts.get(("validation", language, family)) != 128
                or counts.get(("test", language, family)) != 128
            ):
                raise ValueError("V12 replay cell counts are invalid")
    observed_train_cells = {
        f"{split}|{language}|{family}": count
        for (split, language, family), count in counts.items()
        if split == "train"
    }
    if set(multipliers) != set(observed_train_cells):
        raise ValueError("V12 training cell multipliers do not cover the observed cells")
    instruction_fraction = 0.0
    replay_fraction = 0.0
    for key, count in observed_train_cells.items():
        multiplier = float(multipliers[key])
        if key.endswith(f"|{INSTRUCTION_FAMILY}"):
            expected = CODEX_WEIGHT_FRACTION * count / CODEX_TRAIN_EXAMPLES
            instruction_fraction += multiplier
        else:
            expected = REPLAY_WEIGHT_FRACTION / REPLAY_TRAIN_CELLS
            replay_fraction += multiplier
        if not math.isclose(multiplier, expected, abs_tol=1e-15):
            raise ValueError("V12 training cell multiplier value is invalid")
    if not (
        math.isclose(instruction_fraction, CODEX_WEIGHT_FRACTION, abs_tol=1e-12)
        and math.isclose(replay_fraction, REPLAY_WEIGHT_FRACTION, abs_tol=1e-12)
    ):
        raise ValueError("V12 aggregate training weights are invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--parent-database", type=Path, default=DEFAULT_PARENT_DATABASE)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT)
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    database = arguments.database.resolve()
    checkpoint = arguments.checkpoint.resolve()
    parent_database = arguments.parent_database.resolve()
    parent_checkpoint = arguments.parent_checkpoint.resolve()
    status_output = (
        arguments.status_output.resolve()
        if arguments.status_output is not None
        else checkpoint.with_name("training-progress.json")
    )
    _verify_database(database)
    if not PROMOTION_PLAN.is_file() or _sha256_file(PROMOTION_PLAN) != PROMOTION_PLAN_SHA256:
        raise ValueError("frozen V12 promotion plan is missing or changed")
    if not parent_database.is_file() or not parent_checkpoint.is_file():
        raise ValueError("verified V11 remediation-parent artifacts are required")
    if _sha256_file(parent_checkpoint) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("V11 remediation-parent checkpoint hash mismatch")
    resume = checkpoint.is_file()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_bounded_training_trial.py"),
        "--database", str(database),
        "--checkpoint", str(checkpoint),
        "--status-output", str(status_output),
        "--device", "cuda",
        "--dtype", "bfloat16",
        "--full-model",
        "--gradient-checkpointing",
        "--gradient-accumulation", str(GRADIENT_ACCUMULATION),
        "--learning-rate", str(LEARNING_RATE),
        "--warmup-ratio", str(WARMUP_RATIO),
        "--minimum-learning-rate-ratio", str(MINIMUM_LEARNING_RATE_RATIO),
        "--max-steps", str(MAXIMUM_STEPS_PER_CHUNK),
        "--epochs", "1",
        "--initialization-checkpoint-sha256", PARENT_CHECKPOINT_SHA256,
        "--initialization-database-identity-sha256", PARENT_DATABASE_IDENTITY_SHA256,
    ]
    if resume:
        command.append("--resume")
    else:
        command.extend(
            [
                "--initial-model-checkpoint", str(parent_checkpoint),
                "--initial-checkpoint-database", str(parent_database),
            ]
        )
    contract = {
        "schema_version": "bosscrafts-v12-codex-training-chunk-contract-v1",
        "database": str(database),
        "database_identity_sha256": EXPECTED_DATABASE_IDENTITY,
        "database_sha256": EXPECTED_DATABASE_SHA256,
        "checkpoint": str(checkpoint),
        "parent_database": str(parent_database),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "expected_training_plan_sha256": expected_training_plan_sha256(),
        "promotion_plan": str(PROMOTION_PLAN),
        "promotion_plan_sha256": PROMOTION_PLAN_SHA256,
        "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
        "train_examples": TRAIN_EXAMPLES,
        "codex_train_examples": CODEX_TRAIN_EXAMPLES,
        "replay_train_examples": REPLAY_TRAIN_EXAMPLES,
        "codex_weight_fraction": CODEX_WEIGHT_FRACTION,
        "replay_weight_fraction": REPLAY_WEIGHT_FRACTION,
        "device": "cuda",
        "dtype": "bfloat16",
        "epochs": 1,
        "full_model": True,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "gradient_checkpointing": True,
        "learning_rate": LEARNING_RATE,
        "maximum_optimizer_steps_this_chunk": MAXIMUM_STEPS_PER_CHUNK,
        "minimum_learning_rate_ratio": MINIMUM_LEARNING_RATE_RATIO,
        "resume": resume,
        "status_output": str(status_output),
        "warmup_ratio": WARMUP_RATIO,
        "command": command,
    }
    if arguments.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
