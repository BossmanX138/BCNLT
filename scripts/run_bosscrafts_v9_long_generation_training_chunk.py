"""Run one guarded long-generation repair chunk from completed v8."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.balanced_curriculum import CURRICULUM_SCHEMA, TASK_FAMILIES
from bosscrafts_model.bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from bosscrafts_model.resumable_task_packing import TASK_DATABASE_SCHEMA

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-long-generation-repair-v9"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v9-long-generation"
    / "checkpoint.pt"
)
DEFAULT_PARENT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-deterministic-clean-v5"
    / "tasks.sqlite"
)
DEFAULT_PARENT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v8-consolidation"
    / "checkpoint.pt"
)
EXPECTED_DATABASE_IDENTITY = "677925e6500d03c546dfc198e911398d58e5791d9995b46f02d821919378127b"
EXPECTED_REPAIR_POLICY_SHA256 = "cb0aa102aaffbee3073db112d7735b39939653dccba8a4dc0d254fd4f0f61c4d"
EXPECTED_MODEL_FINGERPRINT = "383bc3acb9bea01ed9a40fb6293a5ac81ee7ffb27ffbc4fdcfeafd96358b3bca"
PARENT_CHECKPOINT_SHA256 = "1c367f2440415639ed1ce52e2fc85f2862e198e62f179095450674f7f734b519"
PARENT_DATABASE_IDENTITY_SHA256 = "626186864cf4444db997bea5bd8ea4a18f18b5042509033a0d6939d0bd35afd2"
TRAIN_PER_CELL = 512
MAXIMUM_TARGET_TOKENS = 512
TRAIN_EXAMPLES = 21_504
GRADIENT_ACCUMULATION = 8
PLANNED_OPTIMIZER_STEPS = math.ceil(TRAIN_EXAMPLES / GRADIENT_ACCUMULATION)
LEARNING_RATE = 0.00001
WARMUP_RATIO = 0.01
MINIMUM_LEARNING_RATE_RATIO = 0.2


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
        raise ValueError("v9 long-generation curriculum does not exist")
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        observed = {
            (str(split), str(language), str(family)): int(count)
            for split, language, family, count in connection.execute(
                "SELECT split,language,family,COUNT(*) FROM tasks "
                "GROUP BY split,language,family"
            )
        }
        train_count, maximum_target = connection.execute(
            "SELECT COUNT(*),MAX(token_count-loss_start_index) "
            "FROM tasks WHERE split='train'"
        ).fetchone()
    finally:
        connection.close()
    if (
        metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or metadata.get("packing_complete") != "true"
        or metadata.get("curriculum_schema_version") != CURRICULUM_SCHEMA
        or metadata.get("maximum_sequence_tokens") != "2048"
        or metadata.get("repair_parent_database_identity_sha256")
        != PARENT_DATABASE_IDENTITY_SHA256
        or metadata.get("repair_train_per_cell") != str(TRAIN_PER_CELL)
        or metadata.get("repair_maximum_target_tokens")
        != str(MAXIMUM_TARGET_TOKENS)
        or metadata.get("repair_families") != ",".join(TASK_FAMILIES)
        or metadata.get("generation_repair_policy_sha256")
        != EXPECTED_REPAIR_POLICY_SHA256
    ):
        raise ValueError("v9 long-generation curriculum metadata is invalid")
    languages = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))
    expected = {
        ("train", language, family): TRAIN_PER_CELL
        for language in languages
        for family in TASK_FAMILIES
    }
    expected.update(
        {
            (split, language, family): 128
            for split in ("validation", "test")
            for language in languages
            for family in TASK_FAMILIES
        }
    )
    if observed != expected:
        raise ValueError("v9 long-generation curriculum cell counts are invalid")
    if int(train_count) != TRAIN_EXAMPLES or int(maximum_target) > MAXIMUM_TARGET_TOKENS:
        raise ValueError("v9 long-generation curriculum bounds are invalid")


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
    if not parent_database.is_file() or not parent_checkpoint.is_file():
        raise ValueError("verified v8 parent artifacts are required")
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
        "--max-steps", "100",
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
        "schema_version": "bosscrafts-v9-long-generation-training-chunk-contract-v1",
        "database": str(database),
        "database_identity_sha256": EXPECTED_DATABASE_IDENTITY,
        "checkpoint": str(checkpoint),
        "parent_database": str(parent_database),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "expected_training_plan_sha256": expected_training_plan_sha256(),
        "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
        "train_examples": TRAIN_EXAMPLES,
        "device": "cuda",
        "dtype": "bfloat16",
        "epochs": 1,
        "full_model": True,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "gradient_checkpointing": True,
        "learning_rate": LEARNING_RATE,
        "maximum_optimizer_steps_this_chunk": 100,
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
