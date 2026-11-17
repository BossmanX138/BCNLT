"""Run one guarded low-rate polish chunk from the completed v6 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bosscrafts_v6_repair_training_chunk import (
    DEFAULT_DATABASE,
    EXPECTED_DATABASE_IDENTITY,
    GRADIENT_ACCUMULATION,
    PLANNED_OPTIMIZER_STEPS,
    TRAIN_EXAMPLES,
    _verify_database,
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v7-polish"
    / "checkpoint.pt"
)
DEFAULT_PARENT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v6-repair"
    / "checkpoint.pt"
)
PARENT_CHECKPOINT_SHA256 = "4d4ae6b78c23e9407afac55973db46a5298e65238103f01c93ee6cc91f40439e"
PARENT_DATABASE_IDENTITY_SHA256 = EXPECTED_DATABASE_IDENTITY
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT)
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    database = arguments.database.resolve()
    checkpoint = arguments.checkpoint.resolve()
    parent_checkpoint = arguments.parent_checkpoint.resolve()
    status_output = (
        arguments.status_output.resolve()
        if arguments.status_output is not None
        else checkpoint.with_name("training-progress.json")
    )
    _verify_database(database)
    if not parent_checkpoint.is_file():
        raise ValueError("verified v6 parent checkpoint is required")
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
                "--initial-checkpoint-database", str(database),
            ]
        )
    contract = {
        "schema_version": "bosscrafts-v7-polish-training-chunk-contract-v1",
        "database": str(database),
        "database_identity_sha256": EXPECTED_DATABASE_IDENTITY,
        "checkpoint": str(checkpoint),
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
