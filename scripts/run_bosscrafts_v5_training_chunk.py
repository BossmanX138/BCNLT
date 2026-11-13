"""Run one guarded fresh-start chunk on the deterministic clean v5 curriculum."""
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

from bosscrafts_model.bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from bosscrafts_model.current_task_packing import FAMILIES
from bosscrafts_model.deterministic_curriculum import (
    DETERMINISTIC_CURRICULUM_SCHEMA,
    SPLITS,
)
from bosscrafts_model.resumable_task_packing import TASK_DATABASE_SCHEMA


DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-deterministic-clean-v5"
    / "tasks.sqlite"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluations"
    / "task-prompt-determinism-clean-v5.json"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v5-clean"
    / "checkpoint.pt"
)
EXPECTED_DATABASE_IDENTITY = "626186864cf4444db997bea5bd8ea4a18f18b5042509033a0d6939d0bd35afd2"
EXPECTED_CURRICULUM_POLICY_SHA256 = "53871b42280145abec01f6ca1b57ae7a15bb30f3dc8a7d583c7a79d3d2401c07"
EXPECTED_MODEL_FINGERPRINT = "383bc3acb9bea01ed9a40fb6293a5ac81ee7ffb27ffbc4fdcfeafd96358b3bca"
EXPECTED_QUOTAS = {"train": 2048, "validation": 128, "test": 128}
EXPECTED_LANGUAGES = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))
GRADIENT_ACCUMULATION = 8
TRAIN_EXAMPLES = 86_016
PLANNED_OPTIMIZER_STEPS = math.ceil(TRAIN_EXAMPLES / GRADIENT_ACCUMULATION)
LEARNING_RATE = 0.0002
WARMUP_RATIO = 0.03
MINIMUM_LEARNING_RATE_RATIO = 0.1


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
        "initialization": {"type": "random_seed_7"},
    }
    return hashlib.sha256(
        json.dumps(plan, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _verify_database(database: Path) -> None:
    if not database.is_file():
        raise ValueError("deterministic clean v5 curriculum does not exist")
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
        train_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE split='train'"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    if (
        metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or metadata.get("packing_complete") != "true"
        or metadata.get("curriculum_schema_version")
        != DETERMINISTIC_CURRICULUM_SCHEMA
        or metadata.get("curriculum_policy_sha256")
        != EXPECTED_CURRICULUM_POLICY_SHA256
        or metadata.get("maximum_sequence_tokens") != "2048"
        or not metadata.get("source_task_database_identity_sha256")
    ):
        raise ValueError("deterministic clean v5 curriculum metadata is invalid")
    expected = {
        (split, language, family): EXPECTED_QUOTAS[split]
        for split in SPLITS
        for language in EXPECTED_LANGUAGES
        for family in FAMILIES
    }
    if observed != expected or train_count != TRAIN_EXAMPLES:
        raise ValueError("deterministic clean v5 curriculum is not exactly cell-balanced")


def _verify_determinism_audit(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("determinism audit artifact is missing or invalid") from error
    summary = payload.get("summary")
    if not (
        payload.get("audit_complete") is True
        and payload.get("prompt_targets_deterministic") is True
        and payload.get("heldout_prompt_isolation") is True
        and payload.get("task_database_identity_sha256") == EXPECTED_DATABASE_IDENTITY
        and payload.get("vocabulary_size") == 65_536
        and type(summary) is dict
        and summary.get("example_count") == 96_768
        and summary.get("unique_prompt_count") == 96_768
        and summary.get("duplicate_prompt_group_count") == 0
        and summary.get("conflicting_prompt_group_count") == 0
        and summary.get("cross_split_prompt_group_count") == 0
    ):
        raise ValueError("determinism audit does not authorize v5 training")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--determinism-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    database = arguments.database.resolve()
    audit = arguments.determinism_audit.resolve()
    checkpoint = arguments.checkpoint.resolve()
    status_output = (
        arguments.status_output.resolve()
        if arguments.status_output is not None
        else checkpoint.with_name("training-progress.json")
    )
    _verify_database(database)
    _verify_determinism_audit(audit)
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
    ]
    if resume:
        command.append("--resume")
    contract = {
        "schema_version": "bosscrafts-v5-training-chunk-contract-v1",
        "database": str(database),
        "database_identity_sha256": EXPECTED_DATABASE_IDENTITY,
        "determinism_audit": str(audit),
        "checkpoint": str(checkpoint),
        "device": "cuda",
        "dtype": "bfloat16",
        "epochs": 1,
        "fresh_random_seed": 7,
        "full_model": True,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "gradient_checkpointing": True,
        "learning_rate": LEARNING_RATE,
        "maximum_optimizer_steps_this_chunk": 100,
        "minimum_learning_rate_ratio": MINIMUM_LEARNING_RATE_RATIO,
        "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
        "expected_training_plan_sha256": expected_training_plan_sha256(),
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
