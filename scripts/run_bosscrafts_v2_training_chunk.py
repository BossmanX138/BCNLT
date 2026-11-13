"""Run one bounded warm-start training chunk on the novel v2 curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.balanced_curriculum import CURRICULUM_SCHEMA, SPLITS, TASK_FAMILIES
from bosscrafts_model.bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from bosscrafts_model.resumable_task_packing import TASK_DATABASE_SCHEMA


DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-novel-train-v2"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v2"
    / "checkpoint.pt"
)
DEFAULT_PARENT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-balanced-curriculum-v1"
    / "tasks.sqlite"
)
DEFAULT_PARENT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v1"
    / "checkpoint.pt"
)
EXPECTED_QUOTAS = {"train": 2048, "validation": 128, "test": 128}
EXPECTED_LANGUAGES = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))
PARENT_CHECKPOINT_SHA256 = "080f94ebe9751992991e46971e6d9ff20f241c100f368b114f13fde2aea23342"
PARENT_DATABASE_IDENTITY_SHA256 = "b31654ed0a014240372fcdea3b0827916411ac6f50520a654f12f384d1ef37af"


def _verify_database(database: Path) -> None:
    if not database.is_file():
        raise ValueError("novel v2 curriculum database does not exist")
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        observed = {
            (str(split), str(language), str(family)): int(count)
            for split, language, family, count in connection.execute(
                "SELECT split,language,family,COUNT(*) FROM tasks GROUP BY split,language,family"
            )
        }
    finally:
        connection.close()
    if (
        metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or metadata.get("packing_complete") != "true"
        or metadata.get("curriculum_schema_version") != CURRICULUM_SCHEMA
        or metadata.get("maximum_sequence_tokens") != "2048"
        or not metadata.get("novel_training_policy_sha256")
        or not metadata.get("parent_base_database_identity_sha256")
        or not metadata.get("parent_expanded_database_identity_sha256")
    ):
        raise ValueError("novel v2 curriculum metadata is invalid")
    expected = {
        (split, language, family): EXPECTED_QUOTAS[split]
        for split in SPLITS
        for language in EXPECTED_LANGUAGES
        for family in TASK_FAMILIES
    }
    if observed != expected:
        raise ValueError("novel v2 curriculum is not exactly cell-balanced")


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
        raise ValueError("verified v1 parent artifacts are required")
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
        "--gradient-accumulation", "8",
        "--learning-rate", "0.00005",
        "--warmup-ratio", "0.01",
        "--minimum-learning-rate-ratio", "0.1",
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
        "schema_version": "bosscrafts-v2-training-chunk-contract-v1",
        "database": str(database),
        "checkpoint": str(checkpoint),
        "parent_database": str(parent_database),
        "parent_checkpoint": str(parent_checkpoint),
        "device": "cuda",
        "dtype": "bfloat16",
        "epochs": 1,
        "full_model": True,
        "gradient_accumulation": 8,
        "gradient_checkpointing": True,
        "learning_rate": 0.00005,
        "maximum_optimizer_steps_this_chunk": 100,
        "minimum_learning_rate_ratio": 0.1,
        "resume": resume,
        "status_output": str(status_output),
        "warmup_ratio": 0.01,
        "command": command,
    }
    if arguments.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
