"""Run one explicitly bounded chunk of the frozen BossCrafts training plan."""
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

from bosscrafts_model.balanced_curriculum import (
    CURRICULUM_SCHEMA,
    SPLITS,
    TASK_FAMILIES,
)
from bosscrafts_model.bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from bosscrafts_model.resumable_task_packing import TASK_DATABASE_SCHEMA


DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-balanced-curriculum-v1"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "bosscrafts-engineering-v1" / "checkpoint.pt"
)
EXPECTED_QUOTAS = {"train": 2048, "validation": 128, "test": 128}
EXPECTED_LANGUAGES = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))


def _metadata(database: Path) -> dict[str, str]:
    if not database.is_file():
        raise ValueError("balanced curriculum database does not exist")
    connection = sqlite3.connect(
        f"{database.as_uri()}?mode=ro", uri=True, timeout=30
    )
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        observed_cells = {
            (str(split), str(language), str(family)): int(count)
            for split, language, family, count in connection.execute(
                """SELECT split, language, family, COUNT(*)
                   FROM tasks GROUP BY split, language, family"""
            )
        }
    except sqlite3.DatabaseError as error:
        raise ValueError("balanced curriculum database is structurally invalid") from error
    finally:
        connection.close()
    if (
        metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or metadata.get("packing_complete") != "true"
        or metadata.get("curriculum_schema_version") != CURRICULUM_SCHEMA
        or metadata.get("gate_only") == "true"
        or metadata.get("maximum_sequence_tokens") != "2048"
    ):
        raise ValueError("training requires the completed balanced curriculum database")
    expected_cells = {
        (split, language, family): EXPECTED_QUOTAS[split]
        for split in SPLITS
        for language in EXPECTED_LANGUAGES
        for family in TASK_FAMILIES
    }
    if observed_cells != expected_cells:
        raise ValueError(
            "training requires the exact 14-language, 3-family balanced curriculum"
        )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    database = arguments.database.resolve()
    checkpoint = arguments.checkpoint.resolve()
    status_output = (
        arguments.status_output.resolve()
        if arguments.status_output is not None
        else checkpoint.with_name("training-progress.json")
    )
    _metadata(database)
    resume = checkpoint.is_file()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_bounded_training_trial.py"),
        "--database",
        str(database),
        "--checkpoint",
        str(checkpoint),
        "--status-output",
        str(status_output),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--full-model",
        "--gradient-checkpointing",
        "--gradient-accumulation",
        "8",
        "--learning-rate",
        "0.0002",
        "--warmup-ratio",
        "0.03",
        "--minimum-learning-rate-ratio",
        "0.1",
        "--max-steps",
        "100",
        "--epochs",
        "1",
    ]
    if resume:
        command.append("--resume")
    contract = {
        "schema_version": "bosscrafts-training-chunk-contract-v1",
        "database": str(database),
        "checkpoint": str(checkpoint),
        "device": "cuda",
        "dtype": "bfloat16",
        "epochs": 1,
        "full_model": True,
        "gradient_accumulation": 8,
        "gradient_checkpointing": True,
        "learning_rate": 0.0002,
        "maximum_optimizer_steps_this_chunk": 100,
        "minimum_learning_rate_ratio": 0.1,
        "resume": resume,
        "status_output": str(status_output),
        "warmup_ratio": 0.03,
        "command": command,
    }
    if arguments.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
