"""Run guarded v10 instruction chunks until the frozen plan is complete."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.training_progress import (
    TrainingProgress,
    load_training_progress,
)
from scripts.run_bosscrafts_v10_instruction_training_chunk import (
    EXPECTED_DATABASE_IDENTITY,
    EXPECTED_MODEL_FINGERPRINT,
    PLANNED_OPTIMIZER_STEPS,
    expected_initialization,
    expected_training_plan_sha256,
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v10-instruction"
)
DEFAULT_STATUS = CHECKPOINT_DIR / "training-progress.json"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "checkpoint.pt"


def _load_status(path: Path) -> TrainingProgress:
    return load_training_progress(
        path,
        expected_fields={
            "task_database_identity_sha256": EXPECTED_DATABASE_IDENTITY,
            "model_fingerprint": EXPECTED_MODEL_FINGERPRINT,
            "training_plan_sha256": expected_training_plan_sha256(),
            "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
            "initialization": expected_initialization(),
        },
        checkpoint_path=DEFAULT_CHECKPOINT,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--max-chunks", type=int, default=30)
    arguments = parser.parse_args()
    if not 1 <= arguments.max_chunks <= 30:
        parser.error("max-chunks must be between 1 and 30")
    status_path = arguments.status.resolve()
    previous_steps = -1
    if status_path.is_file():
        status = _load_status(status_path)
        previous_steps = int(status["total_optimizer_steps"])
        if status.get("training_complete") is True:
            print(json.dumps({"already_complete": True, **status}, sort_keys=True), flush=True)
            return 0

    for chunk_index in range(1, arguments.max_chunks + 1):
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "run_bosscrafts_v10_instruction_training_chunk.py"
                ),
                "--status-output",
                str(status_path),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            if completed.stdout:
                print(completed.stdout, file=sys.stderr, flush=True)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, flush=True)
            raise RuntimeError(
                "v10 instruction training chunk failed with "
                f"exit code {completed.returncode}"
            )
        status = _load_status(status_path)
        current_steps = int(status["total_optimizer_steps"])
        if current_steps <= previous_steps:
            raise RuntimeError("v10 training chunk made no optimizer-step progress")
        previous_steps = current_steps
        print(
            json.dumps(
                {
                    "chunk_index": chunk_index,
                    "total_optimizer_steps": current_steps,
                    "optimizer_steps_remaining": status["optimizer_steps_remaining"],
                    "total_examples_seen": status["total_examples_seen"],
                    "mean_loss": status["mean_loss"],
                    "learning_rate": status["learning_rate"],
                    "training_seconds": status["training_seconds"],
                    "training_complete": status["training_complete"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if status.get("training_complete") is True:
            return 0
    raise RuntimeError("v10 training did not complete within the chunk guard")


if __name__ == "__main__":
    raise SystemExit(main())
