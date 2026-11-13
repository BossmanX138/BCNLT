"""Run guarded v7 polish chunks until the frozen one-epoch plan is complete."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bosscrafts_v7_polish_training_chunk import (
    EXPECTED_DATABASE_IDENTITY,
    EXPECTED_MODEL_FINGERPRINT,
    PLANNED_OPTIMIZER_STEPS,
    expected_initialization,
    expected_training_plan_sha256,
)

CHECKPOINT_DIR = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "bosscrafts-engineering-v7-polish"
)
DEFAULT_STATUS = CHECKPOINT_DIR / "training-progress.json"


def _load_status(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError("training status must be a JSON object")
    if not (
        payload.get("checkpoint_written") is True
        and payload.get("gradients_finite") is True
        and payload.get("full_model") is True
        and payload.get("task_database_identity_sha256")
        == EXPECTED_DATABASE_IDENTITY
        and payload.get("model_fingerprint") == EXPECTED_MODEL_FINGERPRINT
        and payload.get("training_plan_sha256") == expected_training_plan_sha256()
        and payload.get("planned_optimizer_steps") == PLANNED_OPTIMIZER_STEPS
        and payload.get("initialization") == expected_initialization()
        and type(payload.get("total_optimizer_steps")) is int
        and type(payload.get("optimizer_steps_remaining")) is int
    ):
        raise ValueError("training status failed the frozen v7 polish contract")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--max-chunks", type=int, default=20)
    arguments = parser.parse_args()
    if not 1 <= arguments.max_chunks <= 20:
        parser.error("max-chunks must be between 1 and 20")
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
                str(PROJECT_ROOT / "scripts" / "run_bosscrafts_v7_polish_training_chunk.py"),
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
                f"v7 polish training chunk failed with exit code {completed.returncode}"
            )
        status = _load_status(status_path)
        current_steps = int(status["total_optimizer_steps"])
        if current_steps <= previous_steps:
            raise RuntimeError("v7 polish training chunk made no optimizer-step progress")
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
    raise RuntimeError("v7 polish training did not complete within the chunk guard")


if __name__ == "__main__":
    raise SystemExit(main())
