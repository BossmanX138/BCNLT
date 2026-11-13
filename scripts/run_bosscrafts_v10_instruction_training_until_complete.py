"""Run guarded v10 instruction chunks until the frozen plan is complete."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
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
    read_v10_heldout_manifest,
    v10_evaluation_identities_from_manifest,
    verify_v10_evaluation_contract,
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v10-instruction"
)
DEFAULT_STATUS = CHECKPOINT_DIR / "training-progress.json"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "checkpoint.pt"


def expected_v10_status_fields(
    *,
    evaluation_task_database_identities_sha256: Mapping[str, str],
    heldout_suite_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "task_database_identity_sha256": EXPECTED_DATABASE_IDENTITY,
        "evaluation_task_database_identities_sha256": (
            dict(evaluation_task_database_identities_sha256)
        ),
        "heldout_suite_manifest_sha256": heldout_suite_manifest_sha256,
        "model_fingerprint": EXPECTED_MODEL_FINGERPRINT,
        "training_plan_sha256": expected_training_plan_sha256(
            evaluation_task_database_identities_sha256,
            heldout_suite_manifest_sha256,
        ),
        "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
        "initialization": expected_initialization(),
    }


def build_v10_chunk_command(
    *,
    status_path: Path,
    validation_database: Path,
    test_database: Path,
    heldout_suite_manifest: Path,
) -> list[str]:
    return [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "run_bosscrafts_v10_instruction_training_chunk.py"
        ),
        "--status-output",
        str(status_path),
        "--validation-database",
        str(validation_database),
        "--test-database",
        str(test_database),
        "--heldout-suite-manifest",
        str(heldout_suite_manifest),
    ]


def _load_status(
    path: Path,
    *,
    evaluation_task_database_identities_sha256: Mapping[str, str],
    heldout_suite_manifest_sha256: str,
) -> TrainingProgress:
    return load_training_progress(
        path,
        expected_fields=expected_v10_status_fields(
            evaluation_task_database_identities_sha256=(
                evaluation_task_database_identities_sha256
            ),
            heldout_suite_manifest_sha256=heldout_suite_manifest_sha256,
        ),
        checkpoint_path=DEFAULT_CHECKPOINT,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--validation-database", type=Path, required=True)
    parser.add_argument("--test-database", type=Path, required=True)
    parser.add_argument("--heldout-suite-manifest", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=30)
    arguments = parser.parse_args()
    if not 1 <= arguments.max_chunks <= 30:
        parser.error("max-chunks must be between 1 and 30")
    status_path = arguments.status.resolve()
    validation_database = arguments.validation_database.resolve()
    test_database = arguments.test_database.resolve()
    heldout_suite_manifest = arguments.heldout_suite_manifest.resolve()
    manifest = read_v10_heldout_manifest(heldout_suite_manifest)
    (
        evaluation_database_identities_sha256,
        heldout_suite_manifest_sha256,
    ) = verify_v10_evaluation_contract(
        task_database_identities_sha256=(
            v10_evaluation_identities_from_manifest(manifest)
        ),
        heldout_manifest=manifest,
    )
    previous_steps = -1
    if status_path.is_file():
        status = _load_status(
            status_path,
            evaluation_task_database_identities_sha256=(
                evaluation_database_identities_sha256
            ),
            heldout_suite_manifest_sha256=heldout_suite_manifest_sha256,
        )
        previous_steps = int(status["total_optimizer_steps"])
        if status.get("training_complete") is True:
            print(json.dumps({"already_complete": True, **status}, sort_keys=True), flush=True)
            return 0

    for chunk_index in range(1, arguments.max_chunks + 1):
        completed = subprocess.run(
            build_v10_chunk_command(
                status_path=status_path,
                validation_database=validation_database,
                test_database=test_database,
                heldout_suite_manifest=heldout_suite_manifest,
            ),
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
        status = _load_status(
            status_path,
            evaluation_task_database_identities_sha256=(
                evaluation_database_identities_sha256
            ),
            heldout_suite_manifest_sha256=heldout_suite_manifest_sha256,
        )
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
