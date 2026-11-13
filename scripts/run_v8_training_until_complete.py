"""Wait for the verified V7 curriculum, then train the full V8 model to completion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.bcnlt_source_pipeline import (  # noqa: E402
    CURRENT_RUNTIME_SPEC_SHA256,
)
from bosscrafts_model.current_task_packing import FAMILIES  # noqa: E402
from bosscrafts_model.resumable_task_packing import audit_task_database  # noqa: E402
from bosscrafts_model.tokenizer import (  # noqa: E402
    NativeTokenizerBinding,
    bound_native_65k_config,
)


V7_ROOT = PROJECT_ROOT.parent / ".codex_staging_bosscrafts_model_v7_candidate"
V7_PIPELINE_STATUS = (
    V7_ROOT / "artifacts" / "pipelines" / "bcnlt-v7-data-v1" / "status.json"
)
DATABASE = (
    V7_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-deterministic-clean-v1"
    / "tasks.sqlite"
)
DETERMINISM_AUDIT = (
    V7_ROOT / "artifacts" / "evaluations" / "task-prompt-determinism-v7.json"
)
CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "bosscrafts-engineering-v8" / "checkpoint.pt"
)
TRAINING_STATUS = CHECKPOINT.with_name("training-progress.json")
DEFAULT_CONTROLLER_STATUS = CHECKPOINT.with_name("training-controller-status.json")
EXPECTED_EXAMPLES = 96_768
EXPECTED_SPLIT_COUNTS = {
    "train": 86_016,
    "validation": 5_376,
    "test": 5_376,
}
GRADIENT_ACCUMULATION = 8
PLANNED_OPTIMIZER_STEPS = math.ceil(
    EXPECTED_SPLIT_COUNTS["train"] / GRADIENT_ACCUMULATION
)
LEARNING_RATE = 0.0002
WARMUP_RATIO = 0.03
MINIMUM_LEARNING_RATE_RATIO = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _wait_for_data(poll_seconds: int) -> dict[str, object]:
    while True:
        if V7_PIPELINE_STATUS.is_file():
            payload = _load_json(V7_PIPELINE_STATUS, "V7 pipeline status")
            if payload.get("data_pipeline_ready") is True:
                return payload
        print(
            json.dumps(
                {
                    "stage": "waiting_for_v7_data_pipeline",
                    "status": str(V7_PIPELINE_STATUS),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def _verify_curriculum(
    binding: NativeTokenizerBinding, pipeline: dict[str, object]
) -> tuple[str, str]:
    audit = audit_task_database(binding, DATABASE)
    if (
        not audit.audit_passed
        or audit.example_count != EXPECTED_EXAMPLES
        or audit.split_counts != EXPECTED_SPLIT_COUNTS
        or pipeline.get("deterministic_task_database_identity_sha256")
        != audit.database_identity_sha256
        or pipeline.get("deterministic_example_count") != EXPECTED_EXAMPLES
    ):
        raise ValueError("V7 deterministic task database failed the V8 contract")

    connection = sqlite3.connect(
        f"{DATABASE.as_uri()}?mode=ro", uri=True, timeout=30
    )
    connection.execute("PRAGMA query_only=ON")
    try:
        cells = {
            (str(split), str(language), str(family)): int(count)
            for split, language, family, count in connection.execute(
                "SELECT split,language,family,COUNT(*) FROM tasks "
                "GROUP BY split,language,family"
            )
        }
    finally:
        connection.close()
    quotas = {"train": 2048, "validation": 128, "test": 128}
    expected_cells = {
        (split, language, family): quotas[split]
        for split in quotas
        for language in sorted(CURRENT_RUNTIME_SPEC_SHA256)
        for family in FAMILIES
    }
    if cells != expected_cells:
        raise ValueError("V7 deterministic curriculum cell quotas are invalid")

    determinism = _load_json(DETERMINISM_AUDIT, "V7 determinism audit")
    summary = determinism.get("summary")
    if not (
        determinism.get("audit_complete") is True
        and determinism.get("prompt_targets_deterministic") is True
        and determinism.get("heldout_prompt_isolation") is True
        and determinism.get("task_database_identity_sha256")
        == audit.database_identity_sha256
        and type(summary) is dict
        and summary.get("example_count") == EXPECTED_EXAMPLES
        and summary.get("unique_prompt_count") == EXPECTED_EXAMPLES
        and summary.get("duplicate_prompt_group_count") == 0
        and summary.get("conflicting_prompt_group_count") == 0
        and summary.get("cross_split_prompt_group_count") == 0
    ):
        raise ValueError("V7 determinism audit does not authorize V8 training")
    return audit.database_identity_sha256, _sha256(DETERMINISM_AUDIT)


def _validate_training_status(
    payload: dict[str, object],
    *,
    database_identity: str,
    model_fingerprint: str,
    expected_plan_sha256: str | None,
) -> tuple[str, int]:
    plan_sha256 = payload.get("training_plan_sha256")
    total_optimizer_steps = payload.get("total_optimizer_steps")
    optimizer_steps_remaining = payload.get("optimizer_steps_remaining")
    if not (
        payload.get("schema_version") == "bosscrafts-training-progress-v1"
        and payload.get("checkpoint_written") is True
        and payload.get("gradients_finite") is True
        and payload.get("full_model") is True
        and payload.get("device") == "cuda"
        and payload.get("dtype") == "bfloat16"
        and payload.get("gradient_checkpointing") is True
        and payload.get("task_database_identity_sha256") == database_identity
        and payload.get("model_fingerprint") == model_fingerprint
        and payload.get("planned_optimizer_steps") == PLANNED_OPTIMIZER_STEPS
        and payload.get("initialization") == {"type": "random_seed_7"}
        and type(plan_sha256) is str
        and len(plan_sha256) == 64
        and type(total_optimizer_steps) is int
        and type(optimizer_steps_remaining) is int
    ):
        raise ValueError("V8 training status failed its frozen contract")
    if expected_plan_sha256 is not None and plan_sha256 != expected_plan_sha256:
        raise ValueError("V8 training plan identity changed between chunks")
    return plan_sha256, total_optimizer_steps


def _training_command() -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_bounded_training_trial.py"),
        "--database",
        str(DATABASE),
        "--checkpoint",
        str(CHECKPOINT),
        "--status-output",
        str(TRAINING_STATUS),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--full-model",
        "--gradient-checkpointing",
        "--gradient-accumulation",
        str(GRADIENT_ACCUMULATION),
        "--learning-rate",
        str(LEARNING_RATE),
        "--warmup-ratio",
        str(WARMUP_RATIO),
        "--minimum-learning-rate-ratio",
        str(MINIMUM_LEARNING_RATE_RATIO),
        "--max-steps",
        "100",
        "--epochs",
        "1",
    ]
    if CHECKPOINT.is_file():
        command.append("--resume")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-chunks", type=int, default=120)
    parser.add_argument("--status", type=Path, default=DEFAULT_CONTROLLER_STATUS)
    parser.add_argument("--status-only", action="store_true")
    arguments = parser.parse_args()
    if not 5 <= arguments.poll_seconds <= 300:
        parser.error("poll-seconds must be between 5 and 300")
    if not 1 <= arguments.max_chunks <= 120:
        parser.error("max-chunks must be between 1 and 120")
    if arguments.status_only:
        payload = {
            "v7_data_ready": V7_PIPELINE_STATUS.is_file(),
            "checkpoint_exists": CHECKPOINT.is_file(),
            "training_status_exists": TRAINING_STATUS.is_file(),
            "controller_status_exists": arguments.status.is_file(),
        }
        if TRAINING_STATUS.is_file():
            payload["training"] = _load_json(TRAINING_STATUS, "V8 training status")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    pipeline = _wait_for_data(arguments.poll_seconds)
    binding = NativeTokenizerBinding.load()
    database_identity, determinism_sha256 = _verify_curriculum(binding, pipeline)
    model_fingerprint = bound_native_65k_config(binding).fingerprint()
    plan_sha256: str | None = None
    previous_steps = -1
    if TRAINING_STATUS.is_file():
        existing = _load_json(TRAINING_STATUS, "V8 training status")
        plan_sha256, previous_steps = _validate_training_status(
            existing,
            database_identity=database_identity,
            model_fingerprint=model_fingerprint,
            expected_plan_sha256=None,
        )
        if existing.get("training_complete") is True:
            final = {
                "schema_version": "bosscrafts-v8-training-controller-status-v1",
                "training_complete": True,
                "already_complete": True,
                "database_identity_sha256": database_identity,
                "determinism_audit_sha256": determinism_sha256,
                "checkpoint": str(CHECKPOINT),
                "checkpoint_sha256": _sha256(CHECKPOINT),
                "training": existing,
            }
            _write_json_atomic(arguments.status, final)
            print(json.dumps(final, indent=2, sort_keys=True), flush=True)
            return 0

    for chunk_index in range(1, arguments.max_chunks + 1):
        command = _training_command()
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"V8 training chunk failed with exit code {completed.returncode}"
            )
        training = _load_json(TRAINING_STATUS, "V8 training status")
        plan_sha256, current_steps = _validate_training_status(
            training,
            database_identity=database_identity,
            model_fingerprint=model_fingerprint,
            expected_plan_sha256=plan_sha256,
        )
        if current_steps <= previous_steps:
            raise RuntimeError("V8 training chunk made no optimizer-step progress")
        previous_steps = current_steps
        progress = {
            "stage": "training",
            "chunk_index": chunk_index,
            "total_optimizer_steps": current_steps,
            "optimizer_steps_remaining": training["optimizer_steps_remaining"],
            "mean_loss": training["mean_loss"],
            "learning_rate": training["learning_rate"],
            "training_seconds": training["training_seconds"],
            "training_complete": training["training_complete"],
        }
        print(json.dumps(progress, sort_keys=True), flush=True)
        if training.get("training_complete") is True:
            final = {
                "schema_version": "bosscrafts-v8-training-controller-status-v1",
                "training_complete": True,
                "already_complete": False,
                "database": str(DATABASE),
                "database_identity_sha256": database_identity,
                "determinism_audit": str(DETERMINISM_AUDIT),
                "determinism_audit_sha256": determinism_sha256,
                "checkpoint": str(CHECKPOINT),
                "checkpoint_sha256": _sha256(CHECKPOINT),
                "model_fingerprint": model_fingerprint,
                "training_plan_sha256": plan_sha256,
                "training": training,
            }
            _write_json_atomic(arguments.status, final)
            print(json.dumps(final, indent=2, sort_keys=True), flush=True)
            return 0
    raise RuntimeError("V8 training did not complete within the chunk guard")


if __name__ == "__main__":
    raise SystemExit(main())
