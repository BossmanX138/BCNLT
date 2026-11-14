"""Typed loading for lineage-bound BossCrafts training progress."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, TypedDict, TypeGuard, cast

TRAINING_PROGRESS_SCHEMA = "bosscrafts-training-progress-v2"
MAX_TRAINING_PROGRESS_BYTES = 1024 * 1024


class TrainingProgress(TypedDict):
    schema_version: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    checkpoint_written: bool
    gradients_finite: bool
    full_model: bool
    task_database_identity_sha256: str
    model_fingerprint: str
    training_plan_sha256: str
    planned_optimizer_steps: int
    initialization: object
    total_optimizer_steps: int
    optimizer_steps_remaining: int
    total_examples_seen: int
    mean_loss: float
    learning_rate: float
    training_seconds: float
    training_complete: bool


def _is_nonnegative_finite_number(value: object) -> TypeGuard[int | float]:
    return (
        (type(value) is int or type(value) is float)
        and math.isfinite(value)
        and value >= 0
    )


def _is_nonnegative_int(value: object) -> TypeGuard[int]:
    return type(value) is int and value >= 0


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate training progress key: {key}")
        payload[key] = value
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_progress(
    path: str | Path,
    *,
    expected_fields: Mapping[str, object],
    checkpoint_path: str | Path | None = None,
) -> TrainingProgress:
    try:
        content = Path(path).read_bytes()
        if len(content) > MAX_TRAINING_PROGRESS_BYTES:
            raise ValueError("training progress exceeds size limit")
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training progress is unreadable") from error
    if type(payload) is not dict:
        raise ValueError("training progress must be a JSON object")
    checkpoint_size = payload.get("checkpoint_size_bytes")
    planned_steps = payload.get("planned_optimizer_steps")
    total_steps = payload.get("total_optimizer_steps")
    remaining_steps = payload.get("optimizer_steps_remaining")
    total_examples = payload.get("total_examples_seen")
    numbers = (
        payload.get("mean_loss"),
        payload.get("learning_rate"),
        payload.get("training_seconds"),
    )
    if (
        payload.get("schema_version") != TRAINING_PROGRESS_SCHEMA
        or payload.get("checkpoint_written") is not True
        or not _is_sha256(payload.get("checkpoint_sha256"))
        or type(checkpoint_size) is not int
        or checkpoint_size < 1
        or payload.get("gradients_finite") is not True
        or payload.get("full_model") is not True
        or not _is_sha256(payload.get("task_database_identity_sha256"))
        or not _is_sha256(payload.get("model_fingerprint"))
        or not _is_sha256(payload.get("training_plan_sha256"))
        or "initialization" not in payload
        or type(planned_steps) is not int
        or planned_steps < 1
        or not _is_nonnegative_int(total_steps)
        or not _is_nonnegative_int(remaining_steps)
        or not _is_nonnegative_int(total_examples)
        or any(not _is_nonnegative_finite_number(value) for value in numbers)
        or type(payload.get("training_complete")) is not bool
    ):
        raise ValueError("training progress contract is invalid")
    if total_steps + remaining_steps != planned_steps:
        raise ValueError("training progress counters are inconsistent")
    if payload["training_complete"] is not (remaining_steps == 0):
        raise ValueError("training progress completion is inconsistent")
    if any(
        key not in payload or payload[key] != value
        for key, value in expected_fields.items()
    ):
        raise ValueError("training progress lineage mismatch")
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).resolve()
        try:
            if (
                not checkpoint.is_file()
                or checkpoint.stat().st_size != checkpoint_size
                or _sha256_file(checkpoint) != payload["checkpoint_sha256"]
            ):
                raise ValueError("training progress checkpoint identity mismatch")
        except OSError as error:
            raise ValueError("training progress checkpoint is unreadable") from error
    return cast(TrainingProgress, payload)
