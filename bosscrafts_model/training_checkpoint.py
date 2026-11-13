"""Atomic, lineage-bound checkpoints for BossCrafts model training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard, cast

import torch
from torch import nn

CHECKPOINT_SCHEMA = "bosscrafts-training-checkpoint-v4"


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class TrainingCheckpointState:
    step: int
    examples_seen: int
    epoch_index: int
    target_epochs: int
    last_example_id: str
    cursor_after_example_id: str | None
    task_database_identity_sha256: str
    binding_sha256: str
    model_fingerprint: str
    training_plan_sha256: str


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    examples_seen: int,
    epoch_index: int,
    target_epochs: int,
    last_example_id: str,
    cursor_after_example_id: str | None,
    task_database_identity_sha256: str,
    binding_sha256: str,
    model_fingerprint: str,
    training_plan_sha256: str,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if type(examples_seen) is not int or examples_seen < 0:
        raise ValueError("examples_seen must be a nonnegative integer")
    if type(target_epochs) is not int or target_epochs < 1:
        raise ValueError("target_epochs must be a positive integer")
    if type(epoch_index) is not int or not 0 <= epoch_index <= target_epochs:
        raise ValueError("epoch_index must be between zero and target_epochs")
    for name, value in (
        ("last_example_id", last_example_id),
        ("task_database_identity_sha256", task_database_identity_sha256),
        ("binding_sha256", binding_sha256),
        ("model_fingerprint", model_fingerprint),
        ("training_plan_sha256", training_plan_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if cursor_after_example_id is not None and not _is_sha256(cursor_after_example_id):
        raise ValueError("cursor_after_example_id must be a lowercase SHA-256 or None")
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "examples_seen": examples_seen,
        "epoch_index": epoch_index,
        "target_epochs": target_epochs,
        "last_example_id": last_example_id,
        "cursor_after_example_id": cursor_after_example_id,
        "task_database_identity_sha256": task_database_identity_sha256,
        "binding_sha256": binding_sha256,
        "model_fingerprint": model_fingerprint,
        "training_plan_sha256": training_plan_sha256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_task_database_identity_sha256: str,
    expected_binding_sha256: str,
    expected_model_fingerprint: str,
    expected_target_epochs: int,
    expected_training_plan_sha256: str,
) -> TrainingCheckpointState:
    payload, state = _load_verified_payload(
        path,
        expected_task_database_identity_sha256=expected_task_database_identity_sha256,
        expected_binding_sha256=expected_binding_sha256,
        expected_model_fingerprint=expected_model_fingerprint,
    )
    if state.target_epochs != expected_target_epochs:
        raise ValueError("training checkpoint epoch plan mismatch")
    if state.training_plan_sha256 != expected_training_plan_sha256:
        raise ValueError("training checkpoint training plan mismatch")
    model.load_state_dict(
        cast(dict[str, Any], payload["model_state_dict"]), strict=True
    )
    optimizer.load_state_dict(
        cast(dict[str, Any], payload["optimizer_state_dict"])
    )
    torch.set_rng_state(cast(torch.Tensor, payload["cpu_rng_state"]))
    cuda_states = payload.get("cuda_rng_state_all")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cast(list[torch.Tensor], cuda_states))
    return state


def load_model_checkpoint_for_inference(
    path: str | Path,
    model: nn.Module,
    *,
    expected_task_database_identity_sha256: str,
    expected_binding_sha256: str,
    expected_model_fingerprint: str,
) -> TrainingCheckpointState:
    payload, state = _load_verified_payload(
        path,
        expected_task_database_identity_sha256=expected_task_database_identity_sha256,
        expected_binding_sha256=expected_binding_sha256,
        expected_model_fingerprint=expected_model_fingerprint,
    )
    model.load_state_dict(
        cast(dict[str, Any], payload["model_state_dict"]), strict=True
    )
    return state


def _load_verified_payload(
    path: str | Path,
    *,
    expected_task_database_identity_sha256: str,
    expected_binding_sha256: str,
    expected_model_fingerprint: str,
) -> tuple[dict[str, object], TrainingCheckpointState]:
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise ValueError("training checkpoint does not exist")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if type(payload) is not dict or payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("training checkpoint schema mismatch")
    expected = {
        "task_database_identity_sha256": expected_task_database_identity_sha256,
        "binding_sha256": expected_binding_sha256,
        "model_fingerprint": expected_model_fingerprint,
    }
    if any(not _is_sha256(value) for value in expected.values()):
        raise ValueError("expected training lineage contains an invalid SHA-256")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("training checkpoint lineage mismatch")
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    cpu_rng_state = payload.get("cpu_rng_state")
    cuda_rng_states = payload.get("cuda_rng_state_all")
    if not isinstance(model_state, dict) or any(
        type(key) is not str for key in model_state
    ):
        raise ValueError("training checkpoint model state is invalid")
    if not isinstance(optimizer_state, dict) or any(
        type(key) is not str for key in optimizer_state
    ):
        raise ValueError("training checkpoint optimizer state is invalid")
    if not isinstance(cpu_rng_state, torch.Tensor) or (
        cpu_rng_state.dtype != torch.uint8 or cpu_rng_state.device.type != "cpu"
    ):
        raise ValueError("training checkpoint CPU RNG state is invalid")
    if cuda_rng_states is not None and (
        type(cuda_rng_states) is not list
        or any(
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.device.type != "cpu"
            for state in cuda_rng_states
        )
    ):
        raise ValueError("training checkpoint CUDA RNG state is invalid")
    step = payload.get("step")
    examples_seen = payload.get("examples_seen")
    epoch_index = payload.get("epoch_index")
    target_epochs = payload.get("target_epochs")
    last_example_id = payload.get("last_example_id")
    cursor_after_example_id = payload.get("cursor_after_example_id")
    training_plan_sha256 = payload.get("training_plan_sha256")
    if (
        type(step) is not int
        or step < 0
        or type(examples_seen) is not int
        or examples_seen < 0
        or type(target_epochs) is not int
        or target_epochs < 1
        or type(epoch_index) is not int
        or not 0 <= epoch_index <= target_epochs
        or not _is_sha256(last_example_id)
        or not _is_sha256(training_plan_sha256)
        or (
            cursor_after_example_id is not None
            and not _is_sha256(cursor_after_example_id)
        )
    ):
        raise ValueError("training checkpoint cursor is invalid")
    state = TrainingCheckpointState(
        step=step,
        examples_seen=examples_seen,
        epoch_index=epoch_index,
        target_epochs=target_epochs,
        last_example_id=last_example_id,
        cursor_after_example_id=cursor_after_example_id,
        task_database_identity_sha256=expected_task_database_identity_sha256,
        binding_sha256=expected_binding_sha256,
        model_fingerprint=expected_model_fingerprint,
        training_plan_sha256=training_plan_sha256,
    )
    return payload, state
