"""Atomic, lineage-bound checkpoints for BossCrafts model training."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard, cast

import torch
from torch import nn

CHECKPOINT_SCHEMA = "bosscrafts-training-checkpoint-v4"
_CHECKPOINT_FIELDS = frozenset(
    {
        "binding_sha256",
        "cpu_rng_state",
        "cuda_rng_state_all",
        "cursor_after_example_id",
        "epoch_index",
        "examples_seen",
        "last_example_id",
        "model_fingerprint",
        "model_state_dict",
        "optimizer_state_dict",
        "schema_version",
        "step",
        "target_epochs",
        "task_database_identity_sha256",
        "training_plan_sha256",
    }
)


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _state_values_are_finite(value: object) -> bool:
    """Reject non-finite scalar or tensor values in nested checkpoint state."""
    pending = [value]
    seen_containers: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, torch.Tensor):
            if (current.is_floating_point() or current.is_complex()) and not bool(
                torch.isfinite(current).all()
            ):
                return False
        elif type(current) is float:
            if not math.isfinite(current):
                return False
        elif type(current) is complex:
            if not math.isfinite(current.real) or not math.isfinite(current.imag):
                return False
        elif isinstance(current, dict):
            identity = id(current)
            if identity not in seen_containers:
                seen_containers.add(identity)
                pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity not in seen_containers:
                seen_containers.add(identity)
                pending.extend(current)
    return True


def _validated_model_state(
    model: nn.Module,
    payload: dict[str, object],
    *,
    require_dtype: bool,
) -> dict[str, Any]:
    model_state = cast(dict[str, Any], payload["model_state_dict"])
    current_state = model.state_dict()
    if set(model_state) != set(current_state):
        raise ValueError("training checkpoint model state is incompatible")
    for name, current in current_state.items():
        incoming = model_state[name]
        if (
            not isinstance(incoming, torch.Tensor)
            or not isinstance(current, torch.Tensor)
            or incoming.shape != current.shape
            or (require_dtype and incoming.dtype != current.dtype)
            or incoming.layout != current.layout
        ):
            raise ValueError("training checkpoint model state is incompatible")
    return model_state


def _validated_optimizer_state(
    optimizer: torch.optim.Optimizer, payload: dict[str, object]
) -> dict[str, Any]:
    optimizer_state = cast(dict[str, Any], payload["optimizer_state_dict"])
    saved_groups = optimizer_state["param_groups"]
    saved_states = optimizer_state["state"]
    current_groups = optimizer.state_dict()["param_groups"]
    actual_groups = optimizer.param_groups
    if (
        len(saved_groups) != len(current_groups)
        or len(saved_groups) != len(actual_groups)
    ):
        raise ValueError("training checkpoint optimizer state is incompatible")
    parameter_by_saved_id: dict[int, torch.Tensor] = {}
    for saved_group, current_group, actual_group in zip(
        saved_groups, current_groups, actual_groups, strict=True
    ):
        if (
            type(saved_group) is not dict
            or set(saved_group) != set(current_group)
            or type(saved_group.get("params")) is not list
            or type(actual_group.get("params")) is not list
            or len(saved_group["params"]) != len(actual_group["params"])
        ):
            raise ValueError("training checkpoint optimizer state is incompatible")
        for saved_id, parameter in zip(
            saved_group["params"], actual_group["params"], strict=True
        ):
            if (
                type(saved_id) is not int
                or saved_id in parameter_by_saved_id
                or not isinstance(parameter, torch.Tensor)
            ):
                raise ValueError("training checkpoint optimizer state is incompatible")
            parameter_by_saved_id[saved_id] = parameter
    if any(
        type(saved_id) is not int
        or saved_id not in parameter_by_saved_id
        or type(state) is not dict
        for saved_id, state in saved_states.items()
    ):
        raise ValueError("training checkpoint optimizer state is incompatible")
    for saved_id, state in saved_states.items():
        parameter = parameter_by_saved_id[saved_id]
        if any(
            isinstance(value, torch.Tensor)
            and value.numel() != 1
            and value.shape != parameter.shape
            for value in state.values()
        ):
            raise ValueError("training checkpoint optimizer state is incompatible")
    return optimizer_state


def _validate_rng_state_compatibility(payload: dict[str, object]) -> None:
    cpu_state = cast(torch.Tensor, payload["cpu_rng_state"])
    if cpu_state.shape != torch.get_rng_state().shape:
        raise ValueError("training checkpoint CPU RNG state is incompatible")
    cuda_states = payload.get("cuda_rng_state_all")
    if cuda_states is None or not torch.cuda.is_available():
        return
    saved_cuda_states = cast(list[torch.Tensor], cuda_states)
    current_cuda_states = torch.cuda.get_rng_state_all()
    if len(saved_cuda_states) != len(current_cuda_states) or any(
        saved.shape != current.shape
        for saved, current in zip(saved_cuda_states, current_cuda_states, strict=True)
    ):
        raise ValueError("training checkpoint CUDA RNG state is incompatible")


@dataclass(frozen=True)
class TrainingCheckpointState:
    checkpoint_sha256: str
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
    if type(step) is not int or step < 1:
        raise ValueError("step must be a positive integer")
    if type(examples_seen) is not int or examples_seen < step:
        raise ValueError("examples_seen must be an integer at least as large as step")
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
    if (
        cursor_after_example_id is not None
        and cursor_after_example_id != last_example_id
    ):
        raise ValueError("checkpoint cursor must match the last trained example")
    if epoch_index == target_epochs and cursor_after_example_id is not None:
        raise ValueError("completed checkpoint cursor must be None")
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    if not _state_values_are_finite(model_state):
        raise ValueError("training checkpoint contains non-finite model state")
    if not _state_values_are_finite(optimizer_state):
        raise ValueError("training checkpoint contains non-finite optimizer state")
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
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
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
    model_state = _validated_model_state(model, payload, require_dtype=True)
    optimizer_state = _validated_optimizer_state(optimizer, payload)
    _validate_rng_state_compatibility(payload)
    try:
        optimizer.load_state_dict(optimizer_state)
        model.load_state_dict(model_state, strict=True)
    except Exception as error:
        raise ValueError("training checkpoint state could not be restored") from error
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
    model_state = _validated_model_state(model, payload, require_dtype=False)
    try:
        model.load_state_dict(model_state, strict=True)
    except Exception as error:
        raise ValueError("training checkpoint model state could not be restored") from error
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
    expected = {
        "task_database_identity_sha256": expected_task_database_identity_sha256,
        "binding_sha256": expected_binding_sha256,
        "model_fingerprint": expected_model_fingerprint,
    }
    if any(not _is_sha256(value) for value in expected.values()):
        raise ValueError("expected training lineage contains an invalid SHA-256")
    try:
        checkpoint_sha256 = _sha256_file(checkpoint_path)
    except OSError as error:
        raise ValueError("training checkpoint is unreadable") from error
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("training checkpoint is unreadable") from error
    try:
        checkpoint_sha256_after_load = _sha256_file(checkpoint_path)
    except OSError as error:
        raise ValueError("training checkpoint is unreadable") from error
    if checkpoint_sha256_after_load != checkpoint_sha256:
        raise ValueError("training checkpoint changed while loading")
    if (
        type(payload) is not dict
        or set(payload) != _CHECKPOINT_FIELDS
        or payload.get("schema_version") != CHECKPOINT_SCHEMA
    ):
        raise ValueError("training checkpoint fields or schema mismatch")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("training checkpoint lineage mismatch")
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    cpu_rng_state = payload.get("cpu_rng_state")
    cuda_rng_states = payload.get("cuda_rng_state_all")
    if (
        not isinstance(model_state, dict)
        or not model_state
        or any(type(key) is not str for key in model_state)
        or any(not isinstance(value, torch.Tensor) for value in model_state.values())
    ):
        raise ValueError("training checkpoint model state is invalid")
    if not _state_values_are_finite(model_state):
        raise ValueError("training checkpoint contains non-finite model state")
    if (
        not isinstance(optimizer_state, dict)
        or set(optimizer_state) != {"state", "param_groups"}
        or not isinstance(optimizer_state.get("state"), dict)
        or not isinstance(optimizer_state.get("param_groups"), list)
    ):
        raise ValueError("training checkpoint optimizer state is invalid")
    if not _state_values_are_finite(optimizer_state):
        raise ValueError("training checkpoint contains non-finite optimizer state")
    if not isinstance(cpu_rng_state, torch.Tensor):
        raise ValueError("training checkpoint CPU RNG state is invalid")
    if cpu_rng_state.dtype != torch.uint8 or cpu_rng_state.device.type != "cpu":
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
        or step < 1
        or type(examples_seen) is not int
        or examples_seen < step
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
        raise ValueError("training checkpoint progress is invalid")
    if (
        cursor_after_example_id is not None
        and cursor_after_example_id != last_example_id
    ):
        raise ValueError("training checkpoint cursor does not match the last example")
    if epoch_index == target_epochs and cursor_after_example_id is not None:
        raise ValueError("training checkpoint cursor at completion is invalid")
    state = TrainingCheckpointState(
        checkpoint_sha256=checkpoint_sha256,
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
