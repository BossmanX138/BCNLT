"""Deterministic batch-one training primitives for verified task rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn

from .model import masked_next_token_loss


@dataclass(frozen=True)
class TrainingRunResult:
    optimizer_steps: int
    examples_seen: int
    sequence_tokens_seen: int
    target_tokens_seen: int
    last_example_id: str
    mean_loss: float
    gradients_finite: bool
    last_learning_rate: float


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _device_matches(actual: torch.device, requested: torch.device) -> bool:
    return actual.type == requested.type and (
        requested.index is None or actual.index == requested.index
    )


def run_training_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: Iterable[dict[str, object]],
    *,
    device: torch.device,
    max_steps: int,
    gradient_accumulation: int = 1,
    step_offset: int = 0,
    learning_rate_schedule: Callable[[int], float] | None = None,
) -> TrainingRunResult:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if type(device) is not torch.device:
        raise TypeError("device must be a torch.device")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if type(gradient_accumulation) is not int or gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    if type(step_offset) is not int or step_offset < 0:
        raise ValueError("step_offset must be a nonnegative integer")
    if learning_rate_schedule is not None and not callable(learning_rate_schedule):
        raise TypeError("learning_rate_schedule must be callable or None")
    parameter = next(model.parameters(), None)
    if parameter is None or not _device_matches(parameter.device, device):
        raise ValueError("model parameters must already be on the requested device")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    examples_seen = 0
    sequence_tokens_seen = 0
    target_tokens_seen = 0
    pending = 0
    loss_sum = 0.0
    gradients_finite = True
    last_example_id = ""
    last_learning_rate = float(optimizer.param_groups[0]["lr"])

    def finish_step() -> None:
        nonlocal optimizer_steps, pending, gradients_finite, last_learning_rate
        if pending < gradient_accumulation:
            correction = gradient_accumulation / pending
            for current in model.parameters():
                if current.grad is not None:
                    current.grad.mul_(correction)
        for current in model.parameters():
            if current.grad is not None and not bool(
                torch.isfinite(current.grad).all()
            ):
                gradients_finite = False
                break
        if not gradients_finite:
            raise RuntimeError("training produced a non-finite gradient")
        if learning_rate_schedule is not None:
            learning_rate = float(learning_rate_schedule(step_offset + optimizer_steps))
            if not math.isfinite(learning_rate) or learning_rate <= 0:
                raise RuntimeError("learning-rate schedule produced an invalid value")
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            last_learning_rate = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        pending = 0

    for example in examples:
        if type(example) is not dict:
            raise TypeError("training examples must be dictionaries")
        example_id = example.get("example_id")
        input_ids = example.get("input_ids")
        loss_start = example.get("loss_start_index")
        sampling_weight = example.get("sampling_weight")
        if not _is_sha256(example_id):
            raise ValueError("training example_id is invalid")
        if (
            type(input_ids) is not list
            or len(input_ids) < 2
            or any(type(token_id) is not int for token_id in input_ids)
        ):
            raise ValueError("training input_ids are invalid")
        if type(loss_start) is not int or not 0 < loss_start < len(input_ids):
            raise ValueError("training loss_start_index is invalid")
        if type(sampling_weight) is int or type(sampling_weight) is float:
            weight = float(sampling_weight)
        else:
            raise ValueError("training sampling_weight is invalid")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("training sampling_weight is invalid")
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        boundaries = torch.tensor([loss_start], dtype=torch.long, device=device)
        logits, _ = model(ids)
        loss = masked_next_token_loss(logits, ids, boundaries)
        weighted_loss = loss * weight
        if not bool(torch.isfinite(weighted_loss)):
            raise RuntimeError("training produced a non-finite loss")
        torch.autograd.backward(weighted_loss / gradient_accumulation)
        loss_sum += float(weighted_loss.detach().float().item())
        examples_seen += 1
        sequence_tokens_seen += len(input_ids)
        target_tokens_seen += len(input_ids) - loss_start
        pending += 1
        last_example_id = str(example_id)
        if pending == gradient_accumulation:
            finish_step()
            if optimizer_steps == max_steps:
                break
    if pending and optimizer_steps < max_steps:
        finish_step()
    if examples_seen == 0:
        raise ValueError("training iterator produced no examples")
    return TrainingRunResult(
        optimizer_steps=optimizer_steps,
        examples_seen=examples_seen,
        sequence_tokens_seen=sequence_tokens_seen,
        target_tokens_seen=target_tokens_seen,
        last_example_id=last_example_id,
        mean_loss=loss_sum / examples_seen,
        gradients_finite=gradients_finite,
        last_learning_rate=last_learning_rate,
    )
