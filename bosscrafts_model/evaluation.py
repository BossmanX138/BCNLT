"""Streaming masked-loss evaluation for BossCrafts task examples."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .model import masked_next_token_loss


@dataclass(frozen=True)
class EvaluationResult:
    example_count: int
    target_token_count: int
    correct_target_token_count: int
    target_token_accuracy: float
    exact_match_count: int
    exact_match_rate: float
    mean_loss: float
    perplexity: float
    language_counts: dict[str, int]
    family_counts: dict[str, int]
    cell_counts: dict[str, int]
    language_mean_loss: dict[str, float]
    family_mean_loss: dict[str, float]
    cell_mean_loss: dict[str, float]


def _device_matches(actual: torch.device, requested: torch.device) -> bool:
    return actual.type == requested.type and (
        requested.index is None or actual.index == requested.index
    )


def evaluate_model(
    model: nn.Module,
    examples: Iterable[dict[str, object]],
    *,
    device: torch.device,
    max_examples: int | None = None,
) -> EvaluationResult:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if type(device) is not torch.device:
        raise TypeError("device must be a torch.device")
    if max_examples is not None and (
        type(max_examples) is not int or max_examples < 1
    ):
        raise ValueError("max_examples must be a positive integer or None")
    parameter = next(model.parameters(), None)
    if parameter is None or not _device_matches(parameter.device, device):
        raise ValueError("model parameters must already be on the requested device")

    example_count = 0
    target_token_count = 0
    correct_target_token_count = 0
    exact_match_count = 0
    weighted_loss_sum = 0.0
    weighted_target_sum = 0.0
    language_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    group_loss: dict[tuple[str, str], float] = defaultdict(float)
    group_weight: dict[tuple[str, str], float] = defaultdict(float)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for example in examples:
                if max_examples is not None and example_count >= max_examples:
                    break
                if type(example) is not dict:
                    raise TypeError("evaluation examples must be dictionaries")
                input_ids = example.get("input_ids")
                loss_start = example.get("loss_start_index")
                sampling_weight = example.get("sampling_weight")
                language = example.get("language")
                family = example.get("family")
                if type(input_ids) is not list or len(input_ids) < 2 or any(
                    type(token_id) is not int for token_id in input_ids
                ):
                    raise ValueError("evaluation input_ids are invalid")
                if type(loss_start) is not int or not 0 < loss_start < len(input_ids):
                    raise ValueError("evaluation loss_start_index is invalid")
                if type(sampling_weight) is int or type(sampling_weight) is float:
                    weight = float(sampling_weight)
                else:
                    raise ValueError("evaluation sampling_weight is invalid")
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError("evaluation sampling_weight is invalid")
                if type(language) is not str or not language or type(family) is not str or not family:
                    raise ValueError("evaluation language and family are required")
                ids = torch.tensor([input_ids], dtype=torch.long, device=device)
                target_projection = getattr(model, "forward_target_logits", None)
                if callable(target_projection):
                    target_logits = target_projection(ids, loss_start)
                    expected_ids = ids[0, loss_start:]
                    loss = nn.functional.cross_entropy(
                        target_logits.reshape(-1, target_logits.shape[-1]),
                        expected_ids,
                    )
                    predicted_ids = target_logits[0].argmax(dim=-1)
                else:
                    boundaries = torch.tensor(
                        [loss_start], dtype=torch.long, device=device
                    )
                    logits, _ = model(ids)
                    loss = masked_next_token_loss(logits, ids, boundaries)
                    predicted_ids = logits[0, loss_start - 1 : -1].argmax(dim=-1)
                    expected_ids = ids[0, loss_start:]
                loss_value = float(loss.detach().float().item())
                if not math.isfinite(loss_value):
                    raise RuntimeError("evaluation produced a non-finite loss")
                targets = len(input_ids) - loss_start
                correct = int((predicted_ids == expected_ids).sum().item())
                correct_target_token_count += correct
                if correct == targets:
                    exact_match_count += 1
                weighted_targets = targets * weight
                weighted_loss_sum += loss_value * weighted_targets
                weighted_target_sum += weighted_targets
                target_token_count += targets
                example_count += 1
                language_counts[language] += 1
                family_counts[family] += 1
                cell = f"{language}|{family}"
                cell_counts[cell] += 1
                for kind, name in (
                    ("language", language),
                    ("family", family),
                    ("cell", cell),
                ):
                    group_loss[(kind, name)] += loss_value * weighted_targets
                    group_weight[(kind, name)] += weighted_targets
    finally:
        model.train(was_training)
    if example_count == 0 or weighted_target_sum <= 0:
        raise ValueError("evaluation iterator produced no examples")
    mean_loss = weighted_loss_sum / weighted_target_sum
    return EvaluationResult(
        example_count=example_count,
        target_token_count=target_token_count,
        correct_target_token_count=correct_target_token_count,
        target_token_accuracy=correct_target_token_count / target_token_count,
        exact_match_count=exact_match_count,
        exact_match_rate=exact_match_count / example_count,
        mean_loss=mean_loss,
        perplexity=math.exp(mean_loss) if mean_loss < 709 else math.inf,
        language_counts=dict(sorted(language_counts.items())),
        family_counts=dict(sorted(family_counts.items())),
        cell_counts=dict(sorted(cell_counts.items())),
        language_mean_loss={
            name: group_loss[("language", name)] / group_weight[("language", name)]
            for name in sorted(language_counts)
        },
        family_mean_loss={
            name: group_loss[("family", name)] / group_weight[("family", name)]
            for name in sorted(family_counts)
        },
        cell_mean_loss={
            name: group_loss[("cell", name)] / group_weight[("cell", name)]
            for name in sorted(cell_counts)
        },
    )
