"""Streaming masked-loss evaluation for BossCrafts task examples."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, TypeGuard

import torch
from torch import nn

from .model import masked_next_token_loss

if TYPE_CHECKING:
    from .resumable_task_packing import TaskExample


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


@dataclass(frozen=True)
class EvaluationBaseline:
    mean_loss: float
    perplexity: float
    target_token_accuracy: float
    target_token_count: int
    correct_target_token_count: int
    exact_match_count: int
    exact_match_rate: float
    language_mean_loss: dict[str, float]
    family_mean_loss: dict[str, float]
    cell_mean_loss: dict[str, float]


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        (type(value) is int or type(value) is float)
        and math.isfinite(value)
    )


def _loss_breakdown(value: object) -> TypeGuard[dict[str, float]]:
    return (
        type(value) is dict
        and bool(value)
        and all(
            type(key) is str
            and bool(key)
            and _finite_number(item)
            and item >= 0
            for key, item in value.items()
        )
    )


def load_evaluation_baseline(
    path: str | Path,
    *,
    expected_split: str,
    expected_example_count: int,
    expected_task_database_identity_sha256: str,
    expected_model_fingerprint: str,
) -> EvaluationBaseline:
    baseline_path = Path(path).resolve()
    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("baseline evaluation is unreadable") from error
    if (
        type(payload) is not dict
        or payload.get("split") != expected_split
        or payload.get("evaluation_complete") is not True
        or payload.get("example_count") != expected_example_count
        or payload.get("split_example_count") != expected_example_count
        or payload.get("task_database_identity_sha256")
        != expected_task_database_identity_sha256
        or payload.get("model_fingerprint") != expected_model_fingerprint
    ):
        raise ValueError("baseline evaluation lineage mismatch")
    mean_loss = payload.get("mean_loss")
    perplexity = payload.get("perplexity")
    target_accuracy = payload.get("target_token_accuracy")
    target_count = payload.get("target_token_count")
    correct_count = payload.get("correct_target_token_count")
    exact_count = payload.get("exact_match_count")
    exact_rate = payload.get("exact_match_rate")
    if (
        not _finite_number(mean_loss)
        or mean_loss <= 0
        or not _finite_number(perplexity)
        or perplexity <= 0
        or not _finite_number(target_accuracy)
        or not 0 <= target_accuracy <= 1
        or type(correct_count) is not int
        or correct_count < 0
        or type(exact_count) is not int
        or not 0 <= exact_count <= expected_example_count
        or not _finite_number(exact_rate)
        or not 0 <= exact_rate <= 1
    ):
        raise ValueError("baseline evaluation metrics are invalid")
    if (
        type(target_count) is not int
        or target_count < 1
        or correct_count > target_count
        or not math.isclose(
            float(target_accuracy),
            correct_count / target_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(perplexity),
            math.exp(float(mean_loss)),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(exact_rate),
            exact_count / expected_example_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("baseline evaluation consistency is invalid")
    language_loss = payload.get("language_mean_loss")
    family_loss = payload.get("family_mean_loss")
    cell_loss = payload.get("cell_mean_loss")
    if (
        not _loss_breakdown(language_loss)
        or not _loss_breakdown(family_loss)
        or not _loss_breakdown(cell_loss)
    ):
        raise ValueError("baseline evaluation breakdown is invalid")
    return EvaluationBaseline(
        mean_loss=float(mean_loss),
        perplexity=float(perplexity),
        target_token_accuracy=float(target_accuracy),
        target_token_count=target_count,
        correct_target_token_count=correct_count,
        exact_match_count=exact_count,
        exact_match_rate=float(exact_rate),
        language_mean_loss=language_loss,
        family_mean_loss=family_loss,
        cell_mean_loss=cell_loss,
    )


def _device_matches(actual: torch.device, requested: torch.device) -> bool:
    return actual.type == requested.type and (
        requested.index is None or actual.index == requested.index
    )


def evaluate_model(
    model: nn.Module,
    examples: Iterable[TaskExample],
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
