"""Deterministic learning-rate schedules for resumable BossCrafts training."""
from __future__ import annotations

import math


def cosine_learning_rate(
    step_index: int,
    *,
    total_steps: int,
    peak_learning_rate: float,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if type(step_index) is not int or not 0 <= step_index < total_steps:
        raise ValueError("step_index must be within the frozen training plan")
    if type(warmup_steps) is not int or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be between zero and total_steps minus one")
    if not math.isfinite(peak_learning_rate) or peak_learning_rate <= 0:
        raise ValueError("peak_learning_rate must be finite and positive")
    if not math.isfinite(minimum_ratio) or not 0 < minimum_ratio <= 1:
        raise ValueError("minimum_ratio must be in (0, 1]")
    if step_index < warmup_steps:
        return peak_learning_rate * (step_index + 1) / warmup_steps
    minimum = peak_learning_rate * minimum_ratio
    decay_steps = total_steps - warmup_steps
    progress = (step_index - warmup_steps) / max(1, decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (peak_learning_rate - minimum) * cosine
