from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bcnlt.adaptive_vocab.model_eval import (
    AdaptiveEvaluationConfig,
    AdaptiveModelInput,
    PairedHeldout,
    evaluate_adaptive_matrix,
    evaluate_adaptive_promotion,
    evaluate_nll_bits_per_byte,
)
from bcnlt.tokenizer_eval import EvaluationCase, EvaluationSuite, GenerationSettings


class UniformModel(nn.Module):
    def __init__(self, vocab_size: int = 2):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, *, input_ids, attention_mask=None, labels=None, use_cache=False):
        shape = (*input_ids.shape, self.vocab_size)
        return SimpleNamespace(logits=torch.zeros(shape, device=input_ids.device) + self.anchor)


def _row(record_id: str, target: bytes):
    return {
        "record_id": record_id,
        "target_bytes": target,
        "input_ids": [0, 0, 0],
        "labels": [-100, 0, 1],
        "attention_mask": [1, 1, 1],
    }


def test_nll_sums_bits_before_dividing_by_identical_canonical_bytes():
    report = evaluate_nll_bits_per_byte(
        UniformModel(),
        [_row("a", b"abc"), _row("b", b"d")],
        [3, 1],
    )

    assert report.total_nll_bits == pytest.approx(4.0)
    assert report.total_target_bytes == 4
    assert report.bits_per_byte == pytest.approx(1.0)
    assert report.mean_nll_bits_per_token == pytest.approx(1.0)
    assert report.scored_tokens == 4
    assert len(report.target_sha256) == 64


def test_nll_ignores_masked_labels_and_rejects_invalid_target_sidecars():
    masked = _row("a", b"z")
    masked["labels"] = [-100, -100, 1]
    report = evaluate_nll_bits_per_byte(UniformModel(), [masked], [1])
    assert report.scored_tokens == 1
    assert report.total_nll_bits == pytest.approx(1.0)

    with pytest.raises(ValueError, match="positive"):
        evaluate_nll_bits_per_byte(UniformModel(), [_row("a", b"")], [0])
    with pytest.raises(ValueError, match="target byte length"):
        evaluate_nll_bits_per_byte(UniformModel(), [_row("a", b"abc")], [2])


SETTINGS = GenerationSettings(seed=17, temperature=0.0, max_new_tokens=64)
CASES = (
    EvaluationCase("struct", "structural", "struct?", "pkg::value"),
    EvaluationCase("semantic", "semantic_to_code", "code?", "x = 1\n"),
    EvaluationCase("raw", "raw_code", "raw?", "λ = 1\n"),
    EvaluationCase("ordinary", "ordinary_instruction", "say?", "ready"),
    EvaluationCase("tokenizer", "tokenizer", "token?", "hello λ\n"),
)
SUITE = EvaluationSuite("bcnlt-tokenizer-eval-v1", SETTINGS, CASES)
TARGETS = (b"alpha\n", b"beta\n")
PAIRED = PairedHeldout(("a", "b"), TARGETS)


def _generator(prompt, settings):
    assert settings == SETTINGS
    return {case.prompt: case.expected for case in CASES}[prompt]


def _state(
    identity: str,
    *,
    tokens: int,
    losses=(1.0,),
    reloadable=True,
    generate=_generator,
    model=None,
):
    return AdaptiveModelInput(
        identity=identity,
        model=model or UniformModel(),
        dataset=(_row("a", TARGETS[0]), _row("b", TARGETS[1])),
        record_ids=("a", "b"),
        target_bytes=TARGETS,
        generate=generate,
        semantic_input_tokens=tokens,
        training_losses=tuple(losses),
        adapter_reloadable=reloadable,
    )


def test_matrix_requires_four_matched_states_and_passes_shared_unit_gates():
    report = evaluate_adaptive_matrix(
        _state("frozen_v4", tokens=100),
        _state("untrained_v5", tokens=85),
        _state("v1_control", tokens=100),
        _state("trained_v5", tokens=85),
        SUITE,
        PAIRED,
        AdaptiveEvaluationConfig(minimum_semantic_compression=0.10),
    )

    assert report.state_names == (
        "frozen_v4", "untrained_v5", "v1_control", "trained_v5"
    )
    assert report.semantic_compression == pytest.approx(0.15)
    assert report.failure_codes == ()
    proposal = evaluate_adaptive_promotion(
        report,
        {"training_eligible": True, "selected_manifest_sha256": "a" * 64},
    )
    assert proposal.eligible is True
    assert proposal.promoted is False
    assert proposal.failure_codes == ()


def test_matrix_reports_stable_failures_and_promotion_never_mutates_state():
    wrong_target = AdaptiveModelInput(
        **{
            **_state("trained_v5", tokens=95, losses=(math.nan,), reloadable=False).__dict__,
            "target_bytes": (b"alpha\n", b"changed\n"),
        }
    )
    report = evaluate_adaptive_matrix(
        _state("frozen_v4", tokens=100),
        _state("untrained_v5", tokens=95),
        _state("v1_control", tokens=100),
        wrong_target,
        SUITE,
        PAIRED,
        AdaptiveEvaluationConfig(minimum_semantic_compression=0.10),
    )

    assert set(report.failure_codes) >= {
        "target_hash_mismatch",
        "semantic_compression_below_threshold",
        "non_finite_training_trace",
        "adapter_not_reloadable",
    }
    proposal = evaluate_adaptive_promotion(report, {"training_eligible": False})
    assert proposal.promoted is False
    assert proposal.eligible is False
    assert "tokenizer_selection_not_eligible" in proposal.failure_codes


def test_matrix_rejects_bits_per_byte_and_exact_final_newline_regressions():
    def missing_newline(prompt, settings):
        return _generator(prompt, settings).rstrip("\n")

    report = evaluate_adaptive_matrix(
        _state("frozen_v4", tokens=100),
        _state("untrained_v5", tokens=85),
        _state("v1_control", tokens=100),
        _state(
            "trained_v5",
            tokens=85,
            model=UniformModel(vocab_size=4),
            generate=missing_newline,
        ),
        SUITE,
        PAIRED,
        AdaptiveEvaluationConfig(minimum_semantic_compression=0.10),
    )

    assert set(report.failure_codes) >= {
        "bits_per_byte_regression_frozen_v4",
        "bits_per_byte_regression_v1_control",
        "exact_output_regression",
    }

