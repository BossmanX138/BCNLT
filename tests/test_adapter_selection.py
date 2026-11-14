from __future__ import annotations

import pytest

from scripts.select_bcnlt_adapter_candidate import select_candidate


def _candidate(*, steps: int, loss: float, promoted: bool = True):
    return {
        "steps": steps,
        "adapter_weights_sha256": str(steps) * 64,
        "heldout_loss": loss,
        "promoted": promoted,
        "failure_codes": [] if promoted else ["ordinary_instruction_accuracy_below_threshold"],
    }


def test_selection_requires_behavior_then_retention_and_chooses_smallest_intervention():
    decision = select_candidate(
        parent_loss=0.185,
        candidates=(
            _candidate(steps=8, loss=0.187, promoted=False),
            _candidate(steps=16, loss=0.190),
            _candidate(steps=32, loss=0.215),
        ),
        maximum_relative_perplexity_regression=0.02,
    )

    assert decision["selected"]["steps"] == 16
    assert decision["selected"]["status"] == "accepted"
    assert decision["candidates"][0]["status"] == "rejected_behavior"
    assert decision["candidates"][2]["status"] == "rejected_retention"


def test_selection_refuses_to_name_a_winner_when_no_candidate_clears_every_gate():
    decision = select_candidate(
        parent_loss=0.185,
        candidates=(
            _candidate(steps=8, loss=0.186, promoted=False),
            _candidate(steps=32, loss=0.230),
        ),
        maximum_relative_perplexity_regression=0.02,
    )

    assert decision["selected"] is None


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_selection_rejects_nonfinite_parent_and_candidate_losses(value: float):
    with pytest.raises(ValueError, match="finite"):
        select_candidate(
            parent_loss=value,
            candidates=(_candidate(steps=8, loss=0.18),),
            maximum_relative_perplexity_regression=0.02,
        )
    with pytest.raises(ValueError, match="finite"):
        select_candidate(
            parent_loss=0.18,
            candidates=(_candidate(steps=8, loss=value),),
            maximum_relative_perplexity_regression=0.02,
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_selection_rejects_nonfinite_retention_threshold(value: float):
    with pytest.raises(ValueError, match="finite"):
        select_candidate(
            parent_loss=0.18,
            candidates=(_candidate(steps=8, loss=0.18),),
            maximum_relative_perplexity_regression=value,
        )


@pytest.mark.parametrize("value", ("false", "true", 0, 1, None))
def test_selection_requires_a_real_boolean_promotion_decision(value: object):
    candidate = _candidate(steps=8, loss=0.18)
    candidate["promoted"] = value

    with pytest.raises(ValueError, match="promoted"):
        select_candidate(
            parent_loss=0.18,
            candidates=(candidate,),
            maximum_relative_perplexity_regression=0.02,
        )


def test_selection_rejects_nonfinite_report_metrics():
    candidate = _candidate(steps=8, loss=0.18)
    candidate["metrics"] = {"raw_code_accuracy": float("nan")}

    with pytest.raises(ValueError, match="candidate metrics.*finite"):
        select_candidate(
            parent_loss=0.18,
            candidates=(candidate,),
            maximum_relative_perplexity_regression=0.02,
        )


def test_selection_rejects_finite_losses_that_overflow_perplexity():
    with pytest.raises(ValueError, match="finite perplexity"):
        select_candidate(
            parent_loss=1000.0,
            candidates=(_candidate(steps=8, loss=0.18),),
            maximum_relative_perplexity_regression=0.02,
        )
