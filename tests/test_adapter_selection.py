from __future__ import annotations

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
