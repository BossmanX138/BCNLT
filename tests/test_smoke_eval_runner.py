from __future__ import annotations

from bcnlt.languages import CANONICAL_LANGUAGE_IDS
from bcnlt.multilingual_training import TRAINING_FAMILIES
from scripts.run_bcnlt_smoke_eval import _select_heldout


def _rows():
    return [
        {
            "record_id": f"{language}:{family}:{copy}",
            "language": language,
            "family": family,
            "input_ids": [copy],
            "labels": [copy],
            "attention_mask": [1],
        }
        for language in CANONICAL_LANGUAGE_IDS
        for family in TRAINING_FAMILIES
        for copy in range(2)
    ]


def test_select_heldout_is_exactly_language_balanced_and_family_stratified():
    rows = _rows()

    view, indices, counts = _select_heldout(
        rows,
        examples_per_language=32,
        maximum_examples=None,
        seed=1729,
    )

    assert len(view) == len(indices) == 32 * len(CANONICAL_LANGUAGE_IDS)
    assert set(counts["languages"].values()) == {32}
    assert max(counts["families"].values()) - min(counts["families"].values()) <= 1


def test_select_heldout_rejects_conflicting_limits():
    try:
        _select_heldout(
            _rows(),
            examples_per_language=32,
            maximum_examples=128,
            seed=1729,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("conflicting evaluation limits were accepted")
