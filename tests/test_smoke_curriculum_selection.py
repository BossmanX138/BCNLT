from __future__ import annotations

import json

import pytest

from scripts.select_bcnlt_smoke_curriculum import select_smoke_rows


def _row(family: str, index: int) -> str:
    return json.dumps(
        {"family": family, "record_id": f"{family}-{index}"},
        sort_keys=True,
    ) + "\n"


def test_selection_takes_first_limit_per_family_and_preserves_source_order():
    lines = [
        _row("alpha", 0),
        _row("beta", 0),
        _row("alpha", 1),
        _row("alpha", 2),
        _row("beta", 1),
    ]

    selected, counts = select_smoke_rows(lines, families=("alpha", "beta"), limit=2)

    assert selected == [lines[0], lines[1], lines[2], lines[4]]
    assert counts == {"alpha": 2, "beta": 2}


def test_selection_rejects_missing_family_capacity():
    with pytest.raises(ValueError, match="beta.*1 of 2"):
        select_smoke_rows(
            [_row("alpha", 0), _row("alpha", 1), _row("beta", 0)],
            families=("alpha", "beta"),
            limit=2,
        )
