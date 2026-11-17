from __future__ import annotations

import math

import pytest

from scripts.audit_opencodeinstruct_source import _score
from scripts.compare_compact_task_packing_heldout import _key


def test_compact_task_key_requires_integer_record_index() -> None:
    example: dict[str, object] = {
        "document_id": "doc-1",
        "family": "code_to_semantic",
        "record_index": 7,
    }

    assert _key(example) == ("doc-1", "code_to_semantic", 7)

    for invalid in (True, "7", 7.0, None):
        example["record_index"] = invalid
        with pytest.raises(ValueError, match="record_index must be an integer"):
            _key(example)


def test_opencode_score_accepts_only_bounded_numeric_scalars() -> None:
    assert _score(0) == 0.0
    assert _score(0.5) == 0.5
    assert _score("1.0") == 1.0

    for invalid in (True, False, -0.1, 1.1, "invalid", object(), math.nan):
        assert _score(invalid) is None
