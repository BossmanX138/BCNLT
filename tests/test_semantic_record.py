from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from bcnlt.cache import ProjectIndex
from bcnlt.canonical import canonical_record_bytes, record_from_canonical_bytes
from bcnlt.core import ContextualUnit
from bcnlt.ir import IRKind, IRNode
from bcnlt.semantic_record import (
    ScoreBand,
    SemanticTokenRecord,
    classify_confidence,
    classify_uncertainty,
    records_from_units,
)
from bcnlt.uncertainty import UncertaintyScore


def _unit(**overrides) -> ContextualUnit:
    values = {
        "surface": "cafe\u0301",
        "event": "DEFINE",
        "scope": "module",
        "span": (3, 1, 3, 7),
        "symbol_id": "module.cafe\u0301",
        "defines": None,
        "ast_relation": "Name.store",
        "ir": IRNode(
            IRKind.FUNCTION,
            name="cafe\u0301",
            value={"z": [2, 1], "a": True},
            children=(IRNode(IRKind.VALUE, value="na\u0308ive"),),
            metadata={"zeta": 2, "alpha": ["b", "a"]},
        ),
        "recent_history": tuple(f"EVENT:{index}" for index in range(10)),
        "metadata": {
            "dynamic_call": True,
            "unknown_node": "Custom",
            "runtime_cache": "must not enter the record",
        },
    }
    values.update(overrides)
    return ContextualUnit(**values)


def test_record_projection_and_canonical_bytes_round_trip_all_stable_fields():
    record = records_from_units([_unit()], ProjectIndex())[0]

    payload = canonical_record_bytes(record)
    decoded = record_from_canonical_bytes(payload)
    raw = json.loads(payload)

    assert decoded == record
    assert payload.endswith(b"\n")
    assert record.surface == "caf\u00e9"
    assert record.symbol_id == "module.caf\u00e9"
    assert record.recent_history == tuple(f"EVENT:{index}" for index in range(2, 10))
    assert record.metadata == (("dynamic_call", True), ("unknown_node", "Custom"))
    assert raw["ir"]["metadata"] == {"alpha": ["b", "a"], "zeta": 2}
    assert raw["ir"]["children"][0]["value"] == "n\u00e4ive"
    assert "runtime_cache" not in raw["metadata"]


def test_absent_optional_fields_are_explicit_nulls():
    record = records_from_units(
        [_unit(symbol_id=None, defines=None, ast_relation=None, ir=None, metadata={})],
        ProjectIndex(),
    )[0]

    payload = json.loads(canonical_record_bytes(record))

    assert payload["symbol_id"] is None
    assert payload["defines"] is None
    assert payload["ast_relation"] is None
    assert payload["ir"] is None
    assert payload["metadata"] == {}


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, ScoreBand.LOW),
        (0.399999, ScoreBand.LOW),
        (0.40, ScoreBand.MEDIUM),
        (0.719999, ScoreBand.MEDIUM),
        (0.72, ScoreBand.HIGH),
        (1.0, ScoreBand.HIGH),
    ],
)
def test_confidence_band_boundaries(score: float, expected: ScoreBand):
    assert classify_confidence(score) is expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, ScoreBand.LOW),
        (0.25, ScoreBand.LOW),
        (0.250001, ScoreBand.MEDIUM),
        (0.60, ScoreBand.MEDIUM),
        (0.600001, ScoreBand.HIGH),
        (1.0, ScoreBand.HIGH),
    ],
)
def test_uncertainty_band_boundaries(score: float, expected: ScoreBand):
    assert classify_uncertainty(score) is expected


@pytest.mark.parametrize("score", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_score_bands_reject_out_of_range_or_non_finite_values(score: float):
    with pytest.raises(ValueError):
        classify_confidence(score)
    with pytest.raises(ValueError):
        classify_uncertainty(score)


def test_projection_rejects_non_json_ir_values():
    with pytest.raises(ValueError, match="JSON-compatible"):
        records_from_units(
            [_unit(ir=IRNode(IRKind.VALUE, value=object()))],
            ProjectIndex(),
        )


def test_projection_rejects_wrong_types_in_allowed_metadata():
    with pytest.raises(ValueError, match="dynamic_call"):
        records_from_units(
            [_unit(metadata={"dynamic_call": "yes"})],
            ProjectIndex(),
        )


def test_projection_rejects_non_finite_scorer_results():
    class BadScorer:
        def score(self, unit, index):
            return UncertaintyScore(uncertainty=math.nan, confidence=0.5, signals=())

    with pytest.raises(ValueError, match="finite"):
        records_from_units([_unit()], ProjectIndex(), scorer=BadScorer())


def test_canonical_decoder_rejects_missing_required_score_bands():
    record = records_from_units([_unit()], ProjectIndex())[0]
    payload = json.loads(canonical_record_bytes(record))
    del payload["confidence_band"]

    with pytest.raises(ValueError, match="confidence_band"):
        record_from_canonical_bytes(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )


def test_semantic_token_record_is_frozen():
    record = records_from_units([_unit()], ProjectIndex())[0]

    with pytest.raises(Exception):
        record.surface = "changed"  # type: ignore[misc]
    assert isinstance(record, SemanticTokenRecord)


@pytest.mark.parametrize("surface", ["cafe\u0301", "first\r\nsecond"])
def test_direct_record_encoding_rejects_noncanonical_strings(surface: str):
    record = records_from_units([_unit()], ProjectIndex())[0]

    with pytest.raises(ValueError, match="NFC|line endings|canonical"):
        canonical_record_bytes(replace(record, surface=surface))


@pytest.mark.parametrize("surface", ["cafe\u0301", "first\r\nsecond"])
def test_canonical_decoder_rejects_noncanonical_strings(surface: str):
    record = records_from_units([_unit()], ProjectIndex())[0]
    payload = json.loads(canonical_record_bytes(record))
    payload["surface"] = surface
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")

    with pytest.raises(ValueError, match="NFC|line endings|canonical"):
        record_from_canonical_bytes(encoded)
