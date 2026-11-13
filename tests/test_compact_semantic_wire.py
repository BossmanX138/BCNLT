from __future__ import annotations

import hashlib
import json

import pytest

from bosscrafts_model.bcnlt_adapter import BCNLTCanonicalRecord
from bosscrafts_model.compact_semantic_wire import CompactSemanticWireV2
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record(
    *,
    event: str,
    surface: str,
    recent_history: list[str],
    confidence: str = "high",
    uncertainty: str = "low",
) -> BCNLTCanonicalRecord:
    payload = {
        "ast_relation": "identifier.self",
        "confidence_band": confidence,
        "defines": None,
        "event": event,
        "ir": {
            "children": [],
            "kind": "variable",
            "metadata": {},
            "name": surface,
            "value": None,
        },
        "metadata": {
            "knowledge_state": "known",
            "uncertainty_signals": [],
            "uyr_rounds": 0,
        },
        "recent_history": recent_history,
        "scope": "module",
        "span": [1, 0, 1, len(surface)],
        "surface": surface,
        "symbol_id": None,
        "uncertainty_band": uncertainty,
    }
    canonical = _canonical(payload)
    return BCNLTCanonicalRecord(
        language="python",
        canonical_json=canonical,
        payload_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        event=event,
        surface=surface,
        confidence_band=confidence,
        uncertainty_band=uncertainty,
        span=(1, 0, 1, len(surface)),
    )


@pytest.fixture(scope="module")
def binding() -> NativeTokenizerBinding:
    return NativeTokenizerBinding.load()


def test_compact_wire_round_trips_derived_and_override_fields(
    binding: NativeTokenizerBinding,
) -> None:
    records = (
        _record(event="DEFINE", surface="alpha", recent_history=[]),
        _record(event="REFERENCE", surface="beta", recent_history=["alpha"]),
        _record(
            event="RETURN",
            surface="gamma",
            recent_history=["explicit override"],
            confidence="high",
            uncertainty="medium",
        ),
    )
    codec = CompactSemanticWireV2(binding)
    encoded = codec.encode(records)

    assert codec.decode(encoded) == "".join(f"{record.canonical_json}\n" for record in records)
    field_end_ids = {
        token_id
        for token, token_id in binding.structural_token_ids.items()
        if ":field:" in token and token.endswith(":end|>")
    }
    assert not field_end_ids.intersection(encoded)
    assert binding.structural_token_ids["<|bcnlt:field:uncertainty:begin|>"] in encoded


def test_compact_wire_rejects_reserved_token_inside_payload(
    binding: NativeTokenizerBinding,
) -> None:
    codec = CompactSemanticWireV2(binding)
    encoded = list(codec.encode((_record(event="DEFINE", surface="alpha", recent_history=[]),)))
    surface_begin = binding.structural_token_ids["<|bcnlt:field:surface:begin|>"]
    position = encoded.index(surface_begin) + 1
    encoded[position] = binding.structural_token_ids["<|bcnlt:task:end|>"]

    with pytest.raises(ValueError, match="reserved token"):
        codec.decode(encoded)


def test_compact_wire_rejects_missing_field_boundary(
    binding: NativeTokenizerBinding,
) -> None:
    codec = CompactSemanticWireV2(binding)
    encoded = list(codec.encode((_record(event="DEFINE", surface="alpha", recent_history=[]),)))
    scope_begin = binding.structural_token_ids["<|bcnlt:field:scope:begin|>"]
    encoded[encoded.index(scope_begin)] = binding.structural_token_ids["<|bcnlt:record:end|>"]

    with pytest.raises(ValueError, match="reserved token|field begin"):
        codec.decode(encoded)
