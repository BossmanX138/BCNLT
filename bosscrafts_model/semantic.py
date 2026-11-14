"""Deterministic semantic-record handoff for the frozen native tokenizer.

This module validates records emitted by a trusted, external semantic parser. It
does not parse source code into semantics and does not execute generated code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from .tokenizer import (
    EXPECTED_BINDING_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    EXPECTED_VOCABULARY_SIZE,
    NativeTokenizerBinding,
)

SEMANTIC_RECORD_SCHEMA = "bosscrafts-native-semantic-record-v1"
_BEGIN = "<|bcnlt:record:begin|>"
_MODE = "<|bcnlt:mode:semantic_code|>"
_END = "<|bcnlt:record:end|>"
_CONFIDENCE_BANDS = frozenset(("low", "medium", "high"))
_PAYLOAD_KEYS = frozenset(("schema_version", "event", "surface", "confidence_band"))


@dataclass(frozen=True)
class SemanticRecord:
    """Small typed record suitable for a reproducible tokenizer handoff."""

    event: str
    surface: str
    confidence_band: str
    schema_version: str = SEMANTIC_RECORD_SCHEMA


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate semantic-record key: {key}")
        payload[key] = value
    return payload


class NativeSemanticRecordCodec:
    """Fail-closed canonical record codec bound to the exact frozen tokenizer."""

    def __init__(self, binding: NativeTokenizerBinding):
        if (
            type(binding) is not NativeTokenizerBinding
            or binding.binding_sha256 != EXPECTED_BINDING_SHA256
            or binding.tokenizer_sha256 != EXPECTED_TOKENIZER_SHA256
            or binding.vocab_size != EXPECTED_VOCABULARY_SIZE
        ):
            raise ValueError("verified native 65K tokenizer binding required")
        required_markers = {_BEGIN: 9, _MODE: 8, _END: 10}
        if any(binding.structural_token_ids.get(token) != token_id for token, token_id in required_markers.items()):
            raise ValueError("semantic-record structural token contract mismatch")
        event_prefix = "<|bcnlt:event:"
        allowed_events = {
            token[len(event_prefix):-2].upper()
            for token in binding.structural_token_ids
            if token.startswith(event_prefix) and token.endswith("|>")
        }
        if not allowed_events:
            raise ValueError("no frozen semantic event vocabulary")
        self._binding = binding
        self._allowed_events = frozenset(allowed_events)

    @property
    def schema_version(self) -> str:
        return SEMANTIC_RECORD_SCHEMA

    @property
    def parser_integrated(self) -> bool:
        """True only for this serialized-record codec, not source-code parsing."""
        return True

    @property
    def allowed_events(self) -> frozenset[str]:
        return self._allowed_events

    def _validate(self, record: SemanticRecord) -> None:
        if type(record) is not SemanticRecord:
            raise TypeError("record must be a SemanticRecord")
        if record.schema_version != SEMANTIC_RECORD_SCHEMA:
            raise ValueError("semantic-record schema mismatch")
        if type(record.event) is not str or record.event not in self._allowed_events:
            raise ValueError("event is not in the frozen semantic event vocabulary")
        if type(record.surface) is not str or not record.surface or len(record.surface) > 4096:
            raise ValueError("surface must be a non-empty string of at most 4096 characters")
        self._binding.encode_payload(record.surface)
        if type(record.confidence_band) is not str or record.confidence_band not in _CONFIDENCE_BANDS:
            raise ValueError("confidence_band must be low, medium, or high")

    def _canonical_payload(self, record: SemanticRecord) -> str:
        return json.dumps(
            {
                "schema_version": record.schema_version,
                "event": record.event,
                "surface": record.surface,
                "confidence_band": record.confidence_band,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def serialize(self, record: SemanticRecord) -> str:
        self._validate(record)
        return "\n".join((_BEGIN, _MODE, self._canonical_payload(record), _END))

    def parse(self, serialized: str) -> SemanticRecord:
        if type(serialized) is not str:
            raise TypeError("serialized semantic record must be a string")
        lines = serialized.split("\n")
        if len(lines) != 4 or lines[0] != _BEGIN or lines[1] != _MODE or lines[3] != _END:
            raise ValueError("invalid semantic-record envelope")
        try:
            payload = json.loads(lines[2], object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("invalid semantic-record JSON") from error
        if type(payload) is not dict or set(payload) != _PAYLOAD_KEYS:
            raise ValueError("semantic-record payload keys mismatch")
        record = SemanticRecord(
            event=payload["event"],
            surface=payload["surface"],
            confidence_band=payload["confidence_band"],
            schema_version=payload["schema_version"],
        )
        self._validate(record)
        if lines[2] != self._canonical_payload(record):
            raise ValueError("semantic-record JSON is not canonical")
        return record

    def encode(self, record: SemanticRecord) -> list[int]:
        begin_id = self._binding.structural_token_ids[_BEGIN]
        mode_id = self._binding.structural_token_ids[_MODE]
        end_id = self._binding.structural_token_ids[_END]
        newline = self._binding.encode_payload("\n")
        payload = self._binding.encode_payload(self._canonical_payload(record))
        ids = [begin_id, *newline, mode_id, *newline, *payload, *newline, end_id]
        if self._binding.decode(ids) != self.serialize(record):
            raise ValueError("semantic-record token stream did not round-trip")
        if not ids or ids[0] != begin_id or ids[-1] != end_id or mode_id not in ids:
            raise ValueError("tokenizer did not preserve the semantic-record envelope")
        if not (ids.index(begin_id) < ids.index(mode_id) < ids.index(end_id)):
            raise ValueError("semantic-record structural token order mismatch")
        allowed_controls = {begin_id, mode_id, end_id}
        if any(token_id <= 76 and token_id not in allowed_controls for token_id in ids):
            raise ValueError("semantic-record payload emitted a reserved control ID")
        return ids

    def decode(self, ids: Sequence[int]) -> SemanticRecord:
        return self.parse(self._binding.decode(ids))
