"""Experimental lossless compact semantic wire over the frozen 65K vocabulary."""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

from .bcnlt_adapter import BCNLTCanonicalRecord, validate_canonical_record_payload
from .tokenizer import NativeTokenizerBinding


COMPACT_SEMANTIC_WIRE_SCHEMA = "bosscrafts-compact-semantic-wire-v2"
_FIELDS = (
    "surface",
    "scope",
    "symbol_id",
    "defines",
    "span",
    "ast_relation",
    "ir",
    "recent_history",
    "metadata",
    "confidence",
)
_JSON_FIELDS = frozenset(("span", "ir", "recent_history", "metadata"))
_NULLABLE_FIELDS = frozenset(("symbol_id", "defines", "recent_history"))
_INVERSE_BAND = {"high": "low", "medium": "medium", "low": "high"}
_MAX_HISTORY_ITEMS = 8
_MAX_HISTORY_SURFACE_CHARS = 512
_BATCH_SIZE = 128


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate compact-wire JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite compact-wire JSON constant: {value}")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_canonical_json(text: str) -> object:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("compact-wire payload is not valid JSON") from error
    if _canonical(value) != text:
        raise ValueError("compact-wire payload JSON is not canonical")
    return value


def _compact_ir(node: object, *, depth: int = 0) -> object:
    if depth > 64 or type(node) is not dict or set(node) != {
        "kind",
        "name",
        "value",
        "children",
        "metadata",
    }:
        raise ValueError("IR cannot be represented by compact wire")
    if node["value"] is not None or type(node["children"]) is not list:
        raise ValueError("IR cannot be represented by compact wire")
    return [
        node["kind"],
        node["name"],
        [_compact_ir(child, depth=depth + 1) for child in node["children"]],
        node["metadata"],
    ]


def _expand_ir(compact: object, *, depth: int = 0) -> object:
    if (
        depth > 64
        or type(compact) is not list
        or len(compact) != 4
        or type(compact[2]) is not list
        or type(compact[3]) is not dict
    ):
        raise ValueError("compact-wire IR payload is invalid")
    return {
        "kind": compact[0],
        "name": compact[1],
        "value": None,
        "children": [_expand_ir(child, depth=depth + 1) for child in compact[2]],
        "metadata": compact[3],
    }


def _compact_metadata(metadata: object) -> object:
    if type(metadata) is not dict or not {
        "knowledge_state",
        "uncertainty_signals",
        "uyr_rounds",
    }.issubset(metadata):
        raise ValueError("metadata cannot be represented by compact wire")
    optional = {
        key: value
        for key, value in metadata.items()
        if key not in {"knowledge_state", "uncertainty_signals", "uyr_rounds"}
    }
    return [
        metadata["knowledge_state"],
        metadata["uncertainty_signals"],
        metadata["uyr_rounds"],
        optional,
    ]


def _expand_metadata(compact: object) -> object:
    if type(compact) is not list or len(compact) != 4 or type(compact[3]) is not dict:
        raise ValueError("compact-wire metadata payload is invalid")
    required = {"knowledge_state", "uncertainty_signals", "uyr_rounds"}
    if required.intersection(compact[3]):
        raise ValueError("compact-wire metadata duplicates a required key")
    return {
        "knowledge_state": compact[0],
        "uncertainty_signals": compact[1],
        "uyr_rounds": compact[2],
        **compact[3],
    }


class CompactSemanticWireV2:
    """Encode canonical records with reconstructible redundancy removed."""

    def __init__(self, binding: NativeTokenizerBinding) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        self._binding = binding
        self._reserved_ids = frozenset(range(77))
        self._event_by_id = {
            token_id: token.removeprefix("<|bcnlt:event:").removesuffix("|>").upper()
            for token, token_id in binding.structural_token_ids.items()
            if token.startswith("<|bcnlt:event:")
        }
        self._allowed_events = frozenset(self._event_by_id.values())

    @property
    def schema_version(self) -> str:
        return COMPACT_SEMANTIC_WIRE_SCHEMA

    def _id(self, name: str) -> int:
        try:
            return self._binding.structural_token_ids[f"<|bcnlt:{name}|>"]
        except KeyError as error:
            raise ValueError(f"frozen structural token is missing: {name}") from error

    @staticmethod
    def _derived_history(prior_surfaces: Sequence[str]) -> list[str]:
        return [
            surface[:_MAX_HISTORY_SURFACE_CHARS]
            for surface in prior_surfaces[-_MAX_HISTORY_ITEMS:]
        ]

    def _encode_fragments(
        self,
        records: Sequence[BCNLTCanonicalRecord],
        *,
        independent_history: bool,
    ) -> tuple[tuple[int, ...], ...]:
        if any(type(record) is not BCNLTCanonicalRecord for record in records):
            raise TypeError("records must contain BCNLTCanonicalRecord values")
        fragments: list[tuple[int, ...]] = []
        prior_surfaces: list[str] = []
        for start in range(0, len(records), _BATCH_SIZE):
            payload_texts: list[str] = []
            prepared: list[
                tuple[BCNLTCanonicalRecord, list[tuple[str, object, int | None]], str | None]
            ] = []
            for record in records[start : start + _BATCH_SIZE]:
                payload = _load_canonical_json(record.canonical_json)
                validate_canonical_record_payload(payload, allowed_events=self._allowed_events)
                if type(payload) is not dict:  # guarded by validation
                    raise ValueError("canonical record payload must be an object")
                if (
                    record.payload_sha256
                    != hashlib.sha256(record.canonical_json.encode("utf-8")).hexdigest()
                    or record.event != payload["event"]
                    or record.surface != payload["surface"]
                    or record.confidence_band != payload["confidence_band"]
                    or record.uncertainty_band != payload["uncertainty_band"]
                    or record.span != tuple(payload["span"])
                ):
                    raise ValueError("canonical record dataclass does not match its payload")
                derived_history = (
                    [] if independent_history else self._derived_history(prior_surfaces)
                )
                fields: list[tuple[str, object, int | None]] = []
                for field in _FIELDS:
                    if field == "confidence":
                        value = payload["confidence_band"]
                    else:
                        value = payload[field]
                    if field == "recent_history" and value == derived_history:
                        value = None
                    elif field == "ir":
                        value = _compact_ir(value)
                    elif field == "metadata":
                        value = _compact_metadata(value)
                    payload_index: int | None = None
                    if value is not None and field != "confidence":
                        text = _canonical(value) if field in _JSON_FIELDS else value
                        if type(text) is not str:
                            raise ValueError(f"compact {field} payload must serialize to text")
                        payload_index = len(payload_texts)
                        payload_texts.append(text)
                    fields.append((field, value, payload_index))
                uncertainty_override = None
                if payload["uncertainty_band"] != _INVERSE_BAND[payload["confidence_band"]]:
                    uncertainty_override = str(payload["uncertainty_band"])
                prepared.append((record, fields, uncertainty_override))
                if not independent_history:
                    prior_surfaces.append(str(payload["surface"]))

            encoded_payloads = self._binding.encode_payload_batch(payload_texts)
            for record, fields, uncertainty_override in prepared:
                fragment = [
                    self._id("record:begin"),
                    self._id(f"event:{record.event.lower()}"),
                ]
                for field, value, payload_index in fields:
                    fragment.append(self._id(f"field:{field}:begin"))
                    if value is None:
                        fragment.append(self._id("value:null"))
                    elif field == "confidence":
                        fragment.append(self._id(f"value:{value}"))
                    elif payload_index is not None:
                        fragment.extend(encoded_payloads[payload_index])
                    else:  # pragma: no cover - guarded above
                        raise RuntimeError("compact payload batch index is missing")
                if uncertainty_override is not None:
                    fragment.extend(
                        (
                            self._id("field:uncertainty:begin"),
                            self._id(f"value:{uncertainty_override}"),
                        )
                    )
                fragment.append(self._id("record:end"))
                fragments.append(tuple(fragment))
        return tuple(fragments)

    def encode(self, records: Sequence[BCNLTCanonicalRecord]) -> tuple[int, ...]:
        fragments = self._encode_fragments(records, independent_history=False)
        ids = [self._id("stream:begin"), self._id("mode:semantic_code")]
        for fragment in fragments:
            ids.extend(fragment)
        ids.append(self._id("stream:end"))
        return tuple(ids)

    def encode_independent_fragments(
        self, records: Sequence[BCNLTCanonicalRecord]
    ) -> tuple[tuple[int, ...], ...]:
        """Encode records so every fragment can decode without earlier records."""

        return self._encode_fragments(records, independent_history=True)

    def _payload_body(
        self,
        values: tuple[int, ...],
        cursor: int,
        next_marker: int,
        *,
        field: str,
    ) -> tuple[list[int], int]:
        body: list[int] = []
        while cursor < len(values) - 1 and values[cursor] != next_marker:
            if values[cursor] in self._reserved_ids:
                raise ValueError(f"unexpected reserved token inside compact {field}")
            body.append(values[cursor])
            cursor += 1
        if cursor >= len(values) - 1 or not body:
            raise ValueError(f"compact {field} field is truncated or empty")
        return body, cursor

    def decode(self, ids: Sequence[int]) -> str:
        values = tuple(ids)
        if len(values) < 3:
            raise ValueError("compact semantic token stream is truncated")
        if values[:2] != (self._id("stream:begin"), self._id("mode:semantic_code")):
            raise ValueError("compact semantic token stream header mismatch")
        if values[-1] != self._id("stream:end"):
            raise ValueError("compact semantic token stream end mismatch")

        cursor = 2
        lines: list[str] = []
        prior_surfaces: list[str] = []
        band_by_id = {
            self._id("value:low"): "low",
            self._id("value:medium"): "medium",
            self._id("value:high"): "high",
        }
        while cursor < len(values) - 1:
            if values[cursor] != self._id("record:begin"):
                raise ValueError("compact semantic record begin mismatch")
            cursor += 1
            try:
                event = self._event_by_id[values[cursor]]
            except (KeyError, IndexError) as error:
                raise ValueError("compact semantic event token mismatch") from error
            cursor += 1
            payload: dict[str, object] = {"event": event}
            for index, field in enumerate(_FIELDS):
                if cursor >= len(values) - 1 or values[cursor] != self._id(
                    f"field:{field}:begin"
                ):
                    raise ValueError(f"compact {field} field begin mismatch")
                cursor += 1
                if field == "confidence":
                    try:
                        payload["confidence_band"] = band_by_id[values[cursor]]
                    except (KeyError, IndexError) as error:
                        raise ValueError("compact confidence band mismatch") from error
                    cursor += 1
                    continue

                next_marker = self._id(f"field:{_FIELDS[index + 1]}:begin")
                if field in _NULLABLE_FIELDS and values[cursor] == self._id("value:null"):
                    value: object = None
                    cursor += 1
                    if cursor >= len(values) - 1 or values[cursor] != next_marker:
                        raise ValueError(f"compact {field} null field boundary mismatch")
                else:
                    body, cursor = self._payload_body(
                        values,
                        cursor,
                        next_marker,
                        field=field,
                    )
                    text = self._binding.decode(body)
                    value = _load_canonical_json(text) if field in _JSON_FIELDS else text

                if field == "recent_history" and value is None:
                    value = self._derived_history(prior_surfaces)
                elif field == "ir":
                    value = _expand_ir(value)
                elif field == "metadata":
                    value = _expand_metadata(value)
                payload[field] = value

            confidence = payload["confidence_band"]
            if cursor < len(values) - 1 and values[cursor] == self._id(
                "field:uncertainty:begin"
            ):
                cursor += 1
                try:
                    uncertainty = band_by_id[values[cursor]]
                except (KeyError, IndexError) as error:
                    raise ValueError("compact uncertainty override mismatch") from error
                cursor += 1
            else:
                try:
                    uncertainty = _INVERSE_BAND[str(confidence)]
                except KeyError as error:  # pragma: no cover - confidence decoded above
                    raise ValueError("compact confidence band is invalid") from error
            payload["uncertainty_band"] = uncertainty
            if cursor >= len(values) - 1 or values[cursor] != self._id("record:end"):
                raise ValueError("compact semantic record end mismatch")
            cursor += 1
            validate_canonical_record_payload(payload, allowed_events=self._allowed_events)
            canonical = _canonical(payload)
            lines.append(canonical)
            prior_surfaces.append(str(payload["surface"]))
        return "".join(f"{line}\n" for line in lines)
