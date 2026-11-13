from __future__ import annotations

import json
import unicodedata
from typing import Any

from .events import SemanticEvent
from .ir import IRKind
from .semantic_record import (
    CanonicalIR,
    FrozenMap,
    ScoreBand,
    SemanticTokenRecord,
    freeze_json,
    thaw_json,
)
from .semantic_schema import validate_uncertainty_metadata


_RECORD_FIELDS = {
    "event",
    "surface",
    "scope",
    "span",
    "symbol_id",
    "defines",
    "ast_relation",
    "ir",
    "recent_history",
    "metadata",
    "confidence_band",
    "uncertainty_band",
}
_IR_FIELDS = {"kind", "name", "value", "children", "metadata"}


def canonical_record_bytes(record: SemanticTokenRecord) -> bytes:
    payload = _record_to_payload(record)
    if _normalize_json_strings(payload) != payload:
        raise ValueError("Semantic record strings must use NFC and LF line endings")
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def record_from_canonical_bytes(payload: bytes) -> SemanticTokenRecord:
    try:
        decoded = payload.decode("utf-8")
        raw = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid canonical semantic record: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Canonical semantic record must be a JSON object")
    if _normalize_json_strings(raw) != raw:
        raise ValueError("Canonical semantic record strings must use NFC and LF line endings")
    _require_exact_fields(raw, _RECORD_FIELDS, "semantic record")
    span = raw["span"]
    if (
        not isinstance(span, list)
        or len(span) != 4
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in span)
    ):
        raise ValueError("span must contain four non-negative integers")
    recent_history = raw["recent_history"]
    if not isinstance(recent_history, list) or len(recent_history) > 8:
        raise ValueError("recent_history must be a list with at most eight entries")
    if any(not isinstance(item, str) for item in recent_history):
        raise ValueError("recent_history entries must be strings")
    metadata = _mapping_items(raw["metadata"], "metadata")
    for key, value in metadata:
        validate_uncertainty_metadata(key, thaw_json(value))
    record = SemanticTokenRecord(
        event=SemanticEvent(_required_string(raw, "event")).value,
        surface=_required_string(raw, "surface"),
        scope=_required_string(raw, "scope"),
        span=tuple(span),  # type: ignore[arg-type]
        symbol_id=_optional_string(raw, "symbol_id"),
        defines=_optional_string(raw, "defines"),
        ast_relation=_optional_string(raw, "ast_relation"),
        ir=_ir_from_payload(raw["ir"]) if raw["ir"] is not None else None,
        recent_history=tuple(recent_history),
        metadata=metadata,
        confidence_band=ScoreBand(_required_string(raw, "confidence_band")),
        uncertainty_band=ScoreBand(_required_string(raw, "uncertainty_band")),
    )
    if canonical_record_bytes(record) != payload:
        raise ValueError("Semantic record bytes are not in canonical form")
    return record


def _record_to_payload(record: SemanticTokenRecord) -> dict[str, Any]:
    return {
        "event": SemanticEvent(record.event).value,
        "surface": record.surface,
        "scope": record.scope,
        "span": list(record.span),
        "symbol_id": record.symbol_id,
        "defines": record.defines,
        "ast_relation": record.ast_relation,
        "ir": _ir_to_payload(record.ir) if record.ir is not None else None,
        "recent_history": list(record.recent_history),
        "metadata": {key: thaw_json(value) for key, value in record.metadata},
        "confidence_band": ScoreBand(record.confidence_band).value,
        "uncertainty_band": ScoreBand(record.uncertainty_band).value,
    }


def _ir_to_payload(node: CanonicalIR) -> dict[str, Any]:
    return {
        "kind": IRKind(node.kind).value,
        "name": node.name,
        "value": thaw_json(node.value),
        "children": [_ir_to_payload(child) for child in node.children],
        "metadata": {key: thaw_json(value) for key, value in node.metadata},
    }


def _ir_from_payload(raw: Any) -> CanonicalIR:
    if not isinstance(raw, dict):
        raise ValueError("ir must be an object or null")
    _require_exact_fields(raw, _IR_FIELDS, "ir")
    children = raw["children"]
    if not isinstance(children, list):
        raise ValueError("ir.children must be a list")
    return CanonicalIR(
        kind=IRKind(_required_string(raw, "kind")).value,
        name=_optional_string(raw, "name"),
        value=freeze_json(raw["value"]),
        children=tuple(_ir_from_payload(child) for child in children),
        metadata=_mapping_items(raw["metadata"], "ir.metadata"),
    )


def _mapping_items(raw: Any, field: str):
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be an object")
    frozen = freeze_json(raw)
    if not isinstance(frozen, FrozenMap):
        raise ValueError(f"{field} must be an object")
    return frozen.items


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw[key]
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _require_exact_fields(raw: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{label} has unsupported fields: {sorted(extra)}")


def _normalize_json_strings(value: Any) -> Any:
    if isinstance(value, str):
        normalized_lines = value.replace("\r\n", "\n").replace("\r", "\n")
        return unicodedata.normalize("NFC", normalized_lines)
    if isinstance(value, list):
        return [_normalize_json_strings(item) for item in value]
    if isinstance(value, dict):
        return {
            _normalize_json_strings(key): _normalize_json_strings(item)
            for key, item in value.items()
        }
    return value
