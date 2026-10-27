from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Protocol

from .cache import ProjectIndex
from .core import ContextualUnit, Span
from .events import SemanticEvent
from .ir import IRKind, IRNode
from .uncertainty import HeuristicUncertaintyScorer, UncertaintyScore


class ScoreBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class FrozenMap:
    items: tuple[tuple[str, "FrozenJSON"], ...]


FrozenJSON = None | bool | int | float | str | tuple["FrozenJSON", ...] | FrozenMap


@dataclass(frozen=True)
class CanonicalIR:
    kind: str
    name: str | None
    value: FrozenJSON
    children: tuple["CanonicalIR", ...]
    metadata: tuple[tuple[str, FrozenJSON], ...]


@dataclass(frozen=True)
class SemanticTokenRecord:
    event: str
    surface: str
    scope: str
    span: Span
    symbol_id: str | None
    defines: str | None
    ast_relation: str | None
    ir: CanonicalIR | None
    recent_history: tuple[str, ...]
    metadata: tuple[tuple[str, FrozenJSON], ...]
    confidence_band: ScoreBand
    uncertainty_band: ScoreBand


class UnitScorer(Protocol):
    def score(self, unit: ContextualUnit, index: ProjectIndex) -> UncertaintyScore:
        raise NotImplementedError


_ALLOWED_CONTEXT_METADATA = {
    "dynamic_call": bool,
    "dynamic_binding": bool,
    "unknown_node": str,
}


def classify_confidence(value: float) -> ScoreBand:
    _validate_score(value)
    if value < 0.40:
        return ScoreBand.LOW
    if value < 0.72:
        return ScoreBand.MEDIUM
    return ScoreBand.HIGH


def classify_uncertainty(value: float) -> ScoreBand:
    _validate_score(value)
    if value <= 0.25:
        return ScoreBand.LOW
    if value <= 0.60:
        return ScoreBand.MEDIUM
    return ScoreBand.HIGH


def records_from_units(
    units: Iterable[ContextualUnit],
    index: ProjectIndex,
    scorer: UnitScorer | None = None,
) -> tuple[SemanticTokenRecord, ...]:
    active_scorer = scorer or HeuristicUncertaintyScorer()
    records = []
    for unit in units:
        score = active_scorer.score(unit, index)
        _validate_score(score.confidence)
        _validate_score(score.uncertainty)
        records.append(_record_from_unit(unit, score))
    return tuple(records)


def freeze_json(value: Any) -> FrozenJSON:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("IR values must contain only finite JSON-compatible values")
        return value
    if isinstance(value, str):
        return _normalize(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON-compatible dictionaries require string keys")
            items.append((_normalize(key), freeze_json(item)))
        return FrozenMap(tuple(sorted(items, key=lambda pair: pair[0])))
    raise ValueError(f"IR values must be JSON-compatible, got {type(value).__name__}")


def thaw_json(value: FrozenJSON) -> Any:
    if isinstance(value, FrozenMap):
        return {key: thaw_json(item) for key, item in value.items}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_ir_from_node(node: IRNode) -> CanonicalIR:
    return CanonicalIR(
        kind=IRKind(node.kind).value,
        name=_normalize_optional(node.name),
        value=freeze_json(node.value),
        children=tuple(canonical_ir_from_node(child) for child in node.children),
        metadata=_freeze_mapping(node.metadata),
    )


def _record_from_unit(unit: ContextualUnit, score: UncertaintyScore) -> SemanticTokenRecord:
    event = SemanticEvent(unit.event).value
    metadata = []
    for key, expected_type in _ALLOWED_CONTEXT_METADATA.items():
        if key not in unit.metadata:
            continue
        value = unit.metadata[key]
        if not isinstance(value, expected_type):
            raise ValueError(f"{key} metadata must be {expected_type.__name__}")
        metadata.append((key, freeze_json(value)))
    return SemanticTokenRecord(
        event=event,
        surface=_normalize(unit.surface),
        scope=_normalize(unit.scope),
        span=_validate_span(unit.span),
        symbol_id=_normalize_optional(unit.symbol_id),
        defines=_normalize_optional(unit.defines),
        ast_relation=_normalize_optional(unit.ast_relation),
        ir=canonical_ir_from_node(unit.ir) if unit.ir is not None else None,
        recent_history=tuple(_normalize(item) for item in unit.recent_history[-8:]),
        metadata=tuple(metadata),
        confidence_band=classify_confidence(score.confidence),
        uncertainty_band=classify_uncertainty(score.uncertainty),
    )


def _freeze_mapping(value: dict[str, Any]) -> tuple[tuple[str, FrozenJSON], ...]:
    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenMap):
        raise ValueError("Expected a JSON-compatible dictionary")
    return frozen.items


def _validate_score(value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Score must be a finite number between 0 and 1")
    if not math.isfinite(value):
        raise ValueError("Score must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError("Score must be between 0 and 1")


def _validate_span(value: Span) -> Span:
    if len(value) != 4 or any(not isinstance(item, int) or item < 0 for item in value):
        raise ValueError("Span must contain four non-negative integers")
    return tuple(value)  # type: ignore[return-value]


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def _normalize_optional(value: str | None) -> str | None:
    return _normalize(value) if value is not None else None
