from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .cache import ProjectIndex
from .core import ContextualUnit


@dataclass(frozen=True)
class ProvisionalMeaning:
    category: str
    hypotheses: tuple[str, ...]
    evidence: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class WildcardMemoryRecord:
    signature: str
    surface: str
    relation: str
    category: str
    hypotheses: tuple[str, ...]
    evidence: tuple[str, ...]
    confidence: float
    seen_count: int = 1


class WildcardMemory:
    """Append-only memory for unknown semantic constructs BCNLT has already seen."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.records: dict[str, WildcardMemoryRecord] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def signature_for(self, unit: ContextualUnit) -> str:
        surface = unit.surface.strip().lower()
        relation = (unit.ast_relation or "").strip().lower()
        event = unit.event.strip().upper()
        return f"{event}|{relation}|{surface}"

    def lookup(self, unit: ContextualUnit) -> WildcardMemoryRecord | None:
        return self.records.get(self.signature_for(unit))

    def remember(self, unit: ContextualUnit, meaning: ProvisionalMeaning) -> WildcardMemoryRecord:
        signature = self.signature_for(unit)
        previous = self.records.get(signature)
        seen_count = 1 if previous is None else previous.seen_count + 1
        evidence = _merge_tuple(previous.evidence if previous else (), meaning.evidence)
        hypotheses = _merge_tuple(previous.hypotheses if previous else (), meaning.hypotheses)
        confidence = max(previous.confidence if previous else 0.0, meaning.confidence)
        record = WildcardMemoryRecord(
            signature=signature,
            surface=unit.surface,
            relation=unit.ast_relation or "",
            category=meaning.category,
            hypotheses=hypotheses,
            evidence=evidence,
            confidence=confidence,
            seen_count=seen_count,
        )
        self.records[signature] = record
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_record_to_json(record), sort_keys=True) + "\n")
        return record

    def _load(self) -> None:
        if self.path is None:
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            record = WildcardMemoryRecord(
                signature=str(payload["signature"]),
                surface=str(payload.get("surface", "")),
                relation=str(payload.get("relation", "")),
                category=str(payload.get("category", "extension")),
                hypotheses=tuple(payload.get("hypotheses", ())),
                evidence=tuple(payload.get("evidence", ())),
                confidence=float(payload.get("confidence", 0.0)),
                seen_count=int(payload.get("seen_count", 1)),
            )
            previous = self.records.get(record.signature)
            if previous is None:
                self.records[record.signature] = record
                continue
            self.records[record.signature] = WildcardMemoryRecord(
                signature=record.signature,
                surface=record.surface or previous.surface,
                relation=record.relation or previous.relation,
                category=record.category or previous.category,
                hypotheses=_merge_tuple(previous.hypotheses, record.hypotheses),
                evidence=_merge_tuple(previous.evidence, record.evidence),
                confidence=max(previous.confidence, record.confidence),
                seen_count=max(previous.seen_count, record.seen_count),
            )


class WildcardResolver:
    """Scientific-method-ish resolver for constructs outside the common core."""

    def __init__(self, memory: WildcardMemory | None = None) -> None:
        self.memory = memory or WildcardMemory()

    def resolve(self, unit: ContextualUnit, index: ProjectIndex) -> ProvisionalMeaning:
        surface = unit.surface.lower()
        relation = (unit.ast_relation or "").lower()
        evidence = [
            f"event={unit.event}",
            f"relation={unit.ast_relation}",
            f"scope={unit.scope}",
            f"history={list(unit.recent_history[-3:])}",
        ]
        remembered = self.memory.lookup(unit)
        if remembered is not None:
            signature = self.memory.signature_for(unit)
            return ProvisionalMeaning(
                category=remembered.category,
                hypotheses=remembered.hypotheses,
                evidence=tuple(
                    evidence
                    + [
                        f"wildcard memory hit:{signature}",
                        f"seen_count={remembered.seen_count}",
                    ]
                    + list(remembered.evidence)
                ),
                confidence=min(0.95, max(0.5, remembered.confidence + 0.15)),
            )

        if "match" in surface or "match" in relation:
            return ProvisionalMeaning(
                category="pattern_matching",
                hypotheses=(
                    "construct dispatches control flow by matching a subject against cases",
                    "bindings produced by successful cases may introduce scoped names",
                ),
                evidence=tuple(evidence + ["python AST node Match observed"]),
                confidence=0.64,
            )
        if unit.event == "UNKNOWN_SEMANTIC":
            return ProvisionalMeaning(
                category="extension",
                hypotheses=("construct is outside the v0.1 Python semantic subset",),
                evidence=tuple(evidence),
                confidence=0.35,
            )
        return ProvisionalMeaning(
            category="control_flow",
            hypotheses=("construct may affect execution order or symbol visibility",),
            evidence=tuple(evidence),
            confidence=0.25,
        )

    def remember(self, unit: ContextualUnit, meaning: ProvisionalMeaning) -> WildcardMemoryRecord:
        return self.memory.remember(unit, meaning)

    def resolve_and_remember(self, unit: ContextualUnit, index: ProjectIndex) -> ProvisionalMeaning:
        meaning = self.resolve(unit, index)
        self.remember(unit, meaning)
        return meaning


def _merge_tuple(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for item in (*left, *right):
        if item not in merged:
            merged.append(item)
    return tuple(merged)


def _record_to_json(record: WildcardMemoryRecord) -> dict[str, object]:
    payload = asdict(record)
    payload["hypotheses"] = list(record.hypotheses)
    payload["evidence"] = list(record.evidence)
    return payload
