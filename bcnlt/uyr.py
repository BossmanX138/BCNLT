from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from .core import ContextualUnit, SourceDocument
from .ir import IRKind, IRNode
from .uncertainty import HeuristicUncertaintyScorer, UncertaintyScore


class LearnedDefinitionRecord(Protocol):
    signature: str
    definition_state: object
    target_event: str | None
    target_ir_kind: str | None
    evidence: tuple[str, ...]
    confidence: float
    extension_name: str | None


class LearnedDefinitionStore(Protocol):
    def lookup(self, unit: ContextualUnit) -> LearnedDefinitionRecord | None:
        raise NotImplementedError


@dataclass(frozen=True)
class LearnedProjection:
    signature: str
    target_event: str
    target_ir_kind: IRKind
    evidence: tuple[str, ...]
    extension_name: str | None = None


@dataclass(frozen=True)
class UYRConfig:
    confidence_threshold: float = 0.72
    max_rounds: int = 4
    max_context_units: int = 8

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("UYR confidence threshold must be in (0, 1]")
        if not 1 <= self.max_rounds <= 4:
            raise ValueError("UYR permits one through four rounds")
        if not 1 <= self.max_context_units <= 8:
            raise ValueError("UYR permits one through eight context units")


@dataclass(frozen=True)
class UYRResult:
    unit: ContextualUnit
    initial_uncertainty: float
    initial_confidence: float
    final_uncertainty: float
    final_confidence: float
    rounds: int
    expanded_context: tuple[ContextualUnit, ...] = field(default_factory=tuple)
    retrievals: tuple[str, ...] = field(default_factory=tuple)
    scores: tuple[UncertaintyScore, ...] = field(default_factory=tuple)
    learned_projection: LearnedProjection | None = None

    def projected_unit(self) -> ContextualUnit:
        projection = self.learned_projection
        if projection is None:
            return self.unit
        metadata = dict(self.unit.metadata)
        metadata["learned_definition_signature"] = projection.signature
        metadata["learned_definition_state"] = "validated"
        if projection.extension_name is not None:
            metadata["learned_extension_name"] = projection.extension_name
        ir = IRNode(
            projection.target_ir_kind,
            name=projection.extension_name or self.unit.surface,
            metadata={"learned_definition_signature": projection.signature},
        )
        return replace(
            self.unit,
            event=projection.target_event,
            ir=ir,
            metadata=metadata,
        )


class UYRResolver:
    """Uncertainty Yo-Yo Retrieval over local context and targeted symbol data."""

    def __init__(self, scorer: HeuristicUncertaintyScorer, config: UYRConfig | None = None) -> None:
        self.scorer = scorer
        self.config = config or UYRConfig()

    def resolve(
        self,
        unit: ContextualUnit,
        document: SourceDocument,
        all_units: list[ContextualUnit],
        *,
        target_index: int | None = None,
        learned_definitions: LearnedDefinitionStore | None = None,
    ) -> UYRResult:
        initial = self.scorer.score(unit, document.index)
        confidence = initial.confidence
        uncertainty = initial.uncertainty
        retrievals: list[str] = []
        scores = [initial]
        expanded: list[ContextualUnit] = []
        rounds = 0
        learned_projection = None

        learned = learned_definitions.lookup(unit) if learned_definitions is not None else None
        if learned is not None:
            state = getattr(learned.definition_state, "value", str(learned.definition_state))
            retrievals.append(f"learned-definition:{state}:{learned.signature}")
            retrievals.extend(f"learned-evidence:{item}" for item in learned.evidence)
            if (
                state == "validated"
                and learned.target_event is not None
                and learned.target_ir_kind is not None
            ):
                learned_projection = LearnedProjection(
                    signature=learned.signature,
                    target_event=learned.target_event,
                    target_ir_kind=IRKind(learned.target_ir_kind),
                    evidence=learned.evidence,
                    extension_name=learned.extension_name,
                )
                confidence = max(
                    confidence,
                    self.config.confidence_threshold,
                    learned.confidence,
                )
                uncertainty = 1.0 - confidence
            else:
                confidence = min(0.71, max(confidence, learned.confidence))
                uncertainty = 1.0 - confidence

        if target_index is None:
            target_index = all_units.index(unit)
        elif not 0 <= target_index < len(all_units) or all_units[target_index] is not unit:
            raise ValueError("UYR target index must identify the exact unit instance")
        window = 2
        while (
            confidence < self.config.confidence_threshold
            and rounds < self.config.max_rounds
            and len(expanded) < self.config.max_context_units
        ):
            rounds += 1
            start = max(0, target_index - window)
            end = min(len(all_units), target_index + window + 1)
            for candidate in all_units[start:end]:
                if candidate not in expanded:
                    expanded.append(candidate)
                if len(expanded) >= self.config.max_context_units:
                    break

            if unit.defines:
                record = document.index.definitions.get(unit.defines)
                if record:
                    retrieval = f"definition:{record.symbol_id}@{record.path}:{record.span[0]}"
                    if retrieval not in retrievals:
                        retrievals.append(retrieval)
                        confidence += 0.2
            for prior in reversed(expanded):
                if prior.event == "DEFINE" and prior.surface == unit.surface:
                    retrieval = f"local-definition:{prior.symbol_id}@line:{prior.span[0]}"
                    if retrieval not in retrievals:
                        retrievals.append(retrieval)
                        confidence += 0.15

            confidence = min(1.0, confidence)
            uncertainty = max(0.0, 1.0 - confidence)
            scores.append(
                UncertaintyScore(
                    uncertainty=uncertainty,
                    confidence=confidence,
                    signals=("uyr expanded context",),
                )
            )
            window += 2

        return UYRResult(
            unit=unit,
            initial_uncertainty=initial.uncertainty,
            initial_confidence=initial.confidence,
            final_uncertainty=uncertainty,
            final_confidence=confidence,
            rounds=rounds,
            expanded_context=tuple(expanded[: self.config.max_context_units]),
            retrievals=tuple(retrievals),
            scores=tuple(scores),
            learned_projection=learned_projection,
        )
