from __future__ import annotations

from dataclasses import dataclass, field

from .core import ContextualUnit, SourceDocument
from .uncertainty import HeuristicUncertaintyScorer, UncertaintyScore


@dataclass(frozen=True)
class UYRConfig:
    confidence_threshold: float = 0.72
    max_rounds: int = 4
    max_context_units: int = 8


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
    ) -> UYRResult:
        initial = self.scorer.score(unit, document.index)
        confidence = initial.confidence
        uncertainty = initial.uncertainty
        retrievals: list[str] = []
        scores = [initial]
        expanded: list[ContextualUnit] = []
        rounds = 0

        target_index = all_units.index(unit)
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

            if expanded:
                confidence += 0.05
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
        )
