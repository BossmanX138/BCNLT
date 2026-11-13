from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
from typing import Protocol

from .cache import ProjectIndex
from .core import ContextualUnit
from .semantic_schema import KnowledgeState


@dataclass(frozen=True)
class UncertaintyScore:
    uncertainty: float
    confidence: float
    signals: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeBoundaryAssessment:
    unit: ContextualUnit
    initial_score: UncertaintyScore
    final_score: UncertaintyScore
    state: KnowledgeState
    rounds: int
    evidence: tuple[str, ...]
    wildcard_category: str | None = None

    @classmethod
    def from_wildcard(
        cls,
        *,
        unit: ContextualUnit,
        initial: UncertaintyScore,
        meaning,
        rounds: int,
    ) -> "KnowledgeBoundaryAssessment":
        final = UncertaintyScore(
            uncertainty=max(0.0, 1.0 - meaning.confidence),
            confidence=meaning.confidence,
            signals=("wildcard provisional",),
        )
        return cls(
            unit=unit,
            initial_score=initial,
            final_score=final,
            state=KnowledgeState.PROVISIONAL,
            rounds=rounds,
            evidence=tuple(meaning.evidence),
            wildcard_category=meaning.category,
        )

    def annotated_unit(self) -> ContextualUnit:
        metadata = dict(self.unit.metadata)
        metadata.update(
            {
                "knowledge_state": self.state.value,
                "uyr_rounds": self.rounds,
                "uncertainty_signals": list(self.initial_score.signals),
            }
        )
        if self.wildcard_category is not None:
            metadata["wildcard_category"] = self.wildcard_category
        return replace(self.unit, metadata=metadata)


class EntropyProvider(Protocol):
    def entropy_for(self, unit: ContextualUnit) -> float:
        ...


class HeuristicUncertaintyScorer:
    def __init__(self, entropy_provider: EntropyProvider | None = None) -> None:
        self.entropy_provider = entropy_provider

    def score(self, unit: ContextualUnit, index: ProjectIndex) -> UncertaintyScore:
        signals: list[str] = []
        uncertainty = 0.05

        if unit.event == "UNKNOWN_SEMANTIC":
            uncertainty += 0.55
            signals.append("unknown semantic")
        if unit.event in {"REFERENCE", "CALL"} and unit.defines is None:
            builtins = {"print", "len", "range", "getattr", "setattr", "eval", "exec"}
            if unit.surface not in builtins:
                uncertainty += 0.25
                signals.append("missing definition")
        if unit.event == "CALL" and unit.metadata.get("dynamic_call"):
            uncertainty += 0.35
            signals.append("dynamic call target")
        if unit.event in {"REFERENCE", "CALL"} and unit.metadata.get("dynamic_binding"):
            uncertainty += 0.3
            signals.append("dynamic binding")
        if unit.metadata.get("macro_expansion"):
            uncertainty += 0.35
            signals.append("macro expansion")
        if unit.metadata.get("overload_unresolved"):
            uncertainty += 0.25
            signals.append("overload unresolved")
        if unit.metadata.get("preprocessor_condition"):
            uncertainty += 0.2
            signals.append("preprocessor condition")
        if unit.event == "MEMBER_ACCESS":
            uncertainty += 0.1
            signals.append("member type unknown")
        if unit.ast_relation is None:
            uncertainty += 0.1
            signals.append("missing AST relation")

        if self.entropy_provider is not None:
            entropy = self.entropy_provider.entropy_for(unit)
            if (
                isinstance(entropy, bool)
                or not isinstance(entropy, (int, float))
                or not math.isfinite(float(entropy))
                or not 0.0 <= float(entropy) <= 1.0
            ):
                raise ValueError("model entropy must be finite and within [0, 1]")
            if entropy > 0.0:
                uncertainty += float(entropy) * 0.2
                signals.append("model entropy")

        uncertainty = min(1.0, uncertainty)
        return UncertaintyScore(
            uncertainty=uncertainty,
            confidence=max(0.0, 1.0 - uncertainty),
            signals=tuple(signals),
        )
