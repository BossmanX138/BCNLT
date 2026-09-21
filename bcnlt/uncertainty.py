from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .cache import ProjectIndex
from .core import ContextualUnit


@dataclass(frozen=True)
class UncertaintyScore:
    uncertainty: float
    confidence: float
    signals: tuple[str, ...]


class EntropyProvider(Protocol):
    def entropy_for(self, unit: ContextualUnit) -> float:
        raise NotImplementedError


class StubEntropyProvider:
    def entropy_for(self, unit: ContextualUnit) -> float:
        return 0.0


class HeuristicUncertaintyScorer:
    def __init__(self, entropy_provider: EntropyProvider | None = None) -> None:
        self.entropy_provider = entropy_provider or StubEntropyProvider()

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
        if unit.event == "MEMBER_ACCESS":
            uncertainty += 0.1
            signals.append("member type unknown")
        if unit.ast_relation is None:
            uncertainty += 0.1
            signals.append("missing AST relation")

        entropy = self.entropy_provider.entropy_for(unit)
        if entropy:
            uncertainty += min(entropy, 1.0) * 0.2
            signals.append("model entropy")

        uncertainty = min(1.0, uncertainty)
        return UncertaintyScore(
            uncertainty=uncertainty,
            confidence=max(0.0, 1.0 - uncertainty),
            signals=tuple(signals),
        )
