from __future__ import annotations

from dataclasses import dataclass

from bcnlt.core import ContextualUnit, SourceDocument


@dataclass(frozen=True, slots=True)
class SemanticCoverage:
    total_units: int
    common_core_units: int
    unknown_units: int

    def __post_init__(self) -> None:
        if min(self.total_units, self.common_core_units, self.unknown_units) < 0:
            raise ValueError("Semantic coverage counts cannot be negative")
        if self.common_core_units + self.unknown_units > self.total_units:
            raise ValueError("Semantic coverage counts exceed total units")

    @property
    def ratio(self) -> float:
        return self.common_core_units / self.total_units if self.total_units else 1.0


@dataclass(frozen=True, slots=True)
class SemanticExtraction:
    schema_version: str
    language: str
    dialect: str | None
    document: SourceDocument
    units: tuple[ContextualUnit, ...]
    coverage: SemanticCoverage
