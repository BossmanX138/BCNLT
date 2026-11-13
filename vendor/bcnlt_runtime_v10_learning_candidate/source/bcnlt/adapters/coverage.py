from __future__ import annotations

from dataclasses import dataclass

from bcnlt.core import ContextualUnit, SourceDocument
from bcnlt.ir import IRKind
from bcnlt.uncertainty import KnowledgeBoundaryAssessment


def is_unresolved_unknown(unit: ContextualUnit) -> bool:
    """Return whether a unit is still outside both Common and Extended Core."""

    if unit.event != "UNKNOWN_SEMANTIC":
        return False
    is_extension = unit.ir is not None and unit.ir.kind is IRKind.EXTENSION
    has_validated_extended_identity = (
        unit.metadata.get("extended_core_state") == "validated"
        and isinstance(unit.metadata.get("extended_core_id"), str)
        and bool(unit.metadata["extended_core_id"])
    )
    has_legacy_validated_identity = (
        unit.metadata.get("learned_definition_state") == "validated"
    )
    return not (
        is_extension
        and (has_validated_extended_identity or has_legacy_validated_identity)
    )


@dataclass(frozen=True, slots=True)
class SemanticCoverage:
    total_units: int
    common_core_units: int
    unknown_units: int
    total_named_nodes: int = 0
    excluded_named_nodes: int = 0
    mapped_named_nodes: int = 0
    unknown_named_nodes: int = 0
    scopes_balanced: bool = True
    links_valid: bool = True

    def __post_init__(self) -> None:
        counts = (
            self.total_units,
            self.common_core_units,
            self.unknown_units,
            self.total_named_nodes,
            self.excluded_named_nodes,
            self.mapped_named_nodes,
            self.unknown_named_nodes,
        )
        if min(counts) < 0:
            raise ValueError("Semantic coverage counts cannot be negative")
        if self.common_core_units + self.unknown_units > self.total_units:
            raise ValueError("Semantic coverage counts exceed total units")

    @property
    def ratio(self) -> float:
        return self.common_core_units / self.total_units if self.total_units else 1.0

    @property
    def mapped_ratio(self) -> float:
        eligible = self.total_named_nodes - self.excluded_named_nodes
        return self.mapped_named_nodes / eligible if eligible else 1.0

    @property
    def unknown_ratio(self) -> float:
        return self.unknown_units / self.total_units if self.total_units else 0.0

    @property
    def common_core_ratio(self) -> float:
        eligible = self.total_units - self.unknown_units
        return self.common_core_units / eligible if eligible else 1.0

    @property
    def meets_common_core_gate(self) -> bool:
        return self.common_core_ratio >= 0.80

    @property
    def admitted(self) -> bool:
        return (
            self.mapped_ratio >= 0.85
            and self.unknown_ratio <= 0.10
            and self.scopes_balanced
            and self.links_valid
        )


@dataclass(frozen=True, slots=True)
class SemanticExtraction:
    schema_version: str
    language: str
    dialect: str | None
    document: SourceDocument
    units: tuple[ContextualUnit, ...]
    coverage: SemanticCoverage
    assessments: tuple[KnowledgeBoundaryAssessment, ...] = ()
