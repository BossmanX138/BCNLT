from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from .core import ContextualUnit, SourceDocument
from .extended_core import (
    CommonCoreProjection,
    extended_core_id,
    optional_common_core_projection,
)
from .ir import IRKind, IRNode
from .uncertainty import HeuristicUncertaintyScorer, UncertaintyScore


class LearnedDefinitionRecord(Protocol):
    @property
    def signature(self) -> str: ...

    @property
    def definition_state(self) -> object: ...

    @property
    def target_event(self) -> str | None: ...

    @property
    def target_ir_kind(self) -> str | None: ...

    @property
    def evidence(self) -> tuple[str, ...]: ...

    @property
    def confidence(self) -> float: ...

    @property
    def extension_name(self) -> str | None: ...

    @property
    def language(self) -> str: ...

    @property
    def semantic_spec_sha256(self) -> str: ...

    @property
    def hypothesis_revision(self) -> int: ...


class LearnedDefinitionStore(Protocol):
    def lookup(self, unit: ContextualUnit) -> LearnedDefinitionRecord | None:
        raise NotImplementedError


@dataclass(frozen=True)
class ExtendedCoreResolution:
    extended_core_id: str
    signature: str
    revision: int
    common_core_projection: CommonCoreProjection | None
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
    extended_core_resolution: ExtendedCoreResolution | None = None

    def resolved_unit(self) -> ContextualUnit:
        resolution = self.extended_core_resolution
        if resolution is None:
            return self.unit
        metadata = dict(self.unit.metadata)
        metadata["extended_core_id"] = resolution.extended_core_id
        metadata["extended_core_signature"] = resolution.signature
        metadata["extended_core_revision"] = resolution.revision
        metadata["extended_core_state"] = "validated"
        projection = resolution.common_core_projection
        if projection is not None:
            metadata["extended_core_projection"] = projection.to_dict()
        if resolution.extension_name is not None:
            metadata["extended_core_name"] = resolution.extension_name
        ir_metadata: dict[str, object] = {
            "extended_core_id": resolution.extended_core_id
        }
        if projection is not None:
            ir_metadata["common_core_projection"] = projection.to_dict()
        ir = IRNode(
            IRKind.EXTENSION,
            name=resolution.extension_name or resolution.extended_core_id,
            metadata=ir_metadata,
        )
        return replace(
            self.unit,
            ir=ir,
            metadata=metadata,
        )

    def projected_unit(self) -> ContextualUnit:
        """Compatibility alias; Extended Core resolution never reclassifies the event."""

        return self.resolved_unit()


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
        learned_definitions: LearnedDefinitionStore | None = None,
        target_index: int | None = None,
    ) -> UYRResult:
        initial = self.scorer.score(unit, document.index)
        confidence = initial.confidence
        uncertainty = initial.uncertainty
        retrievals: list[str] = []
        scores = [initial]
        expanded: list[ContextualUnit] = []
        rounds = 0
        extended_core_resolution = None

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
                extended_core_resolution = ExtendedCoreResolution(
                    extended_core_id=extended_core_id(
                        language=learned.language,
                        signature=learned.signature,
                        revision=learned.hypothesis_revision,
                    ),
                    signature=learned.signature,
                    revision=learned.hypothesis_revision,
                    common_core_projection=optional_common_core_projection(
                        event=learned.target_event,
                        ir_kind=learned.target_ir_kind,
                    ),
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
            # Compatibility path for external callers. The preprocessing hot
            # path always supplies the enumerated index and therefore remains
            # linear in the number of semantic units.
            target_index = next(
                (index for index, candidate in enumerate(all_units) if candidate is unit),
                None,
            )
            if target_index is None:
                raise ValueError("supplied unit is absent from all_units by identity")
        elif (
            type(target_index) is not int
            or not 0 <= target_index < len(all_units)
            or all_units[target_index] is not unit
        ):
            raise ValueError("target_index does not identify the supplied unit")
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
            extended_core_resolution=extended_core_resolution,
        )
