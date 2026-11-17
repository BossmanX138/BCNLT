from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .core import ContextualUnit, SourceDocument
from .semantic_record import SemanticTokenRecord, records_from_assessments
from .semantic_schema import KnowledgeState
from .uncertainty import (
    HeuristicUncertaintyScorer,
    KnowledgeBoundaryAssessment,
    UncertaintyScore,
)
from .uyr import UYRConfig, UYRResolver
from .wildcard import WildcardMemory, WildcardResolver


@dataclass(frozen=True)
class SemanticPreprocessingResult:
    document: SourceDocument
    units: tuple[ContextualUnit, ...]
    assessments: tuple[KnowledgeBoundaryAssessment, ...]
    records: tuple[SemanticTokenRecord, ...]


class SemanticPreprocessor:
    """Single knowledge-boundary pass before canonical training-data admission."""

    def __init__(
        self,
        *,
        memory_root: str | Path | None = None,
        memory_path: str | Path | None = None,
        memory_storage: Literal["jsonl", "sqlite"] = "jsonl",
        memory_read_only: bool = False,
        scorer: HeuristicUncertaintyScorer | None = None,
        uyr_config: UYRConfig | None = None,
    ) -> None:
        self.memory_root = Path(memory_root) if memory_root is not None else None
        self.memory_path = Path(memory_path) if memory_path is not None else None
        self.memory_storage = memory_storage
        self.memory_read_only = memory_read_only
        if self.memory_root is not None and self.memory_path is not None:
            raise ValueError("Choose either a learned-memory root or an exact memory path")
        self.scorer = scorer or HeuristicUncertaintyScorer()
        self.uyr = UYRResolver(self.scorer, uyr_config)
        self._memories: dict[tuple[str, str, str | None], WildcardMemory] = {}

    def process(
        self,
        document: SourceDocument,
        units: Iterable[ContextualUnit],
        *,
        semantic_spec_sha256: str,
    ) -> SemanticPreprocessingResult:
        raw_units = list(units)
        memory_path = self.memory_path
        if memory_path is None and self.memory_root is not None:
            suffix = "sqlite3" if self.memory_storage == "sqlite" else "jsonl"
            memory_path = self.memory_root / f"{document.language}.{suffix}"
        memory_key = (
            document.language,
            semantic_spec_sha256,
            str(memory_path.resolve()) if memory_path is not None else None,
        )
        memory = self._memories.get(memory_key)
        if memory is None:
            memory = WildcardMemory(
                memory_path,
                language=document.language,
                semantic_spec_sha256=semantic_spec_sha256,
                storage=self.memory_storage,
                read_only=self.memory_read_only,
            )
            self._memories[memory_key] = memory
        wildcard = WildcardResolver(memory)
        assessments = tuple(
            self._assess(document, index, unit, raw_units, wildcard)
            for index, unit in enumerate(raw_units)
        )
        annotated = tuple(assessment.annotated_unit() for assessment in assessments)
        records = records_from_assessments(assessments)
        return SemanticPreprocessingResult(
            document=document,
            units=annotated,
            assessments=assessments,
            records=records,
        )

    def _assess(
        self,
        document: SourceDocument,
        unit_index: int,
        unit: ContextualUnit,
        all_units: list[ContextualUnit],
        wildcard: WildcardResolver,
    ) -> KnowledgeBoundaryAssessment:
        result = self.uyr.resolve(
            unit,
            document,
            all_units,
            target_index=unit_index,
            learned_definitions=wildcard.memory,
        )
        initial = result.scores[0]
        final = UncertaintyScore(
            uncertainty=result.final_uncertainty,
            confidence=result.final_confidence,
            signals=tuple(
                dict.fromkeys(
                    (
                        *result.scores[-1].signals,
                        *(("uyr retrieval",) if result.retrievals else ()),
                    )
                )
            ),
        )
        if result.extended_core_resolution is not None:
            return KnowledgeBoundaryAssessment(
                unit=result.resolved_unit(),
                initial_score=initial,
                final_score=final,
                state=KnowledgeState.RETRIEVED,
                rounds=result.rounds,
                evidence=result.retrievals,
            )
        if initial.confidence >= self.uyr.config.confidence_threshold:
            return KnowledgeBoundaryAssessment(
                unit=unit,
                initial_score=initial,
                final_score=final,
                state=KnowledgeState.KNOWN,
                rounds=result.rounds,
                evidence=result.retrievals,
            )
        if result.retrievals and final.confidence >= self.uyr.config.confidence_threshold:
            return KnowledgeBoundaryAssessment(
                unit=unit,
                initial_score=initial,
                final_score=final,
                state=KnowledgeState.RETRIEVED,
                rounds=result.rounds,
                evidence=result.retrievals,
            )

        if unit.event != "UNKNOWN_SEMANTIC":
            return KnowledgeBoundaryAssessment(
                unit=unit,
                initial_score=initial,
                final_score=final,
                state=KnowledgeState.UNRESOLVED,
                rounds=result.rounds,
                evidence=result.retrievals,
            )

        if self.memory_read_only:
            meaning = wildcard.resolve(unit, document.index)
        else:
            meaning = wildcard.resolve_and_remember(
                unit,
                document.index,
                document_identity=document.path,
            )
        assessment = KnowledgeBoundaryAssessment.from_wildcard(
            unit=unit,
            initial=initial,
            meaning=meaning,
            rounds=result.rounds,
        )
        confidence = max(final.confidence, assessment.final_score.confidence)
        assessment = replace(
            assessment,
            final_score=UncertaintyScore(
                uncertainty=1.0 - confidence,
                confidence=confidence,
                signals=tuple(
                    dict.fromkeys((*final.signals, *assessment.final_score.signals))
                ),
            ),
            evidence=tuple(dict.fromkeys((*result.retrievals, *assessment.evidence))),
        )
        if not meaning.hypotheses or not meaning.evidence:
            assessment = replace(assessment, state=KnowledgeState.UNRESOLVED)
        return assessment

    def close(self) -> None:
        for memory in self._memories.values():
            memory.close()


def process_python_source(source: str, path: str = "<memory>") -> list[ContextualUnit]:
    return process_source(source, language="python", path=path)


def process_source(
    source: str,
    *,
    language: str,
    path: str = "<memory>",
    dialect: str | None = None,
) -> list[ContextualUnit]:
    from .adapters.carousel import BCNLTMultilingualAdapter
    from .adapters.tree_sitter import TreeSitterSemanticEngine

    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse(source, language=language, path=path, dialect=dialect)
    return list(adapter.extract(document).units)


def process_python_files(paths: Iterable[str], workers: int = 1) -> dict[str, list[ContextualUnit]]:
    path_list = list(paths)

    def process(path: str) -> tuple[str, list[ContextualUnit]]:
        source = Path(path).read_text(encoding="utf-8")
        return path, process_python_source(source, path=path)

    if workers <= 1:
        return dict(process(path) for path in path_list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(process, path_list))
