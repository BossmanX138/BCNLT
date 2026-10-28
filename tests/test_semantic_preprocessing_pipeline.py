from pathlib import Path

from bcnlt.cache import ProjectIndex
from bcnlt.core import ContextualUnit, SourceDocument
from bcnlt.ir import IRKind
from bcnlt.preprocessing import SemanticPreprocessor
from bcnlt.semantic_schema import KnowledgeState
from bcnlt.uncertainty import HeuristicUncertaintyScorer
from bcnlt.wildcard import WildcardMemory, WildcardResolver


class CountingScorer(HeuristicUncertaintyScorer):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def score(self, unit, index):
        self.calls += 1
        return super().score(unit, index)


def _document(unit: ContextualUnit, path: str = "sample.py") -> SourceDocument:
    return SourceDocument(
        path,
        unit.surface,
        None,
        ProjectIndex(),
        language="python",
    )


def _unknown() -> ContextualUnit:
    return ContextualUnit(
        surface="match value",
        event="UNKNOWN_SEMANTIC",
        scope="module",
        span=(1, 0, 1, 11),
        ast_relation="match_statement.self",
        metadata={"unknown_node": "match_statement"},
    )


def test_preprocessing_is_the_single_uncertainty_and_admission_pipeline(tmp_path: Path):
    unit = _unknown()
    scorer = CountingScorer()
    result = SemanticPreprocessor(memory_root=tmp_path, scorer=scorer).process(
        _document(unit),
        [unit],
        semantic_spec_sha256="a" * 64,
    )

    assert scorer.calls == 1
    assert len(result.units) == len(result.assessments) == len(result.records) == 1
    assert result.assessments[0].state is KnowledgeState.PROVISIONAL
    assert result.records[0].confidence_band.value == "medium"
    assert dict(result.records[0].metadata)["knowledge_state"] == "provisional"


def test_repetition_in_preprocessing_does_not_validate_a_wildcard(tmp_path: Path):
    unit = _unknown()
    preprocessor = SemanticPreprocessor(memory_root=tmp_path)
    for _ in range(4):
        result = preprocessor.process(
            _document(unit),
            [unit],
            semantic_spec_sha256="b" * 64,
        )

    assert result.assessments[0].state is KnowledgeState.PROVISIONAL
    assert result.assessments[0].final_score.confidence < 0.72
    assert result.units[0].event == "UNKNOWN_SEMANTIC"


def test_validated_wildcard_is_injected_before_canonical_admission(tmp_path: Path):
    unit = _unknown()
    memory = WildcardMemory(
        tmp_path / "python.jsonl",
        language="python",
        semantic_spec_sha256="c" * 64,
    )
    WildcardResolver(memory).resolve_and_remember(unit, ProjectIndex())
    memory.validate(
        unit,
        target_event="BRANCH",
        target_ir_kind=IRKind.BRANCH.value,
        evidence=("reviewed match fixture",),
    )

    result = SemanticPreprocessor(memory_root=tmp_path).process(
        _document(unit),
        [unit],
        semantic_spec_sha256="c" * 64,
    )

    assert result.assessments[0].state is KnowledgeState.RETRIEVED
    assert result.units[0].event == "BRANCH"
    assert result.records[0].event == "BRANCH"
    assert result.records[0].confidence_band.value == "high"
    metadata = dict(result.records[0].metadata)
    assert metadata["learned_definition_state"] == "validated"
    assert metadata["learned_definition_signature"]


def test_preprocessing_learns_from_independent_context_and_reuses_it(tmp_path: Path):
    first = _unknown()
    second = ContextualUnit(
        surface=first.surface,
        event=first.event,
        scope="module.worker",
        span=(12, 4, 12, 15),
        ast_relation=first.ast_relation,
        recent_history=("DEFINE:worker", "REFERENCE:value"),
        metadata=first.metadata,
    )
    preprocessor = SemanticPreprocessor(memory_root=tmp_path)

    preprocessor.process(
        _document(first, "first.py"), [first], semantic_spec_sha256="d" * 64
    )
    learned = preprocessor.process(
        _document(second, "second.py"), [second], semantic_spec_sha256="d" * 64
    )
    reused = preprocessor.process(
        _document(second, "second.py"), [second], semantic_spec_sha256="d" * 64
    )

    assert learned.assessments[0].state is KnowledgeState.PROVISIONAL
    assert reused.assessments[0].state is KnowledgeState.RETRIEVED
    assert reused.records[0].event == "BRANCH"
