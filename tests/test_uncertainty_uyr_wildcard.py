from bcnlt.adapters.python import PythonAdapter
from bcnlt.uncertainty import HeuristicUncertaintyScorer
from bcnlt.uyr import UYRConfig, UYRResolver
import json
from pathlib import Path

from bcnlt.wildcard import WildcardMemory, WildcardResolver
from bcnlt.cache import ProjectIndex
from bcnlt.core import ContextualUnit, SourceDocument
from bcnlt.uncertainty import KnowledgeBoundaryAssessment, KnowledgeState


AMBIGUOUS = """
def run(plugin, payload):
    handler = getattr(plugin, payload["name"])
    return handler(payload)

del payload
"""


def test_uncertainty_scores_missing_defs_and_dynamic_calls():
    adapter = PythonAdapter()
    document = adapter.parse(AMBIGUOUS, path="ambiguous.py")
    units = list(adapter.to_units(document))
    scorer = HeuristicUncertaintyScorer()

    scored = [(unit, scorer.score(unit, document.index)) for unit in units]

    dynamic_call_score = next(
        score for unit, score in scored if unit.event == "CALL" and unit.surface == "getattr"
    )
    unknown_score = next(score for unit, score in scored if unit.event == "UNKNOWN_SEMANTIC")

    assert dynamic_call_score.uncertainty >= 0.35
    assert "dynamic call target" in dynamic_call_score.signals
    assert unknown_score.uncertainty >= 0.5
    assert "unknown semantic" in unknown_score.signals


def test_uyr_expands_context_and_records_targeted_retrievals():
    adapter = PythonAdapter()
    document = adapter.parse(AMBIGUOUS, path="ambiguous.py")
    units = list(adapter.to_units(document))
    scorer = HeuristicUncertaintyScorer()
    resolver = UYRResolver(
        scorer=scorer,
        config=UYRConfig(confidence_threshold=0.75, max_rounds=3, max_context_units=8),
    )
    target = next(unit for unit in units if unit.event == "CALL" and unit.surface == "handler")

    result = resolver.resolve(target, document, units)

    assert result.rounds >= 1
    assert result.final_confidence > result.initial_confidence
    assert result.expanded_context
    assert any("definition:" in item for item in result.retrievals)


def test_wildcard_resolver_returns_hypothesis_with_evidence():
    adapter = PythonAdapter()
    document = adapter.parse(AMBIGUOUS, path="ambiguous.py")
    unknown = next(unit for unit in adapter.to_units(document) if unit.event == "UNKNOWN_SEMANTIC")
    resolver = WildcardResolver()

    provisional = resolver.resolve(unknown, document.index)

    assert provisional.category == "extension"
    assert provisional.confidence > 0
    assert provisional.hypotheses
    assert provisional.evidence


def test_wildcard_memory_appends_seen_unknowns(tmp_path: Path):
    adapter = PythonAdapter()
    document = adapter.parse(AMBIGUOUS, path="ambiguous.py")
    unknown = next(unit for unit in adapter.to_units(document) if unit.event == "UNKNOWN_SEMANTIC")
    memory_path = tmp_path / "wildcards.jsonl"
    resolver = WildcardResolver(memory=WildcardMemory(memory_path))

    provisional = resolver.resolve(unknown, document.index)
    resolver.remember(unknown, provisional)

    lines = memory_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["signature"]
    assert payload["category"] == provisional.category
    assert payload["surface"] == unknown.surface
    assert payload["seen_count"] == 1


def test_wildcard_memory_reuses_seen_unknowns(tmp_path: Path):
    adapter = PythonAdapter()
    document = adapter.parse(AMBIGUOUS, path="ambiguous.py")
    unknown = next(unit for unit in adapter.to_units(document) if unit.event == "UNKNOWN_SEMANTIC")
    memory_path = tmp_path / "wildcards.jsonl"

    first_resolver = WildcardResolver(memory=WildcardMemory(memory_path))
    first = first_resolver.resolve(unknown, document.index)
    first_resolver.remember(unknown, first)

    second_resolver = WildcardResolver(memory=WildcardMemory(memory_path))
    second = second_resolver.resolve(unknown, document.index)

    assert second.category == first.category
    assert second.confidence > first.confidence
    assert any("wildcard memory hit" in item for item in second.evidence)


def test_uyr_is_bounded_and_confidence_changes_only_with_recorded_evidence():
    index = ProjectIndex()
    symbol_id = "python:sample.py:module:handler@0"
    index.define("sample.py", symbol_id, "handler", (1, 0, 1, 7))
    unit = ContextualUnit(
        surface="handler",
        event="CALL",
        scope="module",
        span=(2, 0, 2, 7),
        defines=symbol_id,
        ast_relation="call.function",
        metadata={"dynamic_call": True},
    )
    document = SourceDocument("sample.py", "handler()", None, index, language="python")
    resolver = UYRResolver(HeuristicUncertaintyScorer())

    result = resolver.resolve(unit, document, [unit])

    assert result.rounds <= 4
    assert len(result.expanded_context) <= 8
    assert result.final_confidence > result.initial_confidence
    assert result.retrievals
    assert len(result.scores) == result.rounds + 1


def test_context_without_evidence_does_not_improve_confidence():
    unit = ContextualUnit(
        surface="mystery",
        event="CALL",
        scope="module",
        span=(1, 0, 1, 7),
        ast_relation="call.function",
        metadata={"dynamic_call": True},
    )
    document = SourceDocument("sample.py", "mystery()", None, ProjectIndex(), language="python")
    result = UYRResolver(HeuristicUncertaintyScorer()).resolve(unit, document, [unit])

    assert result.retrievals == ()
    assert result.final_confidence == result.initial_confidence


def test_wildcard_signature_includes_language_and_spec_identity(tmp_path: Path):
    unit = ContextualUnit("mystery", "UNKNOWN_SEMANTIC", "module", (1, 0, 1, 7))
    python_memory = WildcardMemory(tmp_path / "python.jsonl", language="python", semantic_spec_sha256="a" * 64)
    rust_memory = WildcardMemory(tmp_path / "rust.jsonl", language="rust", semantic_spec_sha256="b" * 64)

    assert python_memory.signature_for(unit) != rust_memory.signature_for(unit)
    assert python_memory.signature_for(unit).startswith("python|" + "a" * 64)


def test_memory_reuse_remains_provisional_even_after_repetition(tmp_path: Path):
    unit = ContextualUnit("mystery", "UNKNOWN_SEMANTIC", "module", (1, 0, 1, 7))
    memory = WildcardMemory(tmp_path / "memory.jsonl", language="python", semantic_spec_sha256="c" * 64)
    resolver = WildcardResolver(memory)
    first = resolver.resolve_and_remember(unit, ProjectIndex())
    for _ in range(5):
        later = resolver.resolve_and_remember(unit, ProjectIndex())

    assessment = KnowledgeBoundaryAssessment.from_wildcard(
        unit=unit,
        initial=HeuristicUncertaintyScorer().score(unit, ProjectIndex()),
        meaning=later,
        rounds=0,
    )

    assert later.confidence > first.confidence
    assert later.confidence < 0.72
    assert assessment.state is KnowledgeState.PROVISIONAL
    assert assessment.evidence
