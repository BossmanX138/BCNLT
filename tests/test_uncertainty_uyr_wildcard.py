from bcnlt.adapters.python import PythonAdapter
from bcnlt.uncertainty import HeuristicUncertaintyScorer
from bcnlt.uyr import UYRConfig, UYRResolver
import json
from pathlib import Path

from bcnlt.wildcard import WildcardMemory, WildcardResolver


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
        config=UYRConfig(confidence_threshold=0.75, max_rounds=3, max_context_units=12),
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
