from pathlib import Path

import pytest

from bcnlt.cache import ProjectIndex
from bcnlt.core import ContextualUnit, SourceDocument
from bcnlt.ir import IRKind
from bcnlt.uncertainty import HeuristicUncertaintyScorer
from bcnlt.uyr import UYRResolver
from bcnlt.wildcard import DefinitionState, WildcardMemory, WildcardResolver


def _unknown() -> ContextualUnit:
    return ContextualUnit(
        surface="match value",
        event="UNKNOWN_SEMANTIC",
        scope="module",
        span=(1, 0, 1, 11),
        ast_relation="match_statement.self",
        metadata={"unknown_node": "match_statement"},
    )


def test_wildcard_builds_a_provisional_definition_as_it_observes_an_outlier(tmp_path: Path):
    unit = _unknown()
    memory = WildcardMemory(
        tmp_path / "learned.jsonl",
        language="python",
        semantic_spec_sha256="a" * 64,
    )
    resolver = WildcardResolver(memory)

    meaning = resolver.resolve_and_remember(unit, ProjectIndex())
    record = memory.lookup(unit)

    assert meaning.target_event == "BRANCH"
    assert meaning.target_ir_kind == IRKind.BRANCH.value
    assert record is not None
    assert record.definition_state is DefinitionState.PROVISIONAL
    assert record.target_event == "BRANCH"
    assert record.target_ir_kind == IRKind.BRANCH.value
    assert record.evidence


def test_uyr_retrieves_provisional_learning_without_turning_repetition_into_fact(tmp_path: Path):
    unit = _unknown()
    memory = WildcardMemory(
        tmp_path / "learned.jsonl",
        language="python",
        semantic_spec_sha256="b" * 64,
    )
    wildcard = WildcardResolver(memory)
    for _ in range(4):
        wildcard.resolve_and_remember(unit, ProjectIndex())
    document = SourceDocument("sample.py", unit.surface, None, ProjectIndex(), language="python")

    result = UYRResolver(HeuristicUncertaintyScorer()).resolve(
        unit,
        document,
        [unit],
        learned_definitions=memory,
    )

    assert any(item.startswith("learned-definition:provisional:") for item in result.retrievals)
    assert result.learned_projection is None
    assert result.final_confidence < 0.72
    assert result.projected_unit() == unit


def test_validated_learning_is_retrieved_and_injected_into_the_common_core(tmp_path: Path):
    unit = _unknown()
    path = tmp_path / "learned.jsonl"
    memory = WildcardMemory(path, language="python", semantic_spec_sha256="c" * 64)
    WildcardResolver(memory).resolve_and_remember(unit, ProjectIndex())
    validated = memory.validate(
        unit,
        target_event="BRANCH",
        target_ir_kind=IRKind.BRANCH.value,
        evidence=("reviewed golden match fixture",),
    )
    reloaded = WildcardMemory(path, language="python", semantic_spec_sha256="c" * 64)
    document = SourceDocument("sample.py", unit.surface, None, ProjectIndex(), language="python")

    result = UYRResolver(HeuristicUncertaintyScorer()).resolve(
        unit,
        document,
        [unit],
        learned_definitions=reloaded,
    )
    projected = result.projected_unit()

    assert validated.definition_state is DefinitionState.VALIDATED
    assert result.learned_projection is not None
    assert result.final_confidence >= 0.72
    assert projected.event == "BRANCH"
    assert projected.ir is not None and projected.ir.kind is IRKind.BRANCH
    assert projected.metadata["learned_definition_signature"] == validated.signature
    assert any("reviewed golden match fixture" in item for item in result.retrievals)


def test_learned_definitions_are_isolated_by_language_and_spec(tmp_path: Path):
    unit = _unknown()
    path = tmp_path / "learned.jsonl"
    python = WildcardMemory(path, language="python", semantic_spec_sha256="d" * 64)
    WildcardResolver(python).resolve_and_remember(unit, ProjectIndex())

    assert WildcardMemory(path, language="rust", semantic_spec_sha256="d" * 64).lookup(unit) is None
    assert WildcardMemory(path, language="python", semantic_spec_sha256="e" * 64).lookup(unit) is None


def test_independent_context_evidence_promotes_a_definition_without_human_review(tmp_path: Path):
    first = _unknown()
    second = ContextualUnit(
        surface=first.surface,
        event=first.event,
        scope="module.worker",
        span=(20, 4, 20, 15),
        ast_relation=first.ast_relation,
        recent_history=("DEFINE:worker", "REFERENCE:value"),
        metadata=first.metadata,
    )
    memory = WildcardMemory(
        tmp_path / "learned.jsonl",
        language="python",
        semantic_spec_sha256="f" * 64,
    )
    resolver = WildcardResolver(memory)

    resolver.resolve_and_remember(
        first, ProjectIndex(), document_identity="first.py"
    )
    resolver.resolve_and_remember(
        second, ProjectIndex(), document_identity="second.py"
    )
    learned = memory.lookup(second)

    assert learned is not None
    assert learned.definition_state is DefinitionState.VALIDATED
    assert learned.validated_by == "autonomous"
    assert len(learned.context_fingerprints) == 2
    assert {"grammar_structure", "lexical_surface", "context_agreement"} <= set(
        learned.evidence_kinds
    )


def test_wildcard_induces_targets_for_unseen_grammar_outliers(tmp_path: Path):
    unit = ContextualUnit(
        surface="novel switch form",
        event="UNKNOWN_SEMANTIC",
        scope="module",
        span=(1, 0, 1, 17),
        ast_relation="switch_expression.self",
    )
    memory = WildcardMemory(
        tmp_path / "learned.jsonl",
        language="kotlin",
        semantic_spec_sha256="1" * 64,
    )

    meaning = WildcardResolver(memory).resolve(unit, ProjectIndex())

    assert meaning.target_event == "BRANCH"
    assert meaning.target_ir_kind == IRKind.BRANCH.value
    assert meaning.evidence_kinds


def test_validation_rejects_an_event_ir_pair_outside_the_common_catalog(tmp_path: Path):
    unit = _unknown()
    memory = WildcardMemory(tmp_path / "learned.jsonl")
    WildcardResolver(memory).resolve_and_remember(unit, ProjectIndex())

    with pytest.raises(ValueError, match="one Common Core concept"):
        memory.validate(
            unit,
            target_event="CALL",
            target_ir_kind=IRKind.BRANCH.value,
            evidence=("independent execution trace",),
        )


def test_repeated_language_oddity_becomes_a_named_extension_and_is_reused(tmp_path: Path):
    memory = WildcardMemory(
        tmp_path / "learned.jsonl",
        language="kotlin",
        semantic_spec_sha256="2" * 64,
    )
    resolver = WildcardResolver(memory)
    units = [
        ContextualUnit(
            surface="mystery form",
            event="UNKNOWN_SEMANTIC",
            scope=f"module.context_{index}",
            span=(index * 10, 0, index * 10, 12),
            ast_relation="mystery_construct.self",
            recent_history=(f"DEFINE:context_{index}",),
        )
        for index in (1, 2, 3)
    ]
    for index, unit in enumerate(units, start=1):
        resolver.resolve_and_remember(
            unit,
            ProjectIndex(),
            document_identity=f"sample-{index}.kt",
        )

    learned = memory.lookup(units[-1])
    document = SourceDocument("sample.kt", units[-1].surface, None, ProjectIndex(), language="kotlin")
    result = UYRResolver(HeuristicUncertaintyScorer()).resolve(
        units[-1],
        document,
        [units[-1]],
        learned_definitions=memory,
    )
    projected = result.projected_unit()

    assert learned is not None
    assert learned.definition_state is DefinitionState.VALIDATED
    assert learned.target_ir_kind == IRKind.EXTENSION.value
    assert learned.extension_name == "kotlin.mystery_construct"
    assert projected.ir is not None and projected.ir.kind is IRKind.EXTENSION
    assert projected.ir.name == "kotlin.mystery_construct"
    assert projected.metadata["learned_extension_name"] == "kotlin.mystery_construct"


def test_truncated_final_append_record_does_not_destroy_prior_learning(tmp_path: Path):
    unit = _unknown()
    path = tmp_path / "learned.jsonl"
    memory = WildcardMemory(path, language="python", semantic_spec_sha256="3" * 64)
    WildcardResolver(memory).resolve_and_remember(
        unit, ProjectIndex(), document_identity="first.py"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"signature":"truncated"')

    reloaded = WildcardMemory(path, language="python", semantic_spec_sha256="3" * 64)

    assert reloaded.lookup(unit) is not None
    assert reloaded.load_warnings


def test_malformed_middle_append_record_is_rejected(tmp_path: Path):
    unit = _unknown()
    path = tmp_path / "learned.jsonl"
    memory = WildcardMemory(path, language="python", semantic_spec_sha256="4" * 64)
    WildcardResolver(memory).resolve_and_remember(
        unit, ProjectIndex(), document_identity="first.py"
    )
    valid = path.read_text(encoding="utf-8")
    path.write_text(valid + "{broken}\n" + valid, encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        WildcardMemory(path, language="python", semantic_spec_sha256="4" * 64)
