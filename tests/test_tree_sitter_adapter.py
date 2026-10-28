from __future__ import annotations

from dataclasses import replace

import pytest

from bcnlt.adapters.carousel import LanguageCartridge
from bcnlt.adapters.coverage import SemanticCoverage
from bcnlt.adapters.spec import (
    IdentifierRoleRule,
    LanguageSemanticSpec,
    NodeRule,
    ScopeRule,
)
from bcnlt.adapters.symbols import LexicalSymbolTable
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine
from bcnlt.ir import CommonCodingCore, IRKind
from bcnlt.languages import LANGUAGE_BY_ID
from bcnlt.semantic_schema import KnowledgeState


def _python_cartridge() -> LanguageCartridge:
    spec = LanguageSemanticSpec(
        semantic_spec_version=LANGUAGE_BY_ID["python"].semantic_spec_version,
        node_rules=(
            NodeRule("function_definition", "DEFINE", "function", "name"),
            NodeRule("return_statement", "RETURN", "return"),
            NodeRule("call", "CALL", "call", "function"),
            NodeRule("assignment", "ASSIGN", "assignment", "left"),
            NodeRule("string", "LITERAL", "value"),
            NodeRule("integer", "LITERAL", "value"),
        ),
        identifier_roles=(
            IdentifierRoleRule("function_definition", "name", "definition"),
            IdentifierRoleRule("assignment", "left", "definition"),
            IdentifierRoleRule("call", "function", "reference"),
        ),
        scope_rules=(ScopeRule("function_definition", "function", "name"),),
        excluded_node_types=(
            "module",
            "block",
            "parameters",
            "argument_list",
            "expression_statement",
            "identifier",
        ),
    )
    return LanguageCartridge(LANGUAGE_BY_ID["python"], spec)


def test_tree_sitter_engine_emits_in_source_order_with_byte_columns_and_history():
    engine = TreeSitterSemanticEngine()
    cartridge = _python_cartridge()
    source = 'label = "😀"\nprint(label)\n'

    document = engine.parse(source, cartridge=cartridge, path="unicode.py", dialect=None)
    extraction = engine.extract(document, cartridge=cartridge)

    assert [unit.span for unit in extraction.units] == sorted(
        (unit.span for unit in extraction.units), key=lambda span: (span[0], span[1], span[2], span[3])
    )
    emoji = next(unit for unit in extraction.units if "😀" in unit.surface)
    assert emoji.span[1] == len('label = '.encode("utf-8"))
    assert all(len(unit.recent_history) <= 8 for unit in extraction.units)
    assert extraction.coverage.scopes_balanced
    assert len(extraction.assessments) == len(extraction.units)
    assert {assessment.state for assessment in extraction.assessments} <= set(KnowledgeState)
    assert all(assessment.initial_score and assessment.final_score for assessment in extraction.assessments)


def test_scope_events_are_balanced_and_links_resolve_locally():
    engine = TreeSitterSemanticEngine()
    cartridge = _python_cartridge()
    source = "def outer():\n    value = 1\n    return value\n"

    document = engine.parse(source, cartridge=cartridge, path="scope.py", dialect=None)
    extraction = engine.extract(document, cartridge=cartridge)

    events = [unit.event for unit in extraction.units]
    assert events.count("SCOPE_ENTER") == events.count("SCOPE_EXIT") == 1
    assert extraction.coverage.links_valid
    assert all(
        unit.defines is None or unit.defines in document.index.definitions
        for unit in extraction.units
    )


def test_lexical_symbol_table_handles_shadowing_unresolved_and_stable_ids():
    first = LexicalSymbolTable(language="python", path="sample.py")
    outer = first.define("value", (1, 0, 1, 5))
    first.enter_scope("function", "run", 10)
    inner = first.define("value", (2, 4, 2, 9))

    assert first.resolve("value") == inner
    first.exit_scope()
    assert first.resolve("value") == outer
    assert first.resolve("missing") is None

    second = LexicalSymbolTable(language="python", path="sample.py")
    assert second.define("value", (1, 0, 1, 5)) == outer


@pytest.mark.parametrize(
    "source",
    [
        "def broken(:\n    pass\n",
        "if True\n    print('missing colon')\n",
    ],
)
def test_nested_error_or_missing_nodes_reject_whole_document(source: str):
    engine = TreeSitterSemanticEngine()
    cartridge = _python_cartridge()

    with pytest.raises(ValueError, match="ERROR|MISSING"):
        engine.parse(source, cartridge=cartridge, path="broken.py", dialect=None)


def test_coverage_admission_and_common_core_thresholds_are_explicit():
    passing = SemanticCoverage(
        total_units=10,
        common_core_units=8,
        unknown_units=1,
        total_named_nodes=12,
        excluded_named_nodes=2,
        mapped_named_nodes=9,
        unknown_named_nodes=1,
        scopes_balanced=True,
        links_valid=True,
    )
    low_mapping = replace(passing, mapped_named_nodes=8, unknown_named_nodes=2)
    high_unknown = replace(passing, unknown_units=2)
    low_core = replace(passing, common_core_units=7)

    assert passing.mapped_ratio == 0.9
    assert passing.unknown_ratio == 0.1
    assert passing.common_core_ratio == pytest.approx(8 / 9)
    assert passing.admitted
    assert not low_mapping.admitted
    assert not high_unknown.admitted
    assert not low_core.meets_common_core_gate


def test_spec_cannot_exclude_a_node_it_maps():
    with pytest.raises(ValueError, match="exclude.*map"):
        LanguageSemanticSpec(
            semantic_spec_version="test-v1",
            node_rules=(NodeRule("call", "CALL", "call"),),
            excluded_node_types=("call",),
        )


def test_required_shared_concepts_have_common_core_factories():
    core = CommonCodingCore()
    nodes = (
        core.comparison("eq"),
        core.data_structure("list"),
        core.io("write"),
        core.error("raise"),
        core.state("mutable"),
    )

    assert tuple(node.kind for node in nodes) == (
        IRKind.COMPARISON,
        IRKind.DATA_STRUCTURE,
        IRKind.IO,
        IRKind.ERROR,
        IRKind.STATE,
    )
