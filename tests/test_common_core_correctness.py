from __future__ import annotations

import pytest

from bcnlt.adapters.carousel import BCNLTMultilingualAdapter
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine


def _extract(source: str, *, language: str, path: str):
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse(source, language=language, path=path)
    return document, adapter.extract(document)


def test_ordinary_identifier_read_links_to_its_definition():
    document, extraction = _extract(
        "const x = 1; x;",
        language="javascript",
        path="ordinary-reference.js",
    )

    definition = next(unit for unit in extraction.units if unit.event == "DEFINE" and unit.surface == "x")
    reference = next(unit for unit in extraction.units if unit.event == "REFERENCE" and unit.surface == "x")

    assert reference.defines == definition.symbol_id
    assert document.index.references[definition.symbol_id][0].span == reference.span
    assert extraction.coverage.links_valid


@pytest.mark.parametrize(
    ("operator", "expected_surface"),
    (("++", "x++"), ("--", "x--")),
)
def test_javascript_updates_preserve_the_mutation_operator(operator: str, expected_surface: str):
    _, extraction = _extract(
        f"let x = 0; x{operator};",
        language="javascript",
        path=f"update-{operator}.js",
    )

    assert any(
        unit.event == "OPERATOR" and unit.surface == expected_surface
        for unit in extraction.units
    )


def test_javascript_lexical_binding_does_not_escape_its_block():
    _, extraction = _extract(
        "let f = () => 1; { let f = () => 2; } f();",
        language="javascript",
        path="block-let.js",
    )

    definitions = [unit for unit in extraction.units if unit.event == "DEFINE" and unit.surface == "f"]
    call = next(unit for unit in extraction.units if unit.event == "CALL" and unit.surface == "f")

    assert len(definitions) == 2
    assert call.defines == definitions[0].symbol_id


def test_javascript_var_binding_targets_the_enclosing_function_scope():
    _, extraction = _extract(
        "function pick() { { var f = () => 2; } return f(); }",
        language="javascript",
        path="block-var.js",
    )

    definition = next(unit for unit in extraction.units if unit.event == "DEFINE" and unit.surface == "f")
    call = next(unit for unit in extraction.units if unit.event == "CALL" and unit.surface == "f")

    assert definition.scope.endswith("pick")
    assert call.defines == definition.symbol_id


def test_ruby_ordinary_call_is_not_mislabeled_as_an_import():
    _, extraction = _extract(
        'puts("hello")',
        language="ruby",
        path="ordinary-call.rb",
    )

    assert any(unit.event == "CALL" and unit.surface == "puts" for unit in extraction.units)
    assert not any(unit.event == "IMPORT" for unit in extraction.units)


@pytest.mark.parametrize("loader", ("require", "require_relative", "load"))
def test_ruby_loader_calls_remain_imports(loader: str):
    _, extraction = _extract(
        f'{loader} "library"',
        language="ruby",
        path="loader.rb",
    )

    assert any(unit.event == "IMPORT" and unit.surface == loader for unit in extraction.units)
