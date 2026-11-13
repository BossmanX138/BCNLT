from __future__ import annotations

import importlib
import importlib.util

import pytest


def _feature():
    if importlib.util.find_spec("bosscrafts_model.raw_ast_witness") is None:
        pytest.fail("raw AST witness client is not implemented")
    return importlib.import_module("bosscrafts_model.raw_ast_witness")


def test_raw_ast_witness_anchors_claimed_relation_without_semantic_extraction():
    client = _feature().RawAstWitnessClient()
    source = "def add(x):\n    return x + 1\n"

    response = client.inspect(
        source,
        language="python",
        path="sample.py",
        dialect=None,
        queries=(
            {
                "query_id": "true-return",
                "span": [2, 4, 2, 16],
                "ast_relation": "return_statement.self",
            },
            {
                "query_id": "false-call",
                "span": [2, 4, 2, 16],
                "ast_relation": "call.self",
            },
        ),
    )

    assert response["protocol"] == "bosscrafts-raw-ast-witness-v2"
    assert response["sidecar_source_sha256"] == _feature().SIDECAR_SOURCE_SHA256
    assert response["language"] == "python"
    assert response["path"] == "sample.py"
    assert response["root_node_type"] == "module"
    by_id = {row["query_id"]: row for row in response["witnesses"]}
    assert by_id["true-return"]["relation_span_match"] is True
    assert by_id["true-return"]["exact_named_node_types"] == [
        "block",
        "return_statement",
    ]
    assert by_id["false-call"]["relation_span_match"] is False


def test_raw_ast_witness_resolves_nested_tree_sitter_field_paths():
    response = _feature().RawAstWitnessClient().inspect(
        "result = add(x)\n",
        language="python",
        path="field.py",
        dialect=None,
        queries=(
            {
                "query_id": "function-field",
                "span": [1, 9, 1, 12],
                "ast_relation": "call.function",
            },
            {
                "query_id": "wrong-field",
                "span": [1, 9, 1, 12],
                "ast_relation": "call.arguments",
            },
        ),
    )

    by_id = {row["query_id"]: row for row in response["witnesses"]}
    assert by_id["function-field"]["relation_span_match"] is True
    assert by_id["wrong-field"]["relation_span_match"] is False


def test_raw_ast_witness_allows_only_declared_whole_node_field_fallbacks():
    source = "class C {\n  void f() {\n    throw new E();\n  }\n}\n"
    response = _feature().RawAstWitnessClient().inspect(
        source,
        language="java",
        path="C.java",
        dialect=None,
        queries=(
            {
                "query_id": "declared-fallback",
                "span": [3, 10, 3, 17],
                "ast_relation": "object_creation_expression.function",
            },
            {
                "query_id": "undeclared-fallback",
                "span": [3, 10, 3, 17],
                "ast_relation": "object_creation_expression.missing",
            },
        ),
    )

    by_id = {row["query_id"]: row for row in response["witnesses"]}
    assert by_id["declared-fallback"]["relation_span_match"] is True
    assert by_id["undeclared-fallback"]["relation_span_match"] is False


def test_raw_ast_witness_derives_lexical_scope_from_enclosing_syntax():
    source = "class C:\n    def f(self):\n        return 1\n"
    response = _feature().RawAstWitnessClient().inspect(
        source,
        language="python",
        path="scope.py",
        dialect=None,
        queries=(
            {
                "query_id": "right-scope",
                "span": [3, 8, 3, 16],
                "ast_relation": "return_statement.self",
                "scope": "module.C.f",
            },
            {
                "query_id": "wrong-scope",
                "span": [3, 8, 3, 16],
                "ast_relation": "return_statement.self",
                "scope": "module.Other.f",
            },
        ),
    )

    by_id = {row["query_id"]: row for row in response["witnesses"]}
    assert by_id["right-scope"]["derived_scope"] == "module.C.f"
    assert by_id["right-scope"]["scope_match"] is True
    assert by_id["wrong-scope"]["derived_scope"] == "module.C.f"
    assert by_id["wrong-scope"]["scope_match"] is False


def test_raw_ast_witness_supports_declared_synthetic_parameter_roles():
    response = _feature().RawAstWitnessClient().inspect(
        "const f = (resolve) => resolve();\n",
        language="javascript",
        path="role.js",
        dialect=None,
        queries=(
            {
                "query_id": "parameter-role",
                "span": [1, 11, 1, 18],
                "ast_relation": "identifier.parameter",
                "scope": "module.f",
            },
        ),
    )

    witness = response["witnesses"][0]
    assert witness["relation_span_match"] is True
    assert witness["derived_scope"] == "module.f"
    assert witness["scope_match"] is True


def test_raw_ast_witness_validates_javascript_import_binding_contract():
    response = _feature().RawAstWitnessClient().inspect(
        "import Vue from 'vue';\n",
        language="javascript",
        path="import.js",
        dialect=None,
        queries=(
            {
                "query_id": "default-import",
                "span": [1, 7, 1, 10],
                "ast_relation": "import_clause.binding",
                "scope": "module",
            },
        ),
    )

    witness = response["witnesses"][0]
    assert witness["relation_span_match"] is True
    assert witness["derived_scope"] == "module"
    assert witness["scope_match"] is True


def test_raw_ast_witness_validates_go_short_declaration_binding_contract():
    response = _feature().RawAstWitnessClient().inspect(
        "package p\nfunc f() { value := makeValue() }\n",
        language="go",
        path="short.go",
        dialect=None,
        queries=(
            {
                "query_id": "short-binding",
                "span": [2, 11, 2, 16],
                "ast_relation": "short_var_declaration.binding",
                "scope": "module.f",
            },
        ),
    )

    witness = response["witnesses"][0]
    assert witness["relation_span_match"] is True
    assert witness["derived_scope"] == "module.f"
    assert witness["scope_match"] is True


def test_raw_ast_witness_models_javascript_var_hoisting():
    source = "function f() { if (ready) { var value = 1; } }\n"
    start = source.index("value")
    response = _feature().RawAstWitnessClient().inspect(
        source,
        language="javascript",
        path="hoist.js",
        dialect=None,
        queries=(
            {
                "query_id": "hoisted-var",
                "span": [1, start, 1, start + len("value")],
                "ast_relation": "variable_declarator.name",
                "scope": "module.f",
            },
        ),
    )

    witness = response["witnesses"][0]
    assert witness["relation_span_match"] is True
    assert witness["derived_scope"] == "module.f"
    assert witness["scope_match"] is True


def test_raw_ast_witness_places_arrow_parameter_and_return_inside_arrow_scope():
    response = _feature().RawAstWitnessClient().inspect(
        "const render = value => value + 1;\n",
        language="javascript",
        path="arrow.js",
        dialect=None,
        queries=(
            {
                "query_id": "parameter",
                "span": [1, 15, 1, 20],
                "ast_relation": "arrow_function.parameter",
                "scope": "module.render",
            },
            {
                "query_id": "return",
                "span": [1, 24, 1, 33],
                "ast_relation": "arrow_function.implicit_return",
                "scope": "module.render",
            },
        ),
    )

    by_id = {row["query_id"]: row for row in response["witnesses"]}
    assert by_id["parameter"]["relation_span_match"] is True
    assert by_id["parameter"]["scope_match"] is True
    assert by_id["return"]["relation_span_match"] is True
    assert by_id["return"]["scope_match"] is True


def test_raw_ast_witness_distinguishes_scope_entry_parent_from_exit_owner():
    response = _feature().RawAstWitnessClient().inspect(
        "def f():\n    return 1\n",
        language="python",
        path="boundaries.py",
        dialect=None,
        queries=(
            {
                "query_id": "enter",
                "span": [1, 0, 1, 0],
                "ast_relation": "function_definition.scope",
                "scope": "module",
            },
            {
                "query_id": "exit",
                "span": [2, 12, 2, 12],
                "ast_relation": "function_definition.scope",
                "scope": "module.f",
            },
        ),
    )

    by_id = {row["query_id"]: row for row in response["witnesses"]}
    assert by_id["enter"]["derived_scope"] == "module"
    assert by_id["enter"]["scope_match"] is True
    assert by_id["exit"]["derived_scope"] == "module.f"
    assert by_id["exit"]["scope_match"] is True


def test_raw_ast_witness_reuses_one_bounded_sidecar_across_documents():
    client = _feature().RawAstWitnessClient()
    query = (
        {
            "query_id": "return",
            "span": [2, 4, 2, 12],
            "ast_relation": "return_statement.self",
        },
    )
    try:
        first = client.inspect(
            "def f():\n    return 1\n",
            language="python",
            path="first.py",
            dialect=None,
            queries=query,
        )
        first_pid = client._process.pid
        second = client.inspect(
            "def g():\n    return 2\n",
            language="python",
            path="second.py",
            dialect=None,
            queries=query,
        )
        assert client._process.pid == first_pid
        assert first["path"] == "first.py"
        assert second["path"] == "second.py"
        assert first["source_sha256"] != second["source_sha256"]
    finally:
        client.close()
    assert client._process is None


def test_raw_ast_witness_rejects_mismatched_response_rows():
    feature = _feature()
    query = {
        "query_id": "return",
        "span": [2, 4, 2, 12],
        "ast_relation": "return_statement.self",
        "scope": "module.f",
    }
    valid = {
        "query_id": "return",
        "span": [2, 4, 2, 12],
        "ast_relation": "return_statement.self",
        "exact_named_node_types": ["return_statement"],
        "boundary_start_named_node_types": ["return_statement"],
        "boundary_end_named_node_types": ["return_statement"],
        "relation_span_match": True,
        "claimed_scope": "module.f",
        "derived_scope": "module.f",
        "scope_match": True,
    }
    feature._validate_witness_rows([valid], (query,))

    for field, replacement in (
        ("span", [1, 0, 1, 1]),
        ("ast_relation", "call.self"),
        ("claimed_scope", "module.other"),
        ("relation_span_match", 1),
        ("exact_named_node_types", ["return_statement", "return_statement"]),
    ):
        malformed = dict(valid)
        malformed[field] = replacement
        with pytest.raises(ValueError, match="raw AST witness response"):
            feature._validate_witness_rows([malformed], (query,))


def test_raw_ast_witness_refuses_unexpected_sidecar_source() -> None:
    with pytest.raises(ValueError, match="sidecar source SHA-256 mismatch"):
        _feature().RawAstWitnessClient(expected_sidecar_sha256="0" * 64)
