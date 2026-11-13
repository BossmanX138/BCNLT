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

    assert response["protocol"] == "bosscrafts-raw-ast-witness-v1"
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
