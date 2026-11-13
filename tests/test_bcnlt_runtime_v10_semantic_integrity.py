from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate"
SIDECAR = RUNTIME / "bcnlt_source_sidecar.py"
PYTHON_313 = Path(sys.executable).resolve()
PROTOCOL = "bosscrafts-bcnlt-source-sidecar-v4"


def _records(language: str, source: str) -> list[dict[str, object]]:
    request = {
        "protocol": PROTOCOL,
        "source": source,
        "language": language,
        "path": f"semantic-integrity.{language}",
        "dialect": None,
        "learning_mode": "ephemeral",
        "learned_memory_root": None,
    }
    completed = subprocess.run(
        [str(PYTHON_313), "-I", "-B", "-X", "utf8", str(SIDECAR)],
        input=json.dumps(request),
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
        check=False,
        cwd=RUNTIME,
    )
    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    return [
        json.loads(line)
        for line in str(response["canonical_jsonl"]).split("\n")
        if line
    ]


def test_python_lambda_keyword_does_not_create_a_second_function_scope():
    records = _records(
        "python",
        "def f(values):\n    return list(filter(lambda value: value, values))\n",
    )

    boundaries = [
        record
        for record in records
        if record.get("ast_relation") == "lambda.scope"
    ]

    assert [record["event"] for record in boundaries] == [
        "SCOPE_ENTER",
        "SCOPE_EXIT",
    ]
    assert [record["scope"] for record in boundaries] == [
        "module.f",
        "module.f.lambda",
    ]


def test_typescript_type_keyword_is_not_misclassified_as_a_string_literal():
    records = _records("typescript", "let appId: string;\n")

    assert not any(
        record.get("event") == "LITERAL"
        and record.get("surface") == "string"
        for record in records
    )


def test_javascript_function_expression_owns_its_parameter_and_body_scope():
    records = _records(
        "javascript",
        "const convert = function(value) { const result = value + 1; return result; };\n",
    )

    assert not any(
        record.get("event") == "UNKNOWN_SEMANTIC"
        and record.get("ast_relation") == "function_expression.self"
        for record in records
    )
    assert any(
        record.get("event") == "SCOPE_ENTER"
        and record.get("ast_relation") == "function_expression.scope"
        and record.get("surface") == "convert"
        and record.get("scope") == "module"
        for record in records
    )
    parameter = next(
        record
        for record in records
        if record.get("event") == "DEFINE"
        and record.get("surface") == "value"
    )
    assert parameter["scope"] == "module.convert"
    assert any(
        record.get("event") == "RETURN"
        and str(record.get("scope", "")).startswith("module.convert")
        for record in records
    )


def test_javascript_expression_arrow_uses_common_core_return_ir():
    records = _records("javascript", "const double = x => x * 2;\n")

    implicit_return = next(
        record
        for record in records
        if record.get("ast_relation") == "arrow_function.implicit_return"
    )

    assert implicit_return["event"] == "RETURN"
    assert implicit_return["ir"]["kind"] == "return"


def test_rust_tail_expression_uses_common_core_return_ir():
    records = _records("rust", "fn double(x: i32) -> i32 { x * 2 }\n")

    implicit_return = next(
        record
        for record in records
        if record.get("ast_relation") == "function_item.implicit_tail"
    )

    assert implicit_return["event"] == "RETURN"
    assert implicit_return["ir"]["kind"] == "return"


def test_unknown_syntax_cannot_become_confident_without_validated_extended_core():
    source = (
        "def update(items):\n"
        "    first = items[0]\n"
        "    second = items[0]\n"
        "    third = items[0]\n"
        "    fourth = items[0]\n"
        "    items[0] = 1\n"
    )
    records = _records("python", source)
    unknowns = [
        record
        for record in records
        if record.get("event") == "UNKNOWN_SEMANTIC"
        and record.get("ast_relation") == "subscript.self"
    ]

    assert unknowns
    assert all(
        not (
            record.get("confidence_band") == "high"
            or record.get("uncertainty_band") == "low"
            or record["metadata"].get("knowledge_state") == "retrieved"
        )
        for record in unknowns
        if record["metadata"].get("extended_core_state") != "validated"
    )
