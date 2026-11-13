from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate"
SIDECAR = RUNTIME / "bcnlt_source_sidecar.py"
PYTHON_313 = Path(sys.executable).resolve()
PROTOCOL = "bosscrafts-bcnlt-source-sidecar-v4"


def _run_raw_sidecar(request: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON_313), "-I", "-S", "-B", "-X", "utf8", str(SIDECAR)],
        input=request,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
        check=False,
        cwd=RUNTIME,
    )


def _run_sidecar(language: str, source: str) -> list[dict[str, object]]:
    request = {
        "protocol": PROTOCOL,
        "source": source,
        "language": language,
        "path": f"v3-audit.{language}",
        "dialect": None,
        "learning_mode": "ephemeral",
        "learned_memory_root": None,
    }
    completed = _run_raw_sidecar(json.dumps(request))
    if completed.returncode:
        raise AssertionError(completed.stderr)
    response = json.loads(completed.stdout)
    return [json.loads(line) for line in response["canonical_jsonl"].splitlines()]


def _definition(records: list[dict[str, object]], surface: str) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if record.get("event") == "DEFINE" and record.get("surface") == surface
    ]


def _references(records: list[dict[str, object]], surface: str) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if record.get("event") == "REFERENCE" and record.get("surface") == surface
    ]


def _plain_parameters(records: list[dict[str, object]]) -> bool:
    for name in ("left", "right"):
        definitions = _definition(records, name)
        references = _references(records, name)
        if len(definitions) != 1 or not references:
            return False
        if any(record.get("defines") != definitions[0].get("symbol_id") for record in references):
            return False
    return True


def _arrow_body(records: list[dict[str, object]]) -> bool:
    return bool(
        _definition(records, "double")
        and any(
            record.get("event") == "RETURN"
            and "return result" in str(record.get("surface"))
            and "double" in str(record.get("scope"))
            for record in records
        )
    )


def _rust_tail(records: list[dict[str, object]]) -> bool:
    return any(
        record.get("event") == "RETURN"
        and "x*2" in str(record.get("surface")).replace(" ", "")
        for record in records
    )


def _python_class_ownership(records: list[dict[str, object]]) -> bool:
    return [record.get("scope") for record in _definition(records, "value")] == [
        "module.A",
        "module.B",
    ]


def _c_shadowing(records: list[dict[str, object]]) -> bool:
    definitions = _definition(records, "y")
    references = _references(records, "y")
    return (
        len(definitions) == 2
        and definitions[0].get("scope") != definitions[1].get("scope")
        and [record.get("defines") for record in references]
        == [definitions[1].get("symbol_id"), definitions[0].get("symbol_id")]
    )


def _go_multi_binding(records: list[dict[str, object]]) -> bool:
    definitions = {
        record.get("surface"): record
        for record in records
        if record.get("event") == "DEFINE"
    }
    if not {"a", "b"}.issubset(definitions) or "a,b" in definitions:
        return False
    return all(
        _references(records, name)
        and all(
            record.get("defines") == definitions[name].get("symbol_id")
            for record in _references(records, name)
        )
        for name in ("a", "b")
    )


def _go_control_short_definition(records: list[dict[str, object]]) -> bool:
    definitions = _definition(records, "err")
    return (
        len(definitions) == 2
        and definitions[0].get("scope") != definitions[1].get("scope")
        and "if_statement" in str(definitions[1].get("scope"))
    )


def _go_variadic_parameter(records: list[dict[str, object]]) -> bool:
    return len(_definition(records, "handlers")) == 1


def _swift_parameter_name(records: list[dict[str, object]]) -> bool:
    return len(_definition(records, "json")) == 1 and not _definition(records, "JSON")


def _c_unnamed_void_parameter(records: list[dict[str, object]]) -> bool:
    return not _definition(records, "void")


def _type_method_ownership(records: list[dict[str, object]]) -> bool:
    scopes = [record.get("scope") for record in _definition(records, "run")]
    return len(scopes) == 2 and scopes[0] != scopes[1] and all(
        marker in str(scope)
        for marker, scope in zip(("A", "B"), scopes, strict=True)
    )


CASES = (
    (
        "javascript_plain_parameters",
        "javascript",
        "function combine(left, right) { return left + right; }\n",
        _plain_parameters,
    ),
    (
        "ruby_plain_parameters",
        "ruby",
        "def combine(left, right)\n return left + right\nend\n",
        _plain_parameters,
    ),
    (
        "javascript_arrow_body_ownership",
        "javascript",
        "const double = x => {\n const result=x*2;\n return result;\n};\n",
        _arrow_body,
    ),
    ("rust_implicit_tail_return", "rust", "fn double(x:i32)->i32 {\n x*2\n}\n", _rust_tail),
    (
        "python_class_method_ownership",
        "python",
        "class A:\n    def value(self):\n        return 1\nclass B:\n    def value(self):\n        return 2\n",
        _python_class_ownership,
    ),
    (
        "c_block_shadowing",
        "c",
        "int f(int x) {\n int y=x;\n { int y=9; use(y); }\n return y;\n}\n",
        _c_shadowing,
    ),
    (
        "cpp_block_shadowing",
        "cpp",
        "int f(int x) {\n int y=x;\n { int y=9; use(y); }\n return y;\n}\n",
        _c_shadowing,
    ),
    (
        "go_multi_binding",
        "go",
        "package test\nfunc pair() int { a,b := 1,2; return a+b }\n",
        _go_multi_binding,
    ),
    (
        "go_control_short_definition",
        "go",
        "package test\nfunc f() { err := open(); if err := close(); err == nil {} }\n",
        _go_control_short_definition,
    ),
    (
        "go_variadic_parameter",
        "go",
        "package test\nfunc f(handlers ...Handler) {}\n",
        _go_variadic_parameter,
    ),
    (
        "swift_parameter_name",
        "swift",
        "func deserialize(json: JSON) -> JSON { return json }\n",
        _swift_parameter_name,
    ),
    (
        "c_unnamed_void_parameter",
        "c",
        "static int returns_int(void) { return 1; }\n",
        _c_unnamed_void_parameter,
    ),
    (
        "c_sharp_type_method_ownership",
        "c_sharp",
        "class A { void run() {} } class B { void run() {} }",
        _type_method_ownership,
    ),
    (
        "java_type_method_ownership",
        "java",
        "class A { void run() {} } class B { void run() {} }",
        _type_method_ownership,
    ),
    (
        "javascript_type_method_ownership",
        "javascript",
        "class A { run() {} } class B { run() {} }",
        _type_method_ownership,
    ),
    (
        "typescript_type_method_ownership",
        "typescript",
        "class A { run() {} } class B { run() {} }",
        _type_method_ownership,
    ),
    (
        "kotlin_type_method_ownership",
        "kotlin",
        "class A { fun run() {} }\nclass B { fun run() {} }\n",
        _type_method_ownership,
    ),
    (
        "php_type_method_ownership",
        "php",
        "<?php class A { function run() {} } class B { function run() {} }",
        _type_method_ownership,
    ),
    (
        "ruby_type_method_ownership",
        "ruby",
        "class A\n def run; end\nend\nclass B\n def run; end\nend\n",
        _type_method_ownership,
    ),
    (
        "swift_type_method_ownership",
        "swift",
        "class A { func run() {} }\nclass B { func run() {} }\n",
        _type_method_ownership,
    ),
    (
        "cpp_type_method_ownership",
        "cpp",
        "class A { void run() {} }; class B { void run() {} };",
        _type_method_ownership,
    ),
    (
        "rust_type_method_ownership",
        "rust",
        "struct A; impl A { fn run(&self) {} } struct B; impl B { fn run(&self) {} }",
        _type_method_ownership,
    ),
)


class BCNltCurrentRuntimeSemanticTests(unittest.TestCase):
    def test_all_reviewed_semantic_repairs_are_present(self):
        for name, language, source, evaluator in CASES:
            with self.subTest(case=name):
                self.assertTrue(evaluator(_run_sidecar(language, source)))

    def test_missing_owned_grammar_fails_without_pack_fallback(self):
        with tempfile.TemporaryDirectory(prefix="bcnlt-v10-offline-test-") as temporary:
            code = f"""
import os
from pathlib import Path
import sys
from unittest.mock import patch
runtime = Path({str(RUNTIME)!r})
sys.path.insert(0, str(runtime / 'site-packages'))
sys.path.insert(0, str(runtime / 'source'))
os.environ['BCNLT_PARSER_CACHE'] = {temporary!r}
from bcnlt.adapters.tree_sitter import _default_parser_provider
with patch('tree_sitter_language_pack.get_parser', side_effect=AssertionError('pack fallback reached')):
    try:
        _default_parser_provider('python')
    except ValueError as error:
        if 'owned grammar' not in str(error):
            raise
    else:
        raise AssertionError('missing owned grammar was accepted')
"""
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [str(PYTHON_313), "-I", "-S", "-B", "-X", "utf8", "-c", code],
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=30,
                check=False,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sidecar_rejects_duplicate_request_keys(self):
        request = (
            '{"protocol":"' + PROTOCOL + '","source":"x = 1\\n",'
            '"source":"x = 2\\n","language":"python",'
            '"path":"duplicate.py","dialect":null,'
            '"learning_mode":"ephemeral","learned_memory_root":null}'
        )
        completed = _run_raw_sidecar(request)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("ValueError", completed.stderr)
        self.assertNotIn("x = 2", completed.stderr)

    def test_sidecar_rejects_unbounded_dialect_metadata(self):
        request = json.dumps(
            {
                "protocol": PROTOCOL,
                "source": "x = 1\n",
                "language": "python",
                "path": "dialect.py",
                "dialect": "x" * 257,
                "learning_mode": "ephemeral",
                "learned_memory_root": None,
            }
        )
        completed = _run_raw_sidecar(request)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("ValueError", completed.stderr)
        self.assertNotIn("x" * 257, completed.stderr)


if __name__ == "__main__":
    unittest.main()
