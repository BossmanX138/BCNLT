from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from bosscrafts_model.bcnlt_source_pipeline import (
    _DEFAULT_PYTHON_313,
    CURRENT_RUNTIME_SPEC_SHA256,
    BCNLTPreprocessingError,
    BCNLTSourcePipeline,
)
from bosscrafts_model.config import ModelConfig
from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.tokenizer import NativeTokenizerBinding

LANGUAGE_SAMPLES = {
    "python": "def f(x):\n    return x + 1\n",
    "javascript": "function f(x) { return x + 1; }\n",
    "typescript": "function f(x: number): number { return x + 1; }\n",
    "c": "int f(int x) { return x + 1; }\n",
    "cpp": "int f(int x) { return x + 1; }\n",
    "c_sharp": "class C { static int F(int x) { return x + 1; } }\n",
    "java": "class C { static int f(int x) { return x + 1; } }\n",
    "go": "package main\nfunc f(x int) int { return x + 1 }\n",
    "rust": "fn f(x: i32) -> i32 { x + 1 }\n",
    "php": "<?php function f($x) { return $x + 1; }\n",
    "ruby": "def f(x)\n  x + 1\nend\n",
    "kotlin": "fun f(x: Int): Int { return x + 1 }\n",
    "swift": "func f(_ x: Int) -> Int { return x + 1 }\n",
    "bash": "f() { local x=$1; return $((x + 1)); }\n",
}


class BCNLTSourcePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()
        cls.pipeline = BCNLTSourcePipeline(cls.binding)

    def test_default_sidecar_interpreter_is_active_verified_python(self) -> None:
        self.assertEqual(_DEFAULT_PYTHON_313, Path(sys.executable).resolve())

    def test_live_python_and_javascript_semantics_use_the_frozen_structural_ids(self):
        cases = (
            ("python", "def add(a, b):\n    return a + b\n", "sample.py"),
            ("javascript", "function add(a, b) { return a + b; }\n", "sample.js"),
        )
        for language, source, path in cases:
            with self.subTest(language=language):
                prepared = self.pipeline.prepare_semantic(
                    source,
                    language=language,
                    path=path,
                )
                self.assertEqual(prepared.mode, "semantic_code")
                self.assertEqual(prepared.language, language)
                self.assertFalse(prepared.used_raw_fallback)
                self.assertGreater(prepared.record_count, 0)
                self.assertEqual(
                    prepared.semantic_spec_sha256,
                    CURRENT_RUNTIME_SPEC_SHA256[language],
                )
                self.assertEqual(
                    prepared.source_sha256,
                    hashlib.sha256(source.encode("utf-8")).hexdigest(),
                )
                self.assertEqual(
                    prepared.canonical_sha256,
                    hashlib.sha256(
                        prepared.canonical_jsonl.encode("utf-8")
                    ).hexdigest(),
                )
                ids = prepared.token_ids
                structural = self.binding.structural_token_ids
                self.assertEqual(ids[0], structural["<|bcnlt:stream:begin|>"])
                self.assertEqual(ids[1], structural["<|bcnlt:mode:semantic_code|>"])
                self.assertEqual(ids[-1], structural["<|bcnlt:stream:end|>"])
                self.assertEqual(
                    ids.count(structural["<|bcnlt:record:begin|>"]),
                    prepared.record_count,
                )
                self.assertEqual(
                    ids.count(structural["<|bcnlt:record:end|>"]),
                    prepared.record_count,
                )
                self.assertEqual(
                    self.pipeline.decode_semantic(ids),
                    prepared.canonical_jsonl,
                )
                payloads = [
                    json.loads(line) for line in prepared.canonical_jsonl.splitlines()
                ]
                self.assertTrue(
                    all("knowledge_state" in record["metadata"] for record in payloads)
                )
                self.assertTrue(
                    all("uyr_rounds" in record["metadata"] for record in payloads)
                )

    def test_semantic_output_is_deterministic_and_does_not_mutate_the_vocab(self):
        source = "fn add(a: i32, b: i32) -> i32 { a + b }\n"
        before = dict(self.binding.structural_token_ids)
        first = self.pipeline.prepare_semantic(source, language="rust", path="lib.rs")
        second = self.pipeline.prepare_semantic(source, language="rust", path="lib.rs")
        self.assertEqual(first, second)
        self.assertEqual(self.binding.structural_token_ids, before)
        self.assertTrue(
            all(0 <= token_id < self.binding.vocab_size for token_id in first.token_ids)
        )

    def test_semantic_decoder_rejects_null_in_required_surface_field(self):
        prepared = self.pipeline.prepare_semantic(
            "x = 1\n",
            language="python",
            path="decode.py",
        )
        ids = list(prepared.token_ids)
        surface_begin = self.binding.structural_token_ids[
            "<|bcnlt:field:surface:begin|>"
        ]
        surface_end = self.binding.structural_token_ids["<|bcnlt:field:surface:end|>"]
        value_null = self.binding.structural_token_ids["<|bcnlt:value:null|>"]
        start = ids.index(surface_begin)
        end = ids.index(surface_end, start + 1)
        ids[start + 1 : end] = [value_null]
        with self.assertRaisesRegex(ValueError, "surface"):
            self.pipeline.decode_semantic(ids)

    def test_all_fourteen_registered_languages_execute_live(self):
        self.assertEqual(set(LANGUAGE_SAMPLES), set(self.pipeline.supported_languages))
        for language, source in LANGUAGE_SAMPLES.items():
            with self.subTest(language=language):
                prepared = self.pipeline.prepare_semantic(
                    source,
                    language=language,
                    path=f"sample.{language}",
                )
                self.assertGreater(prepared.record_count, 0)
                self.assertEqual(
                    prepared.semantic_spec_sha256, CURRENT_RUNTIME_SPEC_SHA256[language]
                )
                self.assertEqual(
                    self.pipeline.decode_semantic(prepared.token_ids),
                    prepared.canonical_jsonl,
                )

    def test_raw_mode_is_preserved_and_fallback_is_never_silent(self):
        source = "function {"
        with self.assertRaises(BCNLTPreprocessingError):
            self.pipeline.prepare_semantic(
                source,
                language="javascript",
                path="broken.js",
            )

        prepared = self.pipeline.prepare_semantic(
            source,
            language="javascript",
            path="broken.js",
            allow_raw_fallback=True,
        )
        self.assertEqual(prepared.mode, "raw_code")
        self.assertTrue(prepared.used_raw_fallback)
        self.assertIsNotNone(prepared.fallback_reason)
        self.assertEqual(self.pipeline.decode_raw(prepared.token_ids), source)
        self.assertEqual(
            prepared.token_ids[1],
            self.binding.structural_token_ids["<|bcnlt:mode:raw_code|>"],
        )

    def test_unsupported_language_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported BCNLT language"):
            self.pipeline.prepare_semantic("+", language="brainfuck", path="x.bf")

    def test_sidecar_transport_captures_binary_output_before_decoding(self):
        completed = SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"parser rejected input",
        )
        with patch(
            "bosscrafts_model.bcnlt_source_pipeline.subprocess.run",
            return_value=completed,
        ) as run:
            with self.assertRaisesRegex(
                BCNLTPreprocessingError, "parser rejected input"
            ):
                self.pipeline._invoke_sidecar(
                    "x = 1\n",
                    language="python",
                    path="sample.py",
                    dialect=None,
                )

        call = run.call_args
        self.assertIsInstance(call.kwargs["input"], bytes)
        self.assertNotIn("text", call.kwargs)
        self.assertNotIn("encoding", call.kwargs)

    def test_non_utf8_sidecar_stdout_fails_closed_without_reader_thread_decode(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout=b'{"protocol":"broken","payload":\xc2}',
            stderr=b"",
        )
        with patch(
            "bosscrafts_model.bcnlt_source_pipeline.subprocess.run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(BCNLTPreprocessingError, "non-UTF-8 output"):
                self.pipeline._invoke_sidecar(
                    "x = 1\n",
                    language="python",
                    path="sample.py",
                    dialect=None,
                )

    def test_prepared_semantic_ids_feed_the_model_without_retokenization(self):
        prepared = self.pipeline.prepare_semantic(
            "x = 1\n",
            language="python",
            path="tiny.py",
        )
        config = ModelConfig.tiny(
            vocab_size=self.binding.vocab_size,
            max_sequence=512,
            tokenizer_manifest=self.binding.binding_sha256,
        )
        torch.manual_seed(3)
        model = EngineeringModel(config)
        logits, routes = model.forward_prepared(prepared)
        self.assertEqual(
            tuple(logits.shape),
            (1, len(prepared.token_ids), self.binding.vocab_size),
        )
        self.assertEqual(len(routes), config.module_rounds)
        mismatched = EngineeringModel(
            ModelConfig.tiny(vocab_size=self.binding.vocab_size, max_sequence=512)
        )
        with self.assertRaisesRegex(ValueError, "tokenizer binding"):
            mismatched.forward_prepared(prepared)

    def test_pointer_declarator_emits_clean_variable_definition_name(self):
        prepared = self.pipeline.prepare_semantic(
            "void f(void) { char* newBuf = (char*)malloc(2); }\n",
            language="c",
            path="pointer.c",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        definitions = [
            record
            for record in records
            if record["event"] == "DEFINE" and record["surface"] == "newBuf"
        ]
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["ir"]["kind"], "variable")
        self.assertEqual(definitions[0]["ir"]["name"], "newBuf")
        self.assertFalse(
            any(
                record["event"] == "UNKNOWN_SEMANTIC"
                and record["ast_relation"] == "pointer_declarator.self"
                for record in records
            )
        )

    def test_kotlin_function_body_uses_function_name_as_scope(self):
        prepared = self.pipeline.prepare_semantic(
            "class C {\n    @Provides\n    internal fun provideGson(): Gson {\n        return GsonBuilder().create()\n    }\n}\n",
            language="kotlin",
            path="Provider.kt",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        returns = [record for record in records if record["event"] == "RETURN"]
        self.assertEqual(len(returns), 1)
        self.assertEqual(returns[0]["scope"], "module.C.provideGson")

    def test_plain_python_parameter_is_a_definition_not_a_reference(self):
        prepared = self.pipeline.prepare_semantic(
            "def format_person_info(info):\n    return info\n",
            language="python",
            path="person.py",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        parameter_span = [1, 23, 1, 27]
        parameter_records = [
            record
            for record in records
            if record["surface"] == "info" and record["span"] == parameter_span
        ]
        self.assertEqual(len(parameter_records), 1)
        self.assertEqual(parameter_records[0]["event"], "DEFINE")
        self.assertEqual(parameter_records[0]["ir"]["kind"], "variable")
        self.assertEqual(parameter_records[0]["ast_relation"], "identifier.parameter")

    def test_unresolved_reference_is_not_mislabeled_as_control_flow(self):
        prepared = self.pipeline.prepare_semantic(
            "int f() { return JNI_ERR; }\n",
            language="cpp",
            path="jni.cpp",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        reference = next(
            record
            for record in records
            if record["event"] == "REFERENCE" and record["surface"] == "JNI_ERR"
        )
        self.assertEqual(reference["metadata"]["wildcard_category"], "symbol_reference")

    def test_javascript_destructuring_defines_each_local_binding(self):
        prepared = self.pipeline.prepare_semantic(
            "const { typeDefs, resolvers: renamed, nested: { inner }, ...rest } = schemas;\n"
            "use(typeDefs, renamed, inner, rest);\n",
            language="javascript",
            path="destructure.js",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        expected = {"typeDefs", "renamed", "inner", "rest"}
        definitions = {
            record["surface"]: record
            for record in records
            if record["event"] == "DEFINE" and record["surface"] in expected
        }
        self.assertEqual(set(definitions), expected)
        self.assertFalse(
            any(
                record["event"] == "DEFINE" and record["surface"].startswith("{")
                for record in records
            )
        )
        for name, definition in definitions.items():
            references = [
                record
                for record in records
                if record["event"] == "REFERENCE"
                and record["surface"] == name
                and record["span"][0] == 2
            ]
            self.assertEqual(len(references), 1)
            self.assertEqual(references[0]["defines"], definition["symbol_id"])

    def test_literal_content_is_not_duplicated_as_unknown_semantics(self):
        prepared = self.pipeline.prepare_semantic(
            'const moduleName = "apollo-server-express";\n',
            language="javascript",
            path="literal.js",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        self.assertTrue(
            any(
                record["event"] == "LITERAL"
                and record["surface"] == '"apollo-server-express"'
                for record in records
            )
        )
        self.assertFalse(
            any(
                record["event"] == "UNKNOWN_SEMANTIC"
                and record["metadata"].get("unknown_node") == "string_fragment"
                for record in records
            )
        )

    def test_javascript_imports_define_local_names_and_aliases(self):
        prepared = self.pipeline.prepare_semantic(
            'import defaultName, { readFileSync, writeFile as write } from "fs";\n'
            'import * as path from "path";\n'
            "use(defaultName, readFileSync, write, path);\n",
            language="javascript",
            path="imports.js",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        expected = {"defaultName", "readFileSync", "write", "path"}
        definitions = {
            record["surface"]: record
            for record in records
            if record["event"] == "DEFINE" and record["surface"] in expected
        }
        self.assertEqual(set(definitions), expected)
        self.assertFalse(
            any(
                record["event"] == "DEFINE" and record["surface"] == "writeFile"
                for record in records
            )
        )
        for name, definition in definitions.items():
            reference = next(
                record
                for record in records
                if record["event"] == "REFERENCE"
                and record["surface"] == name
                and record["span"][0] == 3
            )
            self.assertEqual(reference["defines"], definition["symbol_id"])

    def test_member_property_is_owned_by_member_access_record(self):
        prepared = self.pipeline.prepare_semantic(
            "const size = object.length;\n",
            language="javascript",
            path="member.js",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        member = [
            record
            for record in records
            if record["event"] == "MEMBER_ACCESS" and record["surface"] == "length"
        ]
        self.assertEqual(len(member), 1)
        self.assertFalse(
            any(
                record["surface"] == "length"
                and record["event"] in {"REFERENCE", "UNKNOWN_SEMANTIC"}
                and record["span"] == member[0]["span"]
                for record in records
            )
        )

    def test_member_property_aliases_are_clean_across_language_grammars(self):
        cases = (
            ("python", "x = obj.field\n", "member.py", "field"),
            ("c", "int f(void) { return obj.field; }\n", "member.c", "field"),
            ("cpp", "int f() { return obj.field; }\n", "member.cpp", "field"),
            (
                "c_sharp",
                "class C { int F() { return obj.field; } }\n",
                "member.cs",
                "field",
            ),
            (
                "java",
                "class C { int f() { return obj.field; } }\n",
                "Member.java",
                "field",
            ),
            ("go", "package p\nfunc f() { x := obj.field }\n", "member.go", "field"),
            ("rust", "fn f() { let x = obj.field; }\n", "member.rs", "field"),
            ("php", "<?php $x = $obj->field;\n", "member.php", "field"),
            ("ruby", "x = Obj::Field\n", "member.rb", "Field"),
            ("kotlin", "fun f() { val x = obj.field }\n", "member.kt", "field"),
            ("swift", "func f() { let x = obj.field }\n", "member.swift", "field"),
        )
        for language, source, path, property_name in cases:
            with self.subTest(language=language):
                prepared = self.pipeline.prepare_semantic(
                    source,
                    language=language,
                    path=path,
                )
                records = [
                    json.loads(line) for line in prepared.canonical_jsonl.splitlines()
                ]
                member = [
                    record
                    for record in records
                    if record["event"] == "MEMBER_ACCESS"
                    and record["surface"] == property_name
                ]
                self.assertEqual(len(member), 1)
                self.assertFalse(
                    any(
                        record["surface"].lstrip(".") == property_name
                        and record["event"] == "UNKNOWN_SEMANTIC"
                        and record["span"] == member[0]["span"]
                        for record in records
                    )
                )
                if language in {"kotlin", "swift"}:
                    self.assertFalse(
                        any(
                            record["event"] == "UNKNOWN_SEMANTIC"
                            and record["metadata"].get("unknown_node")
                            == "navigation_suffix"
                            and property_name in record["surface"]
                            for record in records
                        )
                    )

    def test_typescript_array_destructuring_and_defaults_preserve_binding_order(self):
        prepared = self.pipeline.prepare_semantic(
            "const [first, second = first, ...rest]: [number, number, ...number[]] = values;\n"
            "use(first, second, rest);\n",
            language="typescript",
            path="destructure.ts",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        definitions = {
            record["surface"]: record
            for record in records
            if record["event"] == "DEFINE"
            and record["surface"] in {"first", "second", "rest"}
        }
        self.assertEqual(set(definitions), {"first", "second", "rest"})
        default_reference = next(
            record
            for record in records
            if record["event"] == "REFERENCE"
            and record["surface"] == "first"
            and record["span"][0] == 1
            and record["span"][1] > definitions["second"]["span"][1]
        )
        self.assertEqual(
            default_reference["defines"], definitions["first"]["symbol_id"]
        )

    def test_javascript_destructured_parameters_are_function_scoped_bindings(self):
        prepared = self.pipeline.prepare_semantic(
            "function f({a, b: renamed = a}, [first, ...rest]) {\n"
            "  return use(a, renamed, first, rest);\n"
            "}\n",
            language="javascript",
            path="parameters.js",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        expected = {"a", "renamed", "first", "rest"}
        definitions = {
            record["surface"]: record
            for record in records
            if record["event"] == "DEFINE" and record["surface"] in expected
        }
        self.assertEqual(set(definitions), expected)
        self.assertTrue(
            all(record["scope"] == "module.f" for record in definitions.values())
        )
        for name, definition in definitions.items():
            body_reference = next(
                record
                for record in records
                if record["event"] == "REFERENCE"
                and record["surface"] == name
                and record["span"][0] == 2
            )
            self.assertEqual(body_reference["defines"], definition["symbol_id"])

    def test_isolated_sidecar_roundtrips_non_ascii_source_over_utf8_transport(self):
        source = "// 背景色\nconst value = 'Grüße';\n"
        prepared = self.pipeline.prepare_semantic(
            source,
            language="javascript",
            path="unicode.js",
        )
        self.assertEqual(
            prepared.source_sha256, hashlib.sha256(source.encode()).hexdigest()
        )
        self.assertNotIn("\\udc", prepared.canonical_jsonl)

    def test_semantic_surfaces_and_history_are_bounded_before_handoff(self):
        source = "const value = '" + ("x" * 6000) + "';\n"
        prepared = self.pipeline.prepare_semantic(
            source,
            language="javascript",
            path="large-surface.js",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        self.assertTrue(
            any(record["metadata"].get("surface_truncated") for record in records)
        )
        self.assertLessEqual(max(len(record["surface"]) for record in records), 4096)
        self.assertLessEqual(
            max(
                (len(item) for record in records for item in record["recent_history"]),
                default=0,
            ),
            512,
        )


if __name__ == "__main__":
    unittest.main()
