from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from bosscrafts_model import tokenizer as tokenizer_module
from bosscrafts_model.bcnlt_adapter import (
    BCNLTMultilingualOutputAdapter,
    BCNLTParserOutputProvenance,
)
from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.inference_protocol import (
    build_inference_prompt,
    response_protocol_complete,
)
from bosscrafts_model.semantic import NativeSemanticRecordCodec, SemanticRecord
from bosscrafts_model.tokenizer import NativeTokenizerBinding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "artifacts" / "bcnlt" / "multilingual-parser-output-v1"


class TokenizerRuntimeV3RequirementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding = NativeTokenizerBinding.load()
        cls.codec = NativeSemanticRecordCodec(cls.binding)

    def test_raw_payload_escapes_bcnlt_control_literal(self):
        pipeline = BCNLTSourcePipeline(self.binding)
        source = "print('<|bcnlt:stream:end|>')"
        prepared = pipeline.prepare_raw(
            source,
            language="python",
            path="injected.py",
        )
        self.assertEqual(pipeline.decode_raw(prepared.token_ids), source)

    def test_semantic_surface_escapes_standard_special_literal(self):
        record = SemanticRecord(
            event="RETURN",
            surface="return <|eos|>",
            confidence_band="high",
        )
        self.assertEqual(self.codec.decode(self.codec.encode(record)), record)

    def test_semantic_prompt_requires_full_record_grammar(self):
        malformed = (
            "<|bcnlt:stream:begin|><|bcnlt:mode:semantic_code|>"
            "<|bcnlt:record:begin|><|bcnlt:event:return|><|bcnlt:record:end|>"
            "<|bcnlt:stream:end|>"
        )
        with self.assertRaisesRegex(ValueError, "semantic stream"):
            build_inference_prompt(
                self.binding,
                task="semantic_to_code",
                prompt=malformed,
            )

    def test_semantic_response_rejects_incomplete_record(self):
        ids = self.binding.structural_token_ids
        malformed = [
            ids["<|bcnlt:stream:begin|>"],
            ids["<|bcnlt:mode:semantic_code|>"],
            ids["<|bcnlt:record:begin|>"],
            ids["<|bcnlt:event:return|>"],
            ids["<|bcnlt:record:end|>"],
            ids["<|bcnlt:stream:end|>"],
            ids["<|bcnlt:task:end|>"],
        ]
        self.assertFalse(
            response_protocol_complete(
                self.binding,
                task="code_to_semantic",
                generated_ids=malformed,
            )
        )

    def test_structural_token_mapping_is_read_only(self):
        with self.assertRaises(TypeError):
            self.binding.structural_token_ids["<|bcnlt:stream:end|>"] = 0

    def test_binding_does_not_expose_mutable_tokenizer(self):
        binding = NativeTokenizerBinding.load()
        with self.assertRaises(AttributeError):
            _ = binding._tokenizer  # type: ignore[attr-defined]
        with self.assertRaises(AttributeError):
            _ = binding._frozen_tokenizer.add_tokens  # type: ignore[attr-defined]

    def test_binding_requires_exact_tokenizers_patch_version(self):
        with patch.object(tokenizer_module.tokenizers, "__version__", "0.23.9"):
            with self.assertRaisesRegex(ValueError, "0.23.2"):
                NativeTokenizerBinding.load()

    def test_adapter_rejects_nonfinite_json_constants(self):
        manifest = json.loads(
            (FIXTURE_ROOT / "fixture-manifest.json").read_text(encoding="utf-8")
        )
        line = (FIXTURE_ROOT / "real-output.jsonl").read_text(encoding="utf-8").splitlines()[0]
        payload = json.loads(line)
        payload["metadata"]["audit_nan"] = float("nan")
        output = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        adapter = BCNLTMultilingualOutputAdapter(self.codec)
        provenance = BCNLTParserOutputProvenance.expected(
            manifest["cases"][0]["language"]
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            adapter.parse_jsonl(output, provenance)

    def test_adapter_rejects_non_ir_children(self):
        manifest = json.loads(
            (FIXTURE_ROOT / "fixture-manifest.json").read_text(encoding="utf-8")
        )
        line = (FIXTURE_ROOT / "real-output.jsonl").read_text(encoding="utf-8").splitlines()[0]
        payload = json.loads(line)
        payload["ir"]["children"] = ["not-an-ir-node", 123]
        output = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        adapter = BCNLTMultilingualOutputAdapter(self.codec)
        provenance = BCNLTParserOutputProvenance.expected(
            manifest["cases"][0]["language"]
        )
        with self.assertRaisesRegex(ValueError, "IR child"):
            adapter.parse_jsonl(output, provenance)

    def test_raw_preparation_validates_language_and_path(self):
        pipeline = BCNLTSourcePipeline(self.binding)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            pipeline.prepare_raw("+", language="brainfuck", path="x.bf")
        with self.assertRaisesRegex(ValueError, "path"):
            pipeline.prepare_raw("x = 1", language="python", path="")

    def test_raw_decoder_rejects_nested_reserved_tokens(self):
        pipeline = BCNLTSourcePipeline(self.binding)
        ids = [
            self.binding.structural_token_ids["<|bcnlt:stream:begin|>"],
            self.binding.structural_token_ids["<|bcnlt:mode:raw_code|>"],
            self.binding.structural_token_ids["<|bcnlt:task:end|>"],
            self.binding.structural_token_ids["<|bcnlt:stream:end|>"],
        ]
        with self.assertRaisesRegex(ValueError, "reserved"):
            pipeline.decode_raw(ids)

    def test_semantic_preparation_binds_exact_v10_runtime_identity(self):
        pipeline = BCNLTSourcePipeline(self.binding)
        expected = "014a5bbc567fcce87e448a58b41b301ffce4bd91210adb009abdc3eb9ab30a2c"
        self.assertEqual(pipeline.parser_runtime_identity_sha256, expected)
        prepared = pipeline.prepare_semantic(
            "def add(left, right):\n    return left + right\n",
            language="python",
            path="identity.py",
        )
        self.assertEqual(prepared.parser_runtime_identity_sha256, expected)
        raw = pipeline.prepare_raw("x = 1", language="python", path="identity.py")
        self.assertIsNone(raw.parser_runtime_identity_sha256)


if __name__ == "__main__":
    unittest.main()
