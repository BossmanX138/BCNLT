import io
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from bosscrafts_model.config import ModelConfig
from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.tokenizer import (
    EXPECTED_TOKENIZER_SHA256,
    NativeTokenizerBinding,
    bound_native_65k_config,
    validate_contiguous_vocab,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = PROJECT_ROOT / "artifacts" / "tokenizers" / "bcnlt-native-foundation-65536-v1"


class NativeTokenizerBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding = NativeTokenizerBinding.load(BUNDLE)

    def test_verified_bundle_contract(self):
        binding = self.binding
        self.assertEqual(binding.tokenizer_sha256, EXPECTED_TOKENIZER_SHA256)
        self.assertEqual(binding.vocab_size, 65_536)
        self.assertEqual(binding.structural_token_count, 73)
        self.assertEqual(sorted(binding.structural_token_ids.values()), list(range(4, 77)))
        self.assertEqual(binding.runtime_package, "tokenizers")
        self.assertEqual(binding.runtime_major_minor, "0.23")

    def test_real_multilingual_and_serialized_semantic_roundtrip(self):
        samples = (
            "English: repair the parser without changing public behavior.",
            "def café(value):\n    return value + 1\n",
            "const 東京 = (x) => x?.value ?? 0;",
            "std::vector<int> values{1, 2, 3};",
            "fn main() { println!(\"hello 🚀\"); }",
            (
                "<|bcnlt:record:begin|>\n"
                "<|bcnlt:mode:semantic_code|>\n"
                '{"event":"CALL","surface":"build","confidence_band":"high"}\n'
                "<|bcnlt:record:end|>"
            ),
        )
        for sample in samples:
            ids = self.binding.encode(sample)
            self.assertTrue(ids)
            self.assertTrue(all(type(token_id) is int and 0 <= token_id < 65_536 for token_id in ids))
            self.assertEqual(self.binding.decode(ids), sample)

    def test_public_batch_encoding_matches_scalar_and_guards_payloads(self):
        samples = ["alpha = 1", "function beta() { return 2; }"]
        self.assertEqual(
            self.binding.encode_batch(samples),
            [self.binding.encode(sample) for sample in samples],
        )
        self.assertEqual(
            self.binding.encode_payload_batch(samples),
            [self.binding.encode_payload(sample) for sample in samples],
        )
        escaped = self.binding.encode_payload_batch(["safe", "<|eos|>"])
        self.assertEqual(
            [self.binding.decode(ids) for ids in escaped],
            ["safe", "<|eos|>"],
        )
        self.assertTrue(all(token_id > 76 for ids in escaped for token_id in ids))

    def test_bounds_fail_closed(self):
        for invalid in ([-1], [65_536], [True], ["1"], None):
            with self.assertRaises((TypeError, ValueError)):
                self.binding.decode(invalid)

    def test_runtime_version_mismatch_fails_closed(self):
        with patch("bosscrafts_model.tokenizer.tokenizers.__version__", "0.24.0"):
            with self.assertRaisesRegex(ValueError, "tokenizers runtime mismatch"):
                NativeTokenizerBinding.load(BUNDLE)

    def test_corruption_and_manifest_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            copied = Path(directory) / "bundle"
            shutil.copytree(BUNDLE, copied)
            tokenizer_path = copied / "tokenizer.json"
            tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                NativeTokenizerBinding.load(copied)

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            copied = Path(directory) / "bundle"
            shutil.copytree(BUNDLE, copied)
            binding_path = copied / "binding.json"
            payload = json.loads(binding_path.read_text(encoding="utf-8"))
            payload["vocabulary_size"] = 32_768
            binding_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding manifest SHA-256 mismatch"):
                NativeTokenizerBinding.load(copied)

    def test_contiguity_validator_rejects_gaps_and_duplicates(self):
        validate_contiguous_vocab({"a": 0, "b": 1}, expected_size=2)
        for invalid in ({"a": 0, "b": 2}, {"a": 0, "b": 0}, {"a": True, "b": 1}):
            with self.assertRaises(ValueError):
                validate_contiguous_vocab(invalid, expected_size=2)

    def test_bound_config_count_tied_output_and_reload(self):
        config = bound_native_65k_config(self.binding)
        self.assertEqual(config.vocab_size, 65_536)
        self.assertEqual(config.width, 1280)
        self.assertEqual(config.parameter_count(), 540_094_726)
        config.check_manifest(self.binding.binding_sha256)
        with torch.device("meta"):
            full = EngineeringModel(config)
        self.assertEqual(sum(parameter.numel() for parameter in full.parameters()), 540_094_726)
        with self.assertRaisesRegex(ValueError, "verified native 65K tokenizer binding required"):
            bound_native_65k_config(replace(self.binding, vocab_size=32_768))

        tiny = ModelConfig.tiny(
            vocab_size=65_536,
            tokenizer_manifest=self.binding.binding_sha256,
        )
        model = EngineeringModel(tiny).eval()
        self.assertFalse(hasattr(model, "lm_head"))
        ids = torch.tensor([self.binding.encode("int value = 7;")[:8]], dtype=torch.long)
        with torch.inference_mode():
            logits, _ = model(ids)
        self.assertEqual(tuple(logits.shape), (1, ids.shape[1], 65_536))
        self.assertTrue(torch.isfinite(logits).all())

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = EngineeringModel(tiny).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.inference_mode():
            restored_logits, _ = restored(ids)
        torch.testing.assert_close(restored_logits, logits)

    def test_model_rejects_out_of_range_ids_before_embedding(self):
        tiny = ModelConfig.tiny(vocab_size=65_536, tokenizer_manifest=self.binding.binding_sha256)
        model = EngineeringModel(tiny)
        for invalid in (
            torch.tensor([[-1]], dtype=torch.long),
            torch.tensor([[65_536]], dtype=torch.long),
        ):
            with self.assertRaisesRegex(ValueError, "token ID out of configured vocabulary bounds"):
                model(invalid)


if __name__ == "__main__":
    unittest.main()
