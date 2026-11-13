import hashlib
from dataclasses import replace
from pathlib import Path
import unittest

from bosscrafts_model.semantic import (
    NativeSemanticRecordCodec,
    SEMANTIC_RECORD_SCHEMA,
    SemanticRecord,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizers"
    / "bcnlt-native-foundation-65536-v1"
    / "tokenizer.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeSemanticRecordCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding = NativeTokenizerBinding.load()
        cls.codec = NativeSemanticRecordCodec(cls.binding)

    def test_multilingual_canonical_token_roundtrip_preserves_tokenizer(self):
        before = sha256(TOKENIZER_PATH)
        record = SemanticRecord(
            event="CALL",
            surface="build café / 修复 / исправить",
            confidence_band="high",
        )
        serialized = self.codec.serialize(record)
        ids = self.codec.encode(record)
        self.assertEqual(self.codec.parse(serialized), record)
        self.assertEqual(self.codec.decode(ids), record)
        self.assertEqual(ids[0], self.binding.structural_token_ids["<|bcnlt:record:begin|>"])
        self.assertEqual(ids[-1], self.binding.structural_token_ids["<|bcnlt:record:end|>"])
        self.assertTrue(all(0 <= token_id < 65_536 for token_id in ids))
        self.assertEqual(sha256(TOKENIZER_PATH), before)

    def test_contract_is_explicit_without_claiming_source_parser(self):
        self.assertTrue(self.codec.parser_integrated)
        self.assertFalse(self.binding.semantic_record_parser_integrated)
        self.assertEqual(self.codec.schema_version, SEMANTIC_RECORD_SCHEMA)
        self.assertEqual(len(self.codec.allowed_events), 26)
        self.assertIn("UNKNOWN_SEMANTIC", self.codec.allowed_events)

    def test_invalid_typed_records_fail_closed(self):
        invalid = (
            SemanticRecord(event="call", surface="build", confidence_band="high"),
            SemanticRecord(event="CALL", surface="", confidence_band="high"),
            SemanticRecord(event="CALL", surface="build", confidence_band="certain"),
            SemanticRecord(event="CALL", surface="build", confidence_band="high", schema_version="v2"),
        )
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.codec.serialize(record)
        with self.assertRaises(TypeError):
            self.codec.serialize({"event": "CALL"})

    def test_malformed_or_ambiguous_serialization_fails_closed(self):
        valid = self.codec.serialize(SemanticRecord("CALL", "build", "high"))
        payload = valid.split("\n")[2]
        malformed = (
            valid + "\n",
            valid.replace("record:begin", "stream:begin", 1),
            valid.replace('"event":"CALL"', '"event":"CALL","event":"RETURN"'),
            valid.replace(payload, payload[:-1] + ',"extra":true}'),
            valid.replace('"event":"CALL"', '"event": "CALL"'),
        )
        for serialized in malformed:
            with self.subTest(serialized=serialized), self.assertRaises(ValueError):
                self.codec.parse(serialized)

    def test_binding_and_id_tampering_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "verified native 65K"):
            NativeSemanticRecordCodec(replace(self.binding, tokenizer_sha256="0" * 64))
        with self.assertRaises(ValueError):
            self.codec.decode([65_536])


if __name__ == "__main__":
    unittest.main()
