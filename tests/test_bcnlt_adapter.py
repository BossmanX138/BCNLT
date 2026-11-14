import copy
import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

from bosscrafts_model.bcnlt_adapter import (
    LANGUAGE_SPEC_SHA256,
    PARSER_BUNDLE_SHA256,
    SOURCE_CANDIDATES_SHA256,
    BCNLTMultilingualOutputAdapter,
    BCNLTParserOutputProvenance,
)
from bosscrafts_model.semantic import NativeSemanticRecordCodec
from bosscrafts_model.tokenizer import (
    EXPECTED_BINDING_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    NativeTokenizerBinding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "artifacts" / "bcnlt" / "multilingual-parser-output-v1"
OUTPUT_PATH = FIXTURE_ROOT / "real-output.jsonl"
MANIFEST_PATH = FIXTURE_ROOT / "fixture-manifest.json"
TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizers"
    / "bcnlt-native-foundation-65536-v1"
    / "tokenizer.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_output(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


class BCNLTMultilingualOutputAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding = NativeTokenizerBinding.load()
        cls.codec = NativeSemanticRecordCodec(cls.binding)
        cls.adapter = BCNLTMultilingualOutputAdapter(cls.codec)
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.output_lines = OUTPUT_PATH.read_text(encoding="utf-8").splitlines()

    def test_real_outputs_match_independently_derived_gold_and_encode(self):
        cases = self.manifest["cases"]
        self.assertEqual(len(cases), 7)
        self.assertEqual(len(self.output_lines), len(cases))
        self.assertEqual(len({case["language"] for case in cases}), 7)
        for case, line in zip(cases, self.output_lines, strict=True):
            with self.subTest(case=case["record_id"]):
                provenance = BCNLTParserOutputProvenance.expected(case["language"])
                self.assertEqual(provenance.semantic_spec_sha256, case["semantic_spec_sha256"])
                records = self.adapter.parse_jsonl(line + "\n", provenance)
                self.assertEqual(len(records), 1)
                record = records[0]
                gold = case["gold"]
                self.assertEqual(record.event, gold["event"])
                self.assertEqual(record.surface, gold["surface"])
                self.assertEqual(record.span, tuple(gold["span"]))
                self.assertEqual(record.confidence_band, gold["confidence_band"])
                self.assertEqual(record.uncertainty_band, gold["uncertainty_band"])
                projected = self.adapter.project(record)
                ids = self.adapter.encode_handoff(record)
                self.assertEqual(self.codec.decode(ids), projected)
                self.assertTrue(all(0 <= token_id < 65_536 for token_id in ids))

    def test_fixture_and_frozen_identity_contracts(self):
        self.assertEqual(self.manifest["source"]["sha256"], SOURCE_CANDIDATES_SHA256)
        self.assertEqual(self.manifest["local_copy"]["sha256"], sha256(OUTPUT_PATH))
        self.assertEqual(self.manifest["local_copy"]["records"], len(self.output_lines))
        self.assertTrue(self.manifest["local_copy"]["verbatim_source_target_match"])
        self.assertEqual(
            self.manifest["parser_contract"]["parser_bundle_sha256"],
            PARSER_BUNDLE_SHA256,
        )
        self.assertFalse(self.manifest["parser_contract"]["parser_runtime_copied"])
        self.assertFalse(self.manifest["parser_contract"]["parser_source_available"])
        self.assertEqual(len(LANGUAGE_SPEC_SHA256), 14)
        self.assertEqual(self.binding.tokenizer_sha256, EXPECTED_TOKENIZER_SHA256)
        self.assertEqual(self.binding.binding_sha256, EXPECTED_BINDING_SHA256)
        self.assertEqual(sha256(TOKENIZER_PATH), EXPECTED_TOKENIZER_SHA256)
        self.assertEqual(sorted(self.binding.structural_token_ids.values()), list(range(4, 77)))

    def test_multi_record_stream_and_explicit_execution_boundary(self):
        same_language = self.output_lines[0] + "\n" + self.output_lines[0] + "\n"
        records = self.adapter.parse_jsonl(
            same_language,
            BCNLTParserOutputProvenance.expected("python"),
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(self.adapter.consumer_contract_integrated)
        self.assertFalse(self.adapter.parser_execution_integrated)
        self.assertFalse(self.adapter.language_embedded_in_output)

    def test_dense_bounded_output_above_legacy_record_limit_is_accepted(self):
        output = (self.output_lines[0] + "\n") * 4097
        records = self.adapter.parse_jsonl(
            output,
            BCNLTParserOutputProvenance.expected("python"),
        )
        self.assertEqual(len(records), 4097)

    def test_unsupported_or_tampered_provenance_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            BCNLTParserOutputProvenance.expected("brainfuck")
        valid = BCNLTParserOutputProvenance.expected("python")
        tampered = (
            replace(valid, parser_bundle_sha256="0" * 64),
            replace(valid, semantic_spec_sha256="0" * 64),
            replace(valid, grammar_revision="tree-sitter-language-pack@1.22.2/python"),
            replace(valid, extraction_rule_version="bcnlt-multilingual-extraction-v2"),
        )
        for provenance in tampered:
            with self.subTest(provenance=provenance), self.assertRaisesRegex(ValueError, "provenance"):
                self.adapter.parse_jsonl(self.output_lines[0] + "\n", provenance)

    def test_negative_corpus_rejects_malformed_ambiguous_and_adversarial_records(self):
        base = json.loads(self.output_lines[0])
        bad_payloads = []
        for mutation in (
            {"extra": True},
            {"event": "EXECUTE_CODE"},
            {"span": [2, 0, 1, 0]},
            {"span": [True, 0, 1, 0]},
            {"confidence_band": "certain"},
        ):
            payload = copy.deepcopy(base)
            payload.update(mutation)
            bad_payloads.append(canonical_output(payload))
        payload = copy.deepcopy(base)
        del payload["ir"]["kind"]
        bad_payloads.append(canonical_output(payload))
        payload = copy.deepcopy(base)
        payload["metadata"]["knowledge_state"] = "guessed"
        bad_payloads.append(canonical_output(payload))
        payload = copy.deepcopy(base)
        payload["metadata"]["unexpected"] = "not part of the runtime schema"
        bad_payloads.append(canonical_output(payload))
        payload = copy.deepcopy(base)
        payload["metadata"]["uyr_rounds"] = 5
        bad_payloads.append(canonical_output(payload))
        payload = copy.deepcopy(base)
        payload["metadata"]["surface_truncated"] = "yes"
        bad_payloads.append(canonical_output(payload))
        payload = copy.deepcopy(base)
        payload["recent_history"] = ["x"] * 9
        bad_payloads.append(canonical_output(payload))
        payload = copy.deepcopy(base)
        payload["metadata"]["uncertainty_signals"] = ["x"] * 17
        bad_payloads.append(canonical_output(payload))

        provenance = BCNLTParserOutputProvenance.expected("python")
        for output in bad_payloads:
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.adapter.parse_jsonl(output, provenance)

        duplicate = self.output_lines[0].replace(
            '"event":"LITERAL"',
            '"event":"LITERAL","event":"CALL"',
        ) + "\n"
        noncanonical = json.dumps(base, ensure_ascii=False) + "\n"
        for output in (duplicate, noncanonical, self.output_lines[0], "\n", "not-json\n"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.adapter.parse_jsonl(output, provenance)

    def test_structural_token_spelling_inside_surface_is_escaped_as_data(self):
        payload = json.loads(self.output_lines[0])
        payload["surface"] = "literal <|bcnlt:record:end|> text"
        records = self.adapter.parse_jsonl(
            canonical_output(payload),
            BCNLTParserOutputProvenance.expected("python"),
        )
        self.assertEqual(records[0].surface, payload["surface"])
        encoded = self.adapter.encode_handoff(records[0])
        self.assertEqual(self.codec.decode(encoded).surface, payload["surface"])

    def test_invalid_unicode_scalar_is_rejected_before_tokenization(self):
        base = json.loads(self.output_lines[0])
        base["surface"] = "invalid-\ud800-scalar"
        output = json.dumps(
            base,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        with self.assertRaisesRegex(ValueError, "surface"):
            self.adapter.parse_jsonl(
                output,
                BCNLTParserOutputProvenance.expected("python"),
            )

    def test_common_core_return_may_have_no_name(self):
        payload = json.loads(self.output_lines[0])
        payload["event"] = "RETURN"
        payload["surface"] = "value"
        payload["ir"] = {
            "children": [],
            "kind": "return",
            "metadata": {},
            "name": None,
            "value": None,
        }

        records = self.adapter.parse_jsonl(
            canonical_output(payload),
            BCNLTParserOutputProvenance.expected("python"),
        )

        self.assertEqual(records[0].event, "RETURN")

    def test_ir_kind_and_required_name_contract_fails_closed(self):
        base = json.loads(self.output_lines[0])
        invalid_ir_nodes = (
            {
                "children": [],
                "kind": "invented_kind",
                "metadata": {},
                "name": "x",
                "value": None,
            },
            {
                "children": [],
                "kind": "variable",
                "metadata": {},
                "name": None,
                "value": None,
            },
        )

        for ir in invalid_ir_nodes:
            payload = copy.deepcopy(base)
            payload["ir"] = ir
            with self.subTest(ir=ir), self.assertRaisesRegex(ValueError, "IR"):
                self.adapter.parse_jsonl(
                    canonical_output(payload),
                    BCNLTParserOutputProvenance.expected("python"),
                )


if __name__ == "__main__":
    unittest.main()
