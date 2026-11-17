import hashlib
import json
import unittest
from pathlib import Path

from bosscrafts_model.bcnlt_adapter import LANGUAGE_SPEC_SHA256
from bosscrafts_model.bcnlt_recovery import load_bcnlt_recovery_status
from bosscrafts_model.tokenizer import (
    EXPECTED_BINDING_SHA256,
    EXPECTED_TOKENIZER_SHA256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ROOT = PROJECT_ROOT / "artifacts" / "bcnlt" / "runtime-recovery-v1"


class BCNLTRuntimeRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (RECOVERY_ROOT / "recovery-manifest.json").read_text(encoding="utf-8")
        )
        cls.specs = json.loads(
            (RECOVERY_ROOT / "semantic-specs.json").read_text(encoding="utf-8")
        )

    def test_recovered_source_is_real_but_runtime_remains_fail_closed(self):
        status = load_bcnlt_recovery_status()
        self.assertTrue(status.source_contract_recovered)
        self.assertTrue(status.output_consumer_integrated)
        self.assertFalse(status.live_parser_ready)
        self.assertFalse(status.training_ready)
        self.assertEqual(len(status.blockers), 4)
        self.assertEqual(self.manifest["runtime"]["grammar_dlls_found"], 0)
        self.assertEqual(self.manifest["runtime"]["grammar_dlls_required"], 14)
        self.assertFalse(self.manifest["runtime"]["native_abi_compatible"])
        self.assertFalse(self.manifest["runtime"]["live_parser_executed"])

    def test_current_source_specs_do_not_masquerade_as_admitted_lineage(self):
        comparison = self.manifest["parser_contract"]["semantic_spec_comparison"]
        self.assertEqual(set(comparison), set(LANGUAGE_SPEC_SHA256))
        self.assertFalse(self.manifest["parser_contract"]["all_specs_match_admitted_output"])
        for language, expected_admitted_hash in LANGUAGE_SPEC_SHA256.items():
            with self.subTest(language=language):
                self.assertEqual(comparison[language]["admitted_output_sha256"], expected_admitted_hash)
                self.assertEqual(
                    comparison[language]["recovered_source_sha256"],
                    self.specs["specs"][language]["canonical_sha256"],
                )
                self.assertFalse(comparison[language]["matches"])

    def test_admitted_dataset_is_evidence_not_65k_training_input(self):
        corpus = self.manifest["admitted_corpus"]
        self.assertEqual(corpus["accepted_record_count"], 987_342)
        self.assertEqual(corpus["packed_training_records"], 901_645)
        self.assertEqual(corpus["languages_passed"], 14)
        self.assertEqual(corpus["packed_tokenizer_vocab_size"], 32_073)
        self.assertFalse(corpus["bulk_accepted_jsonl_copied"])
        self.assertFalse(
            self.manifest["readiness"]["admitted_corpus_tokenizer_compatible"]
        )

    def test_frozen_65k_identity_is_unchanged(self):
        frozen = self.manifest["frozen_model_identity"]
        self.assertEqual(frozen["tokenizer_vocab_size"], 65_536)
        self.assertEqual(frozen["tokenizer_sha256"], EXPECTED_TOKENIZER_SHA256)
        self.assertEqual(frozen["binding_sha256"], EXPECTED_BINDING_SHA256)

    def test_copied_parser_config_has_observed_bundle_hash(self):
        config = RECOVERY_ROOT / "source" / "configs" / "bcnlt_parser_bundle_v1.json"
        self.assertEqual(
            hashlib.sha256(config.read_bytes()).hexdigest(),
            self.manifest["parser_contract"]["parser_config_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
