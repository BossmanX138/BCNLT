from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from runtime_import import select_bcnlt_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = (
    PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate" / "source"
)
# These cases originated during the V3 audit, but they are current-runtime
# regressions. Binding them to an ignored historical checkout made a fresh
# clone silently test different code from the candidate under review.
select_bcnlt_runtime(RUNTIME_SOURCE)

from bcnlt.canonical import (  # noqa: E402
    MAX_CANONICAL_RECORD_BYTES,
    record_from_canonical_bytes,
)
from bcnlt.full_run_gate import _load_object, prepare_full_run  # noqa: E402
from bcnlt.hf_tokenizer import HuggingFaceBaseTokenizer  # noqa: E402
from bcnlt.model_eval import _urlopen_transport  # noqa: E402
from bcnlt.wildcard import WildcardMemory  # noqa: E402


class BCNLTRuntimeV3CanonicalSafetyTests(unittest.TestCase):
    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            record_from_canonical_bytes(
                b'{"event":"DEFINE","event":"REFERENCE"}\n'
            )

    def test_oversized_record_is_rejected_before_json_parsing(self):
        with self.assertRaisesRegex(ValueError, "invalid byte length"):
            record_from_canonical_bytes(b" " * (MAX_CANONICAL_RECORD_BYTES + 1))

    def test_boolean_span_coordinate_is_not_treated_as_an_integer(self):
        payload = {
            "ast_relation": None,
            "confidence_band": "high",
            "defines": None,
            "event": "DEFINE",
            "ir": None,
            "metadata": {},
            "recent_history": [],
            "scope": "module",
            "span": [True, 0, 1, 1],
            "surface": "x",
            "symbol_id": None,
            "uncertainty_band": "low",
        }
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "span must contain"):
            record_from_canonical_bytes(encoded)

    def test_full_run_execution_requires_an_external_handoff_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            handoff = Path(temporary) / "handoff.json"
            handoff.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "externally approved handoff"):
                prepare_full_run(
                    handoff,
                    disk_free_bytes=0,
                    gpu_memory_mib=0,
                    start_training=True,
                )

    def test_full_run_handoff_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            handoff = Path(temporary) / "handoff.json"
            handoff.write_text(
                '{"schema_version":"a","schema_version":"b"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unambiguous"):
                prepare_full_run(
                    handoff,
                    disk_free_bytes=0,
                    gpu_memory_mib=0,
                )

    def test_full_run_rejects_boolean_resource_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            handoff = Path(temporary) / "handoff.json"
            handoff.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disk space"):
                prepare_full_run(
                    handoff,
                    disk_free_bytes=True,
                    gpu_memory_mib=0,
                )

    def test_full_run_artifact_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact.json"
            artifact.write_text('{"passed":true,"passed":false}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unambiguous"):
                _load_object(artifact)

    def test_learned_memory_rejects_ambiguous_complete_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "learned.jsonl"
            ledger.write_text(
                '{"signature":"a","signature":"b"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Malformed learned-memory"):
                WildcardMemory(ledger)

    def test_learned_memory_only_recovers_an_unterminated_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "learned.jsonl"
            ledger.write_text('{"signature":', encoding="utf-8")
            memory = WildcardMemory(ledger)
            self.assertEqual(len(memory.load_warnings), 1)
            self.assertEqual(memory.records, {})

    def test_learned_memory_rejects_cross_language_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "learned.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "signature": "javascript|legacy-python|UNKNOWN_SEMANTIC|x|y",
                        "language": "javascript",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "language mismatch"):
                WildcardMemory(ledger)

    def test_learned_memory_rejects_non_list_coding_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "learned.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "signature": "python|legacy-python|UNKNOWN_SEMANTIC|x|y",
                        "coding_outcomes": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "coding outcomes"):
                WildcardMemory(ledger)

    def test_fast_tokenizer_offsets_are_validated(self):
        class FakeFastTokenizer:
            is_fast = True

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": [3], "offset_mapping": [(0, 99)]}

        tokenizer = HuggingFaceBaseTokenizer(FakeFastTokenizer())
        with self.assertRaisesRegex(ValueError, "offset span"):
            tokenizer.encode_with_offsets("short")

    def test_fast_tokenizer_boolean_ids_are_rejected(self):
        class FakeFastTokenizer:
            is_fast = True

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": [True], "offset_mapping": [(0, 1)]}

        tokenizer = HuggingFaceBaseTokenizer(FakeFastTokenizer())
        with self.assertRaisesRegex(ValueError, "token IDs"):
            tokenizer.encode_with_offsets("x")

    def test_model_transport_rejects_non_http_endpoint_before_opening_it(self):
        with self.assertRaisesRegex(ValueError, "HTTP"):
            _urlopen_transport("file:///etc/passwd", {}, {}, 1.0)


if __name__ == "__main__":
    unittest.main()
