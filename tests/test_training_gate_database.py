from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.current_task_packing import EXAMPLE_SCHEMA, PACKED_MANIFEST_SCHEMA
from bosscrafts_model.resumable_task_packing import audit_task_database
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def example(binding: NativeTokenizerBinding, *, document_id: str, spec: str) -> dict[str, object]:
    input_ids = [1, 2, 3, 4]
    identity = {
        "document_id": document_id,
        "family": "code_to_semantic",
        "record_index": 0,
        "input_ids_sha256": hashlib.sha256(canonical(input_ids).encode()).hexdigest(),
    }
    return {
        "binding_sha256": binding.binding_sha256,
        "document_id": document_id,
        "example_id": hashlib.sha256(canonical(identity).encode()).hexdigest(),
        "family": "code_to_semantic",
        "input_ids": input_ids,
        "language": "bash",
        "loss_start_index": 2,
        "record_index": 0,
        "schema_version": EXAMPLE_SCHEMA,
        "semantic_spec_sha256": spec,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "source_sha256": "a" * 64,
        "split": "train",
        "tokenizer_sha256": binding.tokenizer_sha256,
    }


class TrainingGateDatabaseTests(unittest.TestCase):
    def test_imports_only_current_lineage_rows_into_audited_gate_database(self):
        from bosscrafts_model.training_gate_database import build_training_gate_database

        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = root / "examples.jsonl"
            rows = [
                example(
                    binding,
                    document_id=hashlib.sha256(b"doc-current").hexdigest(),
                    spec=CURRENT_RUNTIME_SPEC_SHA256["bash"],
                ),
                example(
                    binding,
                    document_id=hashlib.sha256(b"doc-stale").hexdigest(),
                    spec="f" * 64,
                ),
            ]
            examples.write_text(
                "".join(canonical(row) + "\n" for row in rows), encoding="utf-8"
            )
            unsigned = {
                "binding_sha256": binding.binding_sha256,
                "examples_file": examples.name,
                "examples_sha256": hashlib.sha256(examples.read_bytes()).hexdigest(),
                "maximum_sequence_tokens": 2048,
                "schema_version": PACKED_MANIFEST_SCHEMA,
                "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                "tokenizer_sha256": binding.tokenizer_sha256,
            }
            manifest = {**unsigned, "manifest_sha256": hashlib.sha256(canonical(unsigned).encode()).hexdigest()}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(canonical(manifest) + "\n", encoding="utf-8")
            database = root / "gate.sqlite"

            status = build_training_gate_database(
                binding,
                manifest_path,
                database,
                max_examples=4,
            )

            self.assertEqual(status.imported_count, 1)
            self.assertEqual(status.stale_spec_skipped_count, 1)
            audit = audit_task_database(binding, database)
            self.assertEqual(audit.example_count, 1)

            examples.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "examples SHA-256"):
                build_training_gate_database(
                    binding,
                    manifest_path,
                    root / "tampered.sqlite",
                    max_examples=1,
                )


if __name__ == "__main__":
    unittest.main()
