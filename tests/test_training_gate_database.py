from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.current_task_packing import (
    EXAMPLE_SCHEMA,
    PACKED_MANIFEST_SCHEMA,
    CurrentCorpusTaskPacker,
)
from bosscrafts_model.resumable_task_packing import audit_task_database
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def example(binding: NativeTokenizerBinding, *, document_id: str, spec: str) -> dict[str, object]:
    structural = binding.structural_token_ids
    raw_stream = [
        structural["<|bcnlt:stream:begin|>"],
        structural["<|bcnlt:mode:raw_code|>"],
        *binding.encode_payload("x"),
        structural["<|bcnlt:stream:end|>"],
    ]
    semantic_stream = [
        structural["<|bcnlt:stream:begin|>"],
        structural["<|bcnlt:mode:semantic_code|>"],
        structural["<|bcnlt:stream:end|>"],
    ]
    input_ids = [
        structural["<|bcnlt:task:code_to_semantic|>"],
        *raw_stream,
        *semantic_stream,
        structural["<|bcnlt:task:end|>"],
    ]
    loss_start_index = 1 + len(raw_stream)
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
        "loss_start_index": loss_start_index,
        "record_index": 0,
        "schema_version": EXAMPLE_SCHEMA,
        "semantic_spec_sha256": spec,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "source_sha256": "a" * 64,
        "split": CurrentCorpusTaskPacker._split(document_id),
        "target_token_count": len(input_ids) - loss_start_index,
        "tokenizer_sha256": binding.tokenizer_sha256,
    }


def write_source_manifest(
    root: Path,
    binding: NativeTokenizerBinding,
    rows: list[dict[str, object]],
) -> Path:
    examples = root / "examples.jsonl"
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
    manifest = {
        **unsigned,
        "manifest_sha256": hashlib.sha256(canonical(unsigned).encode()).hexdigest(),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(canonical(manifest) + "\n", encoding="utf-8")
    return manifest_path


class TrainingGateDatabaseTests(unittest.TestCase):
    def test_imports_only_current_lineage_rows_into_audited_gate_database(self):
        from bosscrafts_model.training_gate_database import build_training_gate_database

        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                example(
                    binding,
                    document_id=hashlib.sha256(b"doc-current").hexdigest(),
                    spec=CURRENT_RUNTIME_SPEC_SHA256["bash"],
                ),
                example(
                    binding,
                    document_id=hashlib.sha256(b"doc-stale-0").hexdigest(),
                    spec="f" * 64,
                ),
            ]
            manifest_path = write_source_manifest(root, binding, rows)
            examples = root / "examples.jsonl"
            database = root / "gate.sqlite"
            legacy_temporary = root / ".gate.sqlite.tmp"
            legacy_temporary.write_bytes(b"unrelated sentinel")

            status = build_training_gate_database(
                binding,
                manifest_path,
                database,
                max_examples=4,
            )

            self.assertEqual(status.imported_count, 1)
            self.assertEqual(status.stale_spec_skipped_count, 1)
            self.assertEqual(legacy_temporary.read_bytes(), b"unrelated sentinel")
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

    def test_invalid_task_family_is_rejected_before_database_publication(self):
        from bosscrafts_model.training_gate_database import build_training_gate_database

        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = example(
                binding,
                document_id=hashlib.sha256(b"doc-current").hexdigest(),
                spec=CURRENT_RUNTIME_SPEC_SHA256["bash"],
            )
            row["family"] = "unrecognized_training_task"
            identity = {
                "document_id": row["document_id"],
                "family": row["family"],
                "record_index": row["record_index"],
                "input_ids_sha256": hashlib.sha256(
                    canonical(row["input_ids"]).encode()
                ).hexdigest(),
            }
            row["example_id"] = hashlib.sha256(canonical(identity).encode()).hexdigest()
            manifest_path = write_source_manifest(root, binding, [row])
            database = root / "invalid.sqlite"

            with self.assertRaisesRegex(ValueError, "family"):
                build_training_gate_database(binding, manifest_path, database)
            self.assertFalse(database.exists())


if __name__ == "__main__":
    unittest.main()
