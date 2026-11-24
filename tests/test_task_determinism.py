from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
import zlib
from array import array
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_TOKENIZERS_SITE = PROJECT_ROOT / "vendor" / "tokenizers_runtime_0_23"
if str(PINNED_TOKENIZERS_SITE) not in sys.path:
    sys.path.insert(0, str(PINNED_TOKENIZERS_SITE))

from bosscrafts_model.bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from bosscrafts_model.resumable_task_packing import TASK_DATABASE_SCHEMA
from bosscrafts_model.task_determinism import audit_task_prompt_determinism
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class TaskPromptDeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()

    def _build_database(self, path: Path) -> None:
        rows = (
            ("doc-a", "python", "code_to_semantic", "train", [1, 2, 10, 11]),
            ("doc-b", "python", "code_to_semantic", "test", [1, 2, 10, 11]),
            ("doc-c", "ruby", "code_to_semantic", "validation", [1, 2, 12, 13]),
            ("doc-d", "python", "semantic_to_code", "train", [1, 2, 12, 13]),
            ("doc-e", "python", "semantic_continuation", "train", [3, 4, 14, 15]),
        )
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO metadata VALUES(?,?)",
                sorted(
                    {
                        "schema_version": TASK_DATABASE_SCHEMA,
                        "binding_sha256": self.binding.binding_sha256,
                        "tokenizer_sha256": self.binding.tokenizer_sha256,
                        "vocabulary_size": str(self.binding.vocab_size),
                        "packing_complete": "true",
                        "maximum_sequence_tokens": "4",
                        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                    }.items()
                ),
            )
            connection.execute(
                """CREATE TABLE tasks (
                       example_id TEXT PRIMARY KEY,
                       document_id TEXT NOT NULL,
                       language TEXT NOT NULL,
                       family TEXT NOT NULL,
                       record_index INTEGER NOT NULL,
                       split TEXT NOT NULL,
                       input_ids_zlib BLOB NOT NULL,
                       token_count INTEGER NOT NULL,
                       loss_start_index INTEGER NOT NULL,
                       semantic_spec_sha256 TEXT NOT NULL,
                       parser_runtime_identity_sha256 TEXT NOT NULL,
                       source_sha256 TEXT NOT NULL
                   )"""
            )
            packed_rows = []
            for document_id, language, family, split, input_ids in rows:
                input_sha256 = hashlib.sha256(
                    _canonical_json(input_ids).encode("utf-8")
                ).hexdigest()
                example_id = hashlib.sha256(
                    _canonical_json(
                        {
                            "document_id": document_id,
                            "family": family,
                            "record_index": 0,
                            "input_ids_sha256": input_sha256,
                        }
                    ).encode("utf-8")
                ).hexdigest()
                packed_rows.append(
                    (
                        example_id,
                        document_id,
                        language,
                        family,
                        0,
                        split,
                        zlib.compress(array("H", input_ids).tobytes()),
                        len(input_ids),
                        2,
                        "c" * 64,
                        PARSER_RUNTIME_IDENTITY_SHA256,
                        hashlib.sha256(document_id.encode("utf-8")).hexdigest(),
                    )
                )
            connection.executemany("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", packed_rows)
            connection.commit()
        finally:
            connection.close()

    def test_detects_global_conflicts_leakage_and_family_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tasks.sqlite"
            self._build_database(database)
            result = audit_task_prompt_determinism(
                self.binding, database, sample_limit=5
            )

        summary = result["summary"]
        self.assertEqual(summary["example_count"], 5)
        self.assertEqual(summary["unique_prompt_count"], 3)
        self.assertEqual(summary["duplicate_prompt_group_count"], 1)
        self.assertEqual(summary["conflicting_prompt_group_count"], 1)
        self.assertEqual(summary["rows_in_conflicting_prompt_groups"], 3)
        self.assertEqual(summary["cross_split_prompt_group_count"], 1)
        self.assertEqual(summary["cross_language_prompt_group_count"], 1)
        self.assertFalse(result["prompt_targets_deterministic"])
        self.assertFalse(result["heldout_prompt_isolation"])
        self.assertEqual(len(result["samples"]), 1)
        self.assertEqual(result["samples"][0]["distinct_target_count"], 2)
        self.assertEqual(
            result["families"]["semantic_to_code"]["conflicting_prompt_group_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
