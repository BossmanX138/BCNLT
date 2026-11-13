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
from bosscrafts_model.deterministic_curriculum import derive_deterministic_curriculum
from bosscrafts_model.resumable_task_packing import TASK_DATABASE_SCHEMA
from bosscrafts_model.task_determinism import audit_task_prompt_determinism
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class DeterministicCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()

    def _write_database(self, path: Path) -> None:
        rows: list[tuple[str, str, str, list[int]]] = []
        next_token = 100
        for family in ("code_to_semantic", "semantic_continuation", "semantic_to_code"):
            for split in ("train", "validation", "test"):
                rows.append(
                    (
                        f"unique-{family}-{split}",
                        family,
                        split,
                        [next_token, next_token + 1, next_token + 2, next_token + 3],
                    )
                )
                next_token += 4
        rows.extend(
            [
                ("conflict-a", "code_to_semantic", "train", [400, 401, 402, 403]),
                ("conflict-b", "code_to_semantic", "train", [400, 401, 404, 405]),
                ("leak-a", "semantic_to_code", "train", [410, 411, 412, 413]),
                ("leak-b", "semantic_to_code", "test", [410, 411, 412, 413]),
                ("duplicate-a", "semantic_continuation", "train", [420, 421, 422, 423]),
                ("duplicate-b", "semantic_continuation", "train", [420, 421, 422, 423]),
            ]
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
                        "semantic_spec_sha256": "spec",
                        "source_catalog_sha256": "catalog",
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
            for document_id, family, split, input_ids in rows:
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
                        "python",
                        family,
                        0,
                        split,
                        zlib.compress(array("H", input_ids).tobytes()),
                        4,
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

    def test_filters_ambiguous_and_cross_split_prompts_then_balances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sqlite"
            destination = root / "clean.sqlite"
            self._write_database(source)
            status = derive_deterministic_curriculum(
                self.binding,
                source,
                destination,
                languages=("python",),
                quotas={"train": 1, "validation": 1, "test": 1},
            )
            audit = audit_task_prompt_determinism(
                self.binding, destination, sample_limit=5
            )

        self.assertEqual(status.selected_example_count, 9)
        self.assertEqual(status.conflicting_prompt_group_count, 1)
        self.assertEqual(status.conflicting_prompt_row_count, 2)
        self.assertEqual(status.cross_split_prompt_group_count, 1)
        self.assertEqual(status.cross_split_prompt_row_count, 2)
        self.assertEqual(status.excluded_prompt_group_count, 2)
        self.assertEqual(status.excluded_prompt_row_count, 4)
        self.assertEqual(set(status.cell_counts.values()), {1})
        self.assertTrue(audit["prompt_targets_deterministic"])
        self.assertTrue(audit["heldout_prompt_isolation"])
        self.assertEqual(audit["summary"]["unique_prompt_count"], 9)


if __name__ == "__main__":
    unittest.main()
