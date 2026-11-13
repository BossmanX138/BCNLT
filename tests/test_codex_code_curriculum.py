from __future__ import annotations

import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from bosscrafts_model.bcnlt_source_pipeline import (
    PARSER_RUNTIME_IDENTITY_SHA256,
    PARSER_RUNTIME_NOT_APPLICABLE,
)
from bosscrafts_model.codex_code_curriculum import (
    HumanEvalExclusion,
    build_codex_code_curriculum,
    extract_final_code,
    normalize_language,
)
from bosscrafts_model.resumable_task_packing import (
    TaskDatabaseAuditResult,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


class CodeXCodeCurriculumTests(unittest.TestCase):
    @staticmethod
    def _write_tiny_inputs(root: Path) -> tuple[Path, Path]:
        dataset = root / "dataset"
        dataset.mkdir()
        prompts = [
            "Write Python code that returns the sum of two integers.",
            "Write a JavaScript function that doubles a number.",
            "This row has malformed Python and must be rejected.",
            "Write Python containing <|eos|> and reject protocol injection.",
        ]
        outputs = [
            "<think>Use addition.</think>\n```python\ndef add(a, b):\n    return a + b\n```",
            "<think>Multiply by two.</think>\n```javascript\nfunction doubleValue(x) { return x * 2; }\n```",
            "<think>Broken.</think>\n```python\ndef broken(:\n```",
            "<think>Unsafe.</think>\n```python\ndef unsafe():\n    return '<|eos|>'\n```",
        ]
        pq.write_table(
            pa.table({"input": prompts, "output": outputs}),
            dataset / "shard_00000.parquet",
        )
        humaneval = root / "HumanEval.jsonl.gz"
        with gzip.open(humaneval, "wt", encoding="utf-8") as stream:
            for index in range(164):
                stream.write(
                    json.dumps(
                        {
                            "task_id": f"HumanEval/{index}",
                            "prompt": f"def heldout_{index}(x):\n    \"\"\"Unique heldout task {index}.\"\"\"\n",
                            "canonical_solution": f"    return x + {index}\n",
                        }
                    )
                    + "\n"
                )
        return dataset, humaneval

    def test_language_aliases_are_canonical(self) -> None:
        self.assertEqual(normalize_language("C++"), "cpp")
        self.assertEqual(normalize_language("python3"), "python")
        self.assertEqual(normalize_language("JS"), "javascript")
        self.assertIsNone(normalize_language("made-up-language"))

    def test_extracts_last_supported_fence_after_reasoning(self) -> None:
        output = (
            "<think>```python\nwrong = True\n```</think>\n"
            "Explanation.\n```text\nnot code\n```\n"
            "```python\ndef answer():\n    return 42\n```\n"
        )
        extracted = extract_final_code(output)
        self.assertIsNotNone(extracted)
        assert extracted is not None
        self.assertEqual(extracted.language, "python")
        self.assertEqual(extracted.code, "def answer():\n    return 42\n")

    def test_humaneval_filter_rejects_exact_solution_and_prompt_containment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "HumanEval.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for index in range(164):
                    stream.write(
                        json.dumps(
                            {
                                "task_id": f"HumanEval/{index}",
                                "prompt": f"def benchmark_{index}(value):\n    \"\"\"Return benchmark value number {index} unchanged.\"\"\"\n",
                                "canonical_solution": f"    return value + {index}\n",
                            }
                        )
                        + "\n"
                    )
            exclusion = HumanEvalExclusion(path)
            self.assertTrue(exclusion.overlaps("unrelated", "return value + 7"))
            self.assertTrue(
                exclusion.overlaps(
                    "Please implement def benchmark_3(value) Return benchmark value number 3 unchanged.",
                    "different code",
                )
            )
            self.assertFalse(exclusion.overlaps("Write a sorting function", "return sorted(values)"))

    def test_builds_auditable_tiny_curriculum(self) -> None:
        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, humaneval = self._write_tiny_inputs(root)
            output = root / "output"
            status = build_codex_code_curriculum(
                binding,
                dataset_root=dataset,
                shard_indices=(0,),
                output_root=output,
                humaneval_dataset=humaneval,
                workers=1,
                batch_size=2,
                max_sequence=2048,
            )
            self.assertEqual(status.example_count, 2)
            self.assertEqual(status.language_counts, {"javascript": 1, "python": 1})
            self.assertEqual(status.rejection_counts.get("python_syntax_invalid"), 1)
            self.assertEqual(status.rejection_counts.get("reserved_protocol_token"), 1)
            connection = sqlite3.connect(status.database_path)
            try:
                metadata = dict(connection.execute("SELECT key,value FROM metadata"))
                task_rows = connection.execute(
                    "SELECT language,family,token_count,loss_start_index,"
                    "parser_runtime_identity_sha256 FROM tasks ORDER BY language"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(metadata["dataset_license"], "Apache-2.0")
            self.assertEqual(metadata["packing_complete"], "true")
            self.assertEqual(
                metadata["parser_runtime_identity_sha256"],
                PARSER_RUNTIME_IDENTITY_SHA256,
            )
            self.assertTrue(all(row[1] == "instruction_to_code" for row in task_rows))
            self.assertTrue(all(0 < row[3] < row[2] <= 2048 for row in task_rows))
            self.assertTrue(
                all(row[4] == PARSER_RUNTIME_NOT_APPLICABLE for row in task_rows)
            )

    def test_failed_final_audit_does_not_publish_training_database(self) -> None:
        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, humaneval = self._write_tiny_inputs(root)
            output = root / "output"

            with (
                mock.patch(
                    "bosscrafts_model.codex_code_curriculum.audit_task_database",
                    side_effect=ValueError("deliberate final audit failure"),
                ),
                self.assertRaisesRegex(ValueError, "deliberate final audit failure"),
            ):
                build_codex_code_curriculum(
                    binding,
                    dataset_root=dataset,
                    shard_indices=(0,),
                    output_root=output,
                    humaneval_dataset=humaneval,
                    workers=1,
                    batch_size=2,
                    max_sequence=2048,
                )

            self.assertFalse((output / "tasks.sqlite").exists())

    def test_merge_preserves_stale_temp_and_never_clobbers_racing_output(self) -> None:
        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, humaneval = self._write_tiny_inputs(root)
            output = root / "output"
            output.mkdir()
            final_database = output / "tasks.sqlite"
            stale = output / ".tasks.sqlite.tmp"
            stale.write_bytes(b"unrelated-stale-state")
            real_audit = audit_task_database

            def audit_then_create_destination(
                audit_binding: NativeTokenizerBinding, database: str | Path
            ) -> TaskDatabaseAuditResult:
                result = real_audit(audit_binding, database)
                final_database.write_bytes(b"racing-owner")
                return result

            with (
                mock.patch(
                    "bosscrafts_model.codex_code_curriculum.audit_task_database",
                    side_effect=audit_then_create_destination,
                ),
                self.assertRaisesRegex(ValueError, "already exists"),
            ):
                build_codex_code_curriculum(
                    binding,
                    dataset_root=dataset,
                    shard_indices=(0,),
                    output_root=output,
                    humaneval_dataset=humaneval,
                    workers=1,
                    batch_size=2,
                    max_sequence=2048,
                )

            self.assertEqual(final_database.read_bytes(), b"racing-owner")
            self.assertEqual(stale.read_bytes(), b"unrelated-stale-state")
            self.assertEqual(list(output.glob(".tasks.sqlite.*.tmp")), [])

    def test_rejects_tampered_completed_shard_provenance_on_resume(self) -> None:
        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, humaneval = self._write_tiny_inputs(root)
            output = root / "output"
            build_codex_code_curriculum(
                binding,
                dataset_root=dataset,
                shard_indices=(0,),
                output_root=output,
                humaneval_dataset=humaneval,
                workers=1,
                batch_size=2,
                max_sequence=2048,
            )
            (output / "tasks.sqlite").unlink()
            shard = output / "shards" / "shard_00000.sqlite"
            connection = sqlite3.connect(shard)
            try:
                connection.execute(
                    "UPDATE codex_provenance SET source_parquet_sha256=? "
                    "WHERE example_id=(SELECT example_id FROM codex_provenance LIMIT 1)",
                    ("f" * 64,),
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(ValueError, "provenance"):
                build_codex_code_curriculum(
                    binding,
                    dataset_root=dataset,
                    shard_indices=(0,),
                    output_root=output,
                    humaneval_dataset=humaneval,
                    workers=1,
                    batch_size=2,
                    max_sequence=2048,
                )

            self.assertFalse((output / "tasks.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
