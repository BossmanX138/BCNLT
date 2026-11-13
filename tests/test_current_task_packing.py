from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bosscrafts_model.bcnlt_source_pipeline import (
    BCNLTSourcePipeline,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.current_corpus import CurrentCorpusAdmission, SourceCandidate
from bosscrafts_model.current_task_packing import CurrentCorpusTaskPacker
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _candidate(document_id: str, source: str) -> SourceCandidate:
    return SourceCandidate(
        document_id=document_id,
        language="python",
        source=source,
        relative_path=f"{document_id}.py",
        content_sha256=hashlib.sha256(source.encode()).hexdigest(),
        license_spdx="MIT",
        attribution="fixture-author",
        ownership="third_party",
        source_provider="fixture",
        source_name="fixture/task-packing",
        source_revision="fixture-v1",
        repository_identity="fixture/repository",
        upstream_url="https://example.invalid/fixture",
    )


class CurrentCorpusTaskPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()
        cls.pipeline = BCNLTSourcePipeline(cls.binding)

    def test_packs_three_task_families_with_bounded_loss_masked_sequences(self):
        candidates = (
            _candidate("doc-a", "x = 1\ny = x + 2\n"),
            _candidate("doc-b", "def add(a, b):\n    return a + b\n"),
        )
        with tempfile.TemporaryDirectory() as admission_dir, tempfile.TemporaryDirectory() as output_dir:
            CurrentCorpusAdmission(self.pipeline).materialize(candidates, admission_dir)
            result = CurrentCorpusTaskPacker(self.binding, max_sequence=2048).materialize(
                admission_dir,
                output_dir,
            )
            self.assertGreater(result.example_count, 0)
            self.assertEqual(
                set(result.family_counts),
                {"code_to_semantic", "semantic_continuation", "semantic_to_code"},
            )
            rows = [
                json.loads(line)
                for line in (Path(output_dir) / "examples.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            task_end = self.binding.structural_token_ids["<|bcnlt:task:end|>"]
            family_tokens = {
                name: self.binding.structural_token_ids[f"<|bcnlt:task:{name}|>"]
                for name in result.family_counts
            }
            for row in rows:
                self.assertEqual(
                    row["parser_runtime_identity_sha256"],
                    PARSER_RUNTIME_IDENTITY_SHA256,
                )
                self.assertLessEqual(len(row["input_ids"]), 2048)
                self.assertGreater(row["loss_start_index"], 0)
                self.assertLess(row["loss_start_index"], len(row["input_ids"]))
                self.assertEqual(row["input_ids"][0], family_tokens[row["family"]])
                self.assertEqual(row["input_ids"][-1], task_end)
                self.assertTrue(all(0 <= token_id < self.binding.vocab_size for token_id in row["input_ids"]))
                self.assertEqual(
                    row["target_token_count"],
                    len(row["input_ids"]) - row["loss_start_index"],
                )
            code_to_semantic = [
                row for row in rows if row["family"] == "code_to_semantic"
            ]
            prompts = [
                tuple(row["input_ids"][: row["loss_start_index"]])
                for row in code_to_semantic
            ]
            self.assertEqual(len(prompts), len(set(prompts)))
            record_begin = self.binding.structural_token_ids[
                "<|bcnlt:record:begin|>"
            ]
            self.assertTrue(
                any(
                    row["input_ids"][row["loss_start_index"] :].count(record_begin)
                    > 1
                    for row in code_to_semantic
                )
            )
            splits_by_document: dict[str, set[str]] = {}
            for row in rows:
                splits_by_document.setdefault(row["document_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in splits_by_document.values()))

    def test_packing_is_deterministic_and_rejects_tampered_admission(self):
        candidates = (_candidate("doc-a", "value = 1\n"),)
        with (
            tempfile.TemporaryDirectory() as admission_dir,
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            CurrentCorpusAdmission(self.pipeline).materialize(candidates, admission_dir)
            first = CurrentCorpusTaskPacker(self.binding).materialize(admission_dir, first_dir)
            second = CurrentCorpusTaskPacker(self.binding).materialize(admission_dir, second_dir)
            self.assertEqual(first.examples_sha256, second.examples_sha256)
            self.assertEqual(first.manifest_sha256, second.manifest_sha256)

            records = Path(admission_dir) / "admitted.jsonl"
            records.write_text(records.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "admission records SHA-256"):
                CurrentCorpusTaskPacker(self.binding).materialize(admission_dir, first_dir)

    def test_packer_script_runs_directly_from_the_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "pack_current_65k_tasks.py"), "--help"],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--max-sequence", completed.stdout)


if __name__ == "__main__":
    unittest.main()
