from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bosscrafts_model.bcnlt_source_pipeline import (
    PARSER_RUNTIME_IDENTITY_SHA256,
    BCNLTSourcePipeline,
)
from bosscrafts_model.current_corpus import CurrentCorpusAdmission, SourceCandidate
from bosscrafts_model.current_task_packing import CurrentCorpusTaskPacker
from bosscrafts_model.heldout_semantic_review import (
    HeldoutSemanticReviewAdjudicator,
    HeldoutSemanticReviewQueue,
)
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
        source_name="fixture/heldout-review",
        source_revision="fixture-v1",
        repository_identity="fixture/repository",
        upstream_url="https://example.invalid/fixture",
    )


def _heldout_document_id() -> str:
    for index in range(10_000):
        document_id = f"heldout-{index}"
        if CurrentCorpusTaskPacker._split(document_id) in {"validation", "test"}:
            return document_id
    raise AssertionError("failed to find deterministic held-out fixture id")


class HeldoutSemanticReviewQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()
        cls.pipeline = BCNLTSourcePipeline(cls.binding)

    def test_materializes_deterministic_source_to_semantic_review_items(self):
        candidate = _candidate(_heldout_document_id(), "value = 1\nprint(value)\n")
        with (
            tempfile.TemporaryDirectory() as admission_dir,
            tempfile.TemporaryDirectory() as tasks_dir,
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            CurrentCorpusAdmission(self.pipeline).materialize((candidate,), admission_dir)
            CurrentCorpusTaskPacker(self.binding).materialize(admission_dir, tasks_dir)

            first = HeldoutSemanticReviewQueue(self.binding, per_language=1).materialize(
                admission_dir, tasks_dir, first_dir
            )
            second = HeldoutSemanticReviewQueue(self.binding, per_language=1).materialize(
                admission_dir, tasks_dir, second_dir
            )

            self.assertEqual(first.queue_sha256, second.queue_sha256)
            self.assertEqual(first.item_count, 1)
            item = json.loads(
                (Path(first_dir) / "review_queue.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(item["language"], "python")
            self.assertIn(item["split"], {"validation", "test"})
            self.assertEqual(item["family"], "code_to_semantic")
            self.assertTrue(item["source_excerpt"].strip())
            self.assertIsInstance(item["canonical_record"], dict)
            self.assertEqual(item["review_status"], "unreviewed")
            self.assertEqual(
                item["parser_runtime_identity_sha256"],
                PARSER_RUNTIME_IDENTITY_SHA256,
            )
            self.assertEqual(
                item["rubric"],
                {
                    "event_matches_source": None,
                    "ir_kind_plausible": None,
                    "scope_plausible": None,
                    "span_matches_source": None,
                    "uncertainty_calibrated": None,
                },
            )
            manifest = json.loads(
                (Path(first_dir) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["review_complete"])
            self.assertEqual(manifest["represented_languages"], ["python"])
            self.assertEqual(manifest["missing_languages"], [])
            self.assertEqual(
                manifest["parser_runtime_identity_sha256"],
                PARSER_RUNTIME_IDENTITY_SHA256,
            )

    def test_rejects_tampered_packed_examples_before_selecting_review_items(self):
        candidate = _candidate(_heldout_document_id(), "value = 1\n")
        with (
            tempfile.TemporaryDirectory() as admission_dir,
            tempfile.TemporaryDirectory() as tasks_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            CurrentCorpusAdmission(self.pipeline).materialize((candidate,), admission_dir)
            CurrentCorpusTaskPacker(self.binding).materialize(admission_dir, tasks_dir)
            examples_path = Path(tasks_dir) / "examples.jsonl"
            examples_path.write_text(
                examples_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "packed examples SHA-256"):
                HeldoutSemanticReviewQueue(self.binding).materialize(
                    admission_dir, tasks_dir, output_dir
                )

    def test_materializes_review_items_directly_from_heldout_admission(self):
        candidate = _candidate(
            _heldout_document_id(),
            "before = 0\nvalue = 1\nprint(value)\nafter = 2\n",
        )
        with tempfile.TemporaryDirectory() as admission_dir, tempfile.TemporaryDirectory() as output_dir:
            CurrentCorpusAdmission(self.pipeline).materialize((candidate,), admission_dir)
            result = HeldoutSemanticReviewQueue(
                self.binding, per_language=1
            ).materialize_admission_only(admission_dir, output_dir)
            self.assertEqual(result.item_count, 1)
            item = json.loads(
                (Path(output_dir) / "review_queue.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(item["split"], CurrentCorpusTaskPacker._split(candidate.document_id))
            self.assertEqual(item["language"], "python")
            self.assertIsInstance(item["canonical_record"], dict)
            self.assertIn("[source context lines ", item["source_excerpt"])
            self.assertIn("target span", item["source_excerpt"])
            numbered_source_lines = [
                line for line in item["source_excerpt"].splitlines() if "|" in line
            ]
            self.assertGreaterEqual(len(numbered_source_lines), 3)

    def test_review_builder_script_runs_from_the_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "build_heldout_semantic_review.py"), "--help"],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--per-language", completed.stdout)

    def test_adjudication_requires_one_complete_decision_per_review_item(self):
        candidate = _candidate(_heldout_document_id(), "value = 1\n")
        with (
            tempfile.TemporaryDirectory() as admission_dir,
            tempfile.TemporaryDirectory() as queue_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            CurrentCorpusAdmission(self.pipeline).materialize((candidate,), admission_dir)
            HeldoutSemanticReviewQueue(self.binding, per_language=1).materialize_admission_only(
                admission_dir, queue_dir
            )
            item = json.loads(
                (Path(queue_dir) / "review_queue.jsonl").read_text(encoding="utf-8").strip()
            )
            decision = {
                "review_id": item["review_id"],
                "decision": "keep",
                "reviewer": "fixture-reviewer",
                "notes": "Source and semantic record align.",
                "rubric": {
                    "event_matches_source": True,
                    "ir_kind_plausible": True,
                    "scope_plausible": True,
                    "span_matches_source": True,
                    "uncertainty_calibrated": True,
                },
            }
            result = HeldoutSemanticReviewAdjudicator().materialize(
                queue_dir, (decision,), output_dir
            )
            self.assertTrue(result.review_complete)
            self.assertTrue(result.semantic_review_passed)
            self.assertEqual(result.decision_counts, {"keep": 1, "reject": 0, "uncertain": 0})
            manifest = json.loads((Path(output_dir) / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["training_ready"])
            self.assertEqual(
                manifest["parser_runtime_identity_sha256"],
                PARSER_RUNTIME_IDENTITY_SHA256,
            )

            with self.assertRaisesRegex(ValueError, "missing review decisions"):
                HeldoutSemanticReviewAdjudicator().materialize(queue_dir, (), output_dir)

    def test_queue_and_adjudication_generations_are_never_overwritten(self):
        candidate = _candidate(_heldout_document_id(), "value = 1\n")
        with (
            tempfile.TemporaryDirectory() as admission_dir,
            tempfile.TemporaryDirectory() as queue_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            CurrentCorpusAdmission(self.pipeline).materialize(
                (candidate,), admission_dir
            )
            queue_builder = HeldoutSemanticReviewQueue(
                self.binding, per_language=1
            )
            queue_builder.materialize_admission_only(admission_dir, queue_dir)
            queue_root = Path(queue_dir)
            original_queue = {
                path.name: path.read_bytes()
                for path in queue_root.iterdir()
                if path.is_file()
            }
            with self.assertRaisesRegex(ValueError, "already contains"):
                queue_builder.materialize_admission_only(admission_dir, queue_dir)
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in queue_root.iterdir()
                    if path.is_file()
                },
                original_queue,
            )

            item = json.loads(
                (queue_root / "review_queue.jsonl").read_text(encoding="utf-8")
            )
            decision = {
                "review_id": item["review_id"],
                "decision": "keep",
                "reviewer": "fixture-reviewer",
                "notes": "Source and semantic record align.",
                "rubric": {
                    "event_matches_source": True,
                    "ir_kind_plausible": True,
                    "scope_plausible": True,
                    "span_matches_source": True,
                    "uncertainty_calibrated": True,
                },
            }
            adjudicator = HeldoutSemanticReviewAdjudicator()
            adjudicator.materialize(queue_dir, (decision,), output_dir)
            original_adjudication = {
                path.name: path.read_bytes()
                for path in Path(output_dir).iterdir()
                if path.is_file()
            }
            with self.assertRaisesRegex(ValueError, "already contains"):
                adjudicator.materialize(queue_dir, (decision,), output_dir)
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in Path(output_dir).iterdir()
                    if path.is_file()
                },
                original_adjudication,
            )

    def test_adjudicator_rejects_queue_file_path_escape(self):
        candidate = _candidate(_heldout_document_id(), "value = 1\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admission = root / "admission"
            queue = root / "queue"
            output = root / "output"
            CurrentCorpusAdmission(self.pipeline).materialize(
                (candidate,), admission
            )
            HeldoutSemanticReviewQueue(
                self.binding, per_language=1
            ).materialize_admission_only(admission, queue)
            queue_text = (queue / "review_queue.jsonl").read_text(
                encoding="utf-8"
            )
            (root / "outside.jsonl").write_text(
                queue_text, encoding="utf-8", newline="\n"
            )
            manifest_path = queue / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("manifest_sha256")
            manifest["queue_file"] = "../outside.jsonl"
            canonical_manifest = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            manifest["manifest_sha256"] = hashlib.sha256(
                canonical_manifest.encode("utf-8")
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            item = json.loads(queue_text)
            decision = {
                "review_id": item["review_id"],
                "decision": "keep",
                "reviewer": "fixture-reviewer",
                "notes": "Source and semantic record align.",
                "rubric": {
                    "event_matches_source": True,
                    "ir_kind_plausible": True,
                    "scope_plausible": True,
                    "span_matches_source": True,
                    "uncertainty_calibrated": True,
                },
            }

            with self.assertRaisesRegex(ValueError, "file inventory"):
                HeldoutSemanticReviewAdjudicator().materialize(
                    queue, (decision,), output
                )


if __name__ == "__main__":
    unittest.main()
