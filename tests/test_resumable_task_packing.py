from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.current_corpus import SourceCandidate
from bosscrafts_model.resumable_current_corpus import ResumableCurrentCorpusAdmission
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
        source_name="fixture/resumable-packing",
        source_revision="fixture-v1",
        repository_identity="fixture/repository",
        upstream_url="https://example.invalid/fixture",
    )


def _write_catalog(path: Path, candidates: tuple[SourceCandidate, ...]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """CREATE TABLE documents (
                   document_id TEXT PRIMARY KEY,
                   canonical_language TEXT NOT NULL,
                   content TEXT NOT NULL,
                   relative_path TEXT NOT NULL,
                   content_sha256 TEXT NOT NULL,
                   license_spdx TEXT NOT NULL,
                   attribution TEXT NOT NULL,
                   ownership TEXT NOT NULL,
                   source_provider TEXT NOT NULL,
                   source_name TEXT NOT NULL,
                   source_revision TEXT NOT NULL,
                   repository_identity TEXT NOT NULL,
                   upstream_url TEXT NOT NULL,
                   dialect TEXT
               )"""
        )
        connection.executemany(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    item.document_id,
                    item.language,
                    item.source,
                    item.relative_path,
                    item.content_sha256,
                    item.license_spdx,
                    item.attribution,
                    item.ownership,
                    item.source_provider,
                    item.source_name,
                    item.source_revision,
                    item.repository_identity,
                    item.upstream_url,
                    item.dialect,
                )
                for item in candidates
            ],
        )
        connection.commit()
    finally:
        connection.close()


class ResumableTaskPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()
        cls.pipeline = BCNLTSourcePipeline(cls.binding)

    def test_commits_each_document_and_resumes_without_duplicate_tasks(self):
        module_name = "bosscrafts_model.resumable_task_packing"
        spec = importlib.util.find_spec(module_name)
        self.assertIsNotNone(spec, "resumable task packing module is required")
        if spec is None:
            return
        module = importlib.import_module(module_name)
        candidates = (
            _candidate("doc-a", "x = 1\ny = x + 2\n"),
            _candidate("doc-b", "def add(a, b):\n    return a + b\n"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.sqlite"
            admission_path = root / "admission.sqlite"
            tasks_path = root / "tasks.sqlite"
            _write_catalog(catalog_path, candidates)
            catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
            admission = ResumableCurrentCorpusAdmission(
                self.pipeline,
                admission_path,
                source_catalog_sha256=catalog_sha256,
                review_manifest_sha256="a" * 64,
                host_code_identity_sha256="b" * 64,
            )
            admitted = admission.process(candidates)
            self.assertTrue(admitted.catalog_exhausted)

            packer = module.ResumableCurrentTaskPacking(
                self.binding,
                admission_path,
                catalog_path,
                tasks_path,
                max_sequence=2048,
            )
            first = packer.process(max_documents=1)
            self.assertEqual(first.processed_this_run, 1)
            self.assertEqual(first.processed_documents, 1)
            self.assertGreater(first.task_count, 0)
            self.assertFalse(first.packing_complete)
            second = packer.process()
            self.assertEqual(second.processed_this_run, 1)
            self.assertEqual(second.processed_documents, 2)
            self.assertGreater(second.task_count, first.task_count)
            self.assertTrue(second.packing_complete)
            resumed = packer.process()
            self.assertEqual(resumed.processed_this_run, 0)
            self.assertEqual(resumed.task_count, second.task_count)
            self.assertEqual(resumed.rejected_count, second.rejected_count)
            self.assertTrue(resumed.packing_complete)
            connection = sqlite3.connect(tasks_path)
            try:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
                runtime_rows = connection.execute(
                    "SELECT DISTINCT parser_runtime_identity_sha256 FROM tasks"
                ).fetchall() if "parser_runtime_identity_sha256" in columns else []
            finally:
                connection.close()
            expected_runtime = self.pipeline.parser_runtime_identity_sha256
            self.assertEqual(metadata.get("parser_runtime_identity_sha256"), expected_runtime)
            self.assertIn("parser_runtime_identity_sha256", columns)
            self.assertEqual(runtime_rows, [(expected_runtime,)])
            dataset_type = getattr(module, "ResumableTaskDataset", None)
            self.assertIsNotNone(
                dataset_type,
                "completed task databases require a verified training reader",
            )
            dataset = dataset_type(self.binding, tasks_path)
            self.assertTrue(hasattr(dataset, "identity_sha256"))
            self.assertEqual(len(dataset.identity_sha256), 64)
            rows = list(dataset.iter_examples())
            self.assertEqual(len(rows), second.task_count)
            self.assertEqual(dataset.example_count(), len(rows))
            self.assertEqual(
                dataset.example_count(split="train"),
                sum(row["split"] == "train" for row in rows),
            )
            self.assertEqual(
                [row["example_id"] for row in rows],
                sorted(row["example_id"] for row in rows),
            )
            for row in rows:
                self.assertIsInstance(row.get("record_index"), int)
                self.assertGreaterEqual(row["record_index"], 0)
                self.assertGreater(row["sampling_weight"], 0.0)
                self.assertEqual(row["parser_runtime_identity_sha256"], expected_runtime)
                self.assertIn(row["split"], {"train", "validation", "test"})
                self.assertGreater(row["loss_start_index"], 0)
                self.assertLess(row["loss_start_index"], len(row["input_ids"]))
                self.assertTrue(
                    all(0 <= token_id < self.binding.vocab_size for token_id in row["input_ids"])
                )
            victim = rows[0]
            hardening = sqlite3.connect(tasks_path)
            try:
                hardening.execute(
                    "UPDATE tasks SET token_count=? WHERE example_id=?",
                    (10**9, victim["example_id"]),
                )
                hardening.commit()
            finally:
                hardening.close()
            with patch.object(
                module,
                "decompress_bounded",
                side_effect=AssertionError("oversized row reached decompression"),
            ):
                with self.assertRaisesRegex(ValueError, "token count"):
                    module.audit_task_database(self.binding, tasks_path)
            hardening = sqlite3.connect(tasks_path)
            try:
                hardening.execute(
                    "UPDATE tasks SET token_count=? WHERE example_id=?",
                    (len(victim["input_ids"]), victim["example_id"]),
                )
                hardening.execute(
                    "UPDATE tasks SET split='untrusted' WHERE example_id=?",
                    (victim["example_id"],),
                )
                hardening.commit()
            finally:
                hardening.close()
            with self.assertRaisesRegex(ValueError, "split"):
                dataset_type(self.binding, tasks_path)
            hardening = sqlite3.connect(tasks_path)
            try:
                hardening.execute(
                    "UPDATE tasks SET split=? WHERE example_id=?",
                    (victim["split"], victim["example_id"]),
                )
                hardening.commit()
            finally:
                hardening.close()
            try:
                remaining = list(
                    dataset.iter_examples(after_example_id=rows[0]["example_id"])
                )
            except TypeError as error:
                self.fail(f"training reader must support resume cursors: {error}")
            self.assertEqual(
                [row["example_id"] for row in remaining],
                [row["example_id"] for row in rows[1:]],
            )
            cells: dict[tuple[str, str, str], list[dict[str, object]]] = {}
            for row in rows:
                cell = (str(row["split"]), str(row["language"]), str(row["family"]))
                cells.setdefault(cell, []).append(row)
            eligible_splits = {
                split
                for split, _, _ in cells
                if sum(cell[0] == split for cell in cells) >= 2
            }
            self.assertTrue(eligible_splits)
            weighted_split = sorted(eligible_splits)[0]
            weighted_cells = sorted(cell for cell in cells if cell[0] == weighted_split)
            target_cell = weighted_cells[0]
            multiplier_key = "|".join(target_cell)
            weighting = sqlite3.connect(tasks_path)
            try:
                weighting.execute(
                    "INSERT INTO metadata(key,value) VALUES(?,?)",
                    (
                        module.CELL_WEIGHT_MULTIPLIERS_METADATA_KEY,
                        json.dumps({multiplier_key: 2.0}, sort_keys=True),
                    ),
                )
                weighting.commit()
            finally:
                weighting.close()
            weighted_dataset = dataset_type(self.binding, tasks_path)
            weighted_rows = list(weighted_dataset.iter_examples(split=weighted_split))
            self.assertAlmostEqual(
                sum(float(row["sampling_weight"]) for row in weighted_rows)
                / len(weighted_rows),
                1.0,
                places=12,
            )
            weighted_totals: dict[tuple[str, str, str], float] = {}
            for row in weighted_rows:
                cell = (
                    str(row["split"]),
                    str(row["language"]),
                    str(row["family"]),
                )
                weighted_totals[cell] = weighted_totals.get(cell, 0.0) + float(
                    row["sampling_weight"]
                )
            comparison_cell = weighted_cells[1]
            self.assertAlmostEqual(
                weighted_totals[target_cell],
                2.0 * weighted_totals[comparison_cell],
                places=10,
            )
            audit_task_database = getattr(module, "audit_task_database", None)
            self.assertIsNotNone(
                audit_task_database,
                "completed task databases require a full identity audit",
            )
            audit = audit_task_database(self.binding, tasks_path)
            self.assertTrue(audit.audit_passed)
            self.assertEqual(audit.example_count, second.task_count)
            self.assertEqual(sum(audit.split_counts.values()), second.task_count)
            self.assertRegex(audit.content_sha256, r"^[0-9a-f]{64}$")
            original_content_sha256 = audit.content_sha256
            original_database_identity = audit.database_identity_sha256
            lineage_tamper = sqlite3.connect(tasks_path)
            try:
                lineage_tamper.execute(
                    "UPDATE tasks SET source_sha256=? WHERE example_id=?",
                    ("f" * 64, rows[0]["example_id"]),
                )
                lineage_tamper.commit()
            finally:
                lineage_tamper.close()
            lineage_audit = audit_task_database(self.binding, tasks_path)
            self.assertNotEqual(lineage_audit.content_sha256, original_content_sha256)
            self.assertNotEqual(
                lineage_audit.database_identity_sha256,
                original_database_identity,
            )
            tamper = sqlite3.connect(tasks_path)
            try:
                tamper.execute(
                    "UPDATE tasks SET example_id=? WHERE example_id=?",
                    ("e" * 64, rows[0]["example_id"]),
                )
                tamper.commit()
            finally:
                tamper.close()
            with self.assertRaisesRegex(ValueError, "example identity"):
                audit_task_database(self.binding, tasks_path)

    def test_resumable_packer_script_runs_from_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "pack_full_resumable_tasks.py"),
                "--help",
            ],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--status-only", completed.stdout)

        audit = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "audit_full_task_database.py"),
                "--help",
            ],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(audit.returncode, 0, audit.stderr)
        self.assertIn("--database", audit.stdout)
        self.assertIn("--output", audit.stdout)
        self.assertIn("--max-documents", completed.stdout)

    def test_compact_wire_is_pinned_in_task_database_lineage(self):
        module = importlib.import_module("bosscrafts_model.resumable_task_packing")
        candidate = _candidate("doc-compact", "x = 1\nprint(x)\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.sqlite"
            admission_path = root / "admission.sqlite"
            tasks_path = root / "tasks.sqlite"
            _write_catalog(catalog_path, (candidate,))
            catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
            admission = ResumableCurrentCorpusAdmission(
                self.pipeline,
                admission_path,
                source_catalog_sha256=catalog_sha256,
                review_manifest_sha256="a" * 64,
                host_code_identity_sha256="b" * 64,
            )
            self.assertTrue(admission.process((candidate,)).catalog_exhausted)

            compact = module.ResumableCurrentTaskPacking(
                self.binding,
                admission_path,
                catalog_path,
                tasks_path,
                max_sequence=2048,
                semantic_wire="compact_v2",
            )
            status = compact.process()
            self.assertTrue(status.packing_complete)
            metadata_connection = sqlite3.connect(tasks_path)
            try:
                metadata = dict(metadata_connection.execute("SELECT key,value FROM metadata"))
            finally:
                metadata_connection.close()
            self.assertEqual(
                metadata.get("semantic_wire_schema_version"),
                "bosscrafts-compact-semantic-wire-v2",
            )
            with self.assertRaisesRegex(ValueError, "lineage mismatch"):
                module.ResumableCurrentTaskPacking(
                    self.binding,
                    admission_path,
                    catalog_path,
                    tasks_path,
                    max_sequence=2048,
                    semantic_wire="full_v3",
                )

    def test_audited_eligibility_filters_documents_and_forces_train_split(self):
        module = importlib.import_module("bosscrafts_model.resumable_task_packing")
        audit_module = importlib.import_module(
            "bosscrafts_model.resumable_training_semantic_audit"
        )
        candidates = (
            _candidate("doc-a", "x = 1\ny = x + 2\n"),
            _candidate("doc-b", "def add(a, b):\n    return a + b\n"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.sqlite"
            admission_path = root / "admission.sqlite"
            tasks_path = root / "tasks.sqlite"
            eligibility_root = root / "eligibility"
            eligibility_root.mkdir()
            _write_catalog(catalog_path, candidates)
            catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
            admission = ResumableCurrentCorpusAdmission(
                self.pipeline,
                admission_path,
                source_catalog_sha256=catalog_sha256,
                review_manifest_sha256="a" * 64,
                host_code_identity_sha256="b" * 64,
            )
            self.assertTrue(admission.process(candidates).catalog_exhausted)
            connection = sqlite3.connect(admission_path)
            try:
                admission_metadata = dict(
                    connection.execute("SELECT key,value FROM metadata")
                )
                admission_identity = audit_module.ResumableTrainingSemanticAudit._admission_identity(
                    connection, admission_metadata
                )
            finally:
                connection.close()
            eligible = {
                "schema_version": "bosscrafts-training-eligible-document-v1",
                "document_id": "doc-a",
            }
            eligible_text = json.dumps(
                eligible, sort_keys=True, separators=(",", ":")
            ) + "\n"
            eligible_sha = hashlib.sha256(eligible_text.encode()).hexdigest()
            manifest = {
                "schema_version": "bosscrafts-training-eligibility-manifest-v1",
                "eligibility_complete": True,
                "training_ready": False,
                "source_catalog_sha256": catalog_sha256,
                "source_split": "train",
                "source_admission_identity_sha256": admission_identity,
                "source_audit_identity_sha256": "c" * 64,
                "eligible_file": "eligible.jsonl",
                "eligible_sha256": eligible_sha,
                "eligible_document_count": 1,
            }
            manifest_sha = hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            manifest["manifest_sha256"] = manifest_sha
            (eligibility_root / "eligible.jsonl").write_bytes(
                eligible_text.encode("utf-8")
            )
            (eligibility_root / "manifest.json").write_bytes(
                (
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
            )

            packer = module.ResumableCurrentTaskPacking(
                self.binding,
                admission_path,
                catalog_path,
                tasks_path,
                max_sequence=2048,
                eligibility_root=eligibility_root,
                expected_eligibility_manifest_sha256=manifest_sha,
            )
            status = packer.process()
            self.assertTrue(status.packing_complete)
            self.assertEqual(status.processed_documents, 1)
            connection = sqlite3.connect(tasks_path)
            try:
                task_documents = {
                    str(row[0])
                    for row in connection.execute("SELECT DISTINCT document_id FROM tasks")
                }
                task_splits = {
                    str(row[0])
                    for row in connection.execute("SELECT DISTINCT split FROM tasks")
                }
                metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            finally:
                connection.close()
            self.assertEqual(task_documents, {"doc-a"})
            self.assertEqual(task_splits, {"train"})
            self.assertEqual(
                metadata["training_eligibility_manifest_sha256"], manifest_sha
            )
            self.assertEqual(
                metadata["audited_source_eligibility_manifest_sha256"], manifest_sha
            )

    def test_explicit_frozen_evaluation_split_overrides_hash_split(self):
        module = importlib.import_module("bosscrafts_model.resumable_task_packing")
        audit_module = importlib.import_module(
            "bosscrafts_model.resumable_training_semantic_audit"
        )
        candidate = _candidate("eval-doc", "def f(x):\n    return x + 1\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.sqlite"
            admission_path = root / "admission.sqlite"
            tasks_path = root / "tasks.sqlite"
            eligibility_root = root / "eligibility"
            eligibility_root.mkdir()
            _write_catalog(catalog_path, (candidate,))
            catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
            admission = ResumableCurrentCorpusAdmission(
                self.pipeline,
                admission_path,
                source_catalog_sha256=catalog_sha256,
                review_manifest_sha256="a" * 64,
                host_code_identity_sha256="b" * 64,
            )
            self.assertTrue(admission.process((candidate,)).catalog_exhausted)
            connection = sqlite3.connect(admission_path)
            try:
                admission_metadata = dict(
                    connection.execute("SELECT key,value FROM metadata")
                )
                admission_identity = (
                    audit_module.ResumableTrainingSemanticAudit._admission_identity(
                        connection, admission_metadata
                    )
                )
            finally:
                connection.close()
            eligible = {
                "schema_version": "bosscrafts-training-eligible-document-v1",
                "document_id": "eval-doc",
            }
            eligible_text = json.dumps(
                eligible, sort_keys=True, separators=(",", ":")
            ) + "\n"
            manifest = {
                "schema_version": "bosscrafts-training-eligibility-manifest-v1",
                "eligibility_complete": True,
                "training_ready": False,
                "source_catalog_sha256": catalog_sha256,
                "source_split": "validation",
                "source_admission_identity_sha256": admission_identity,
                "source_audit_identity_sha256": "c" * 64,
                "eligible_file": "eligible.jsonl",
                "eligible_sha256": hashlib.sha256(
                    eligible_text.encode()
                ).hexdigest(),
                "eligible_document_count": 1,
            }
            manifest_sha = hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            manifest["manifest_sha256"] = manifest_sha
            (eligibility_root / "eligible.jsonl").write_text(
                eligible_text, encoding="utf-8"
            )
            (eligibility_root / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            packer = module.ResumableCurrentTaskPacking(
                self.binding,
                admission_path,
                catalog_path,
                tasks_path,
                max_sequence=2048,
                eligibility_root=eligibility_root,
                expected_eligibility_manifest_sha256=manifest_sha,
                forced_split="validation",
            )
            self.assertTrue(packer.process().packing_complete)
            connection = sqlite3.connect(tasks_path)
            try:
                splits = {
                    str(row[0])
                    for row in connection.execute("SELECT DISTINCT split FROM tasks")
                }
                metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            finally:
                connection.close()
            self.assertEqual(splits, {"validation"})
            self.assertEqual(metadata["forced_task_split"], "validation")
            self.assertEqual(
                metadata["task_split_policy"], "explicit_frozen_source_split"
            )
            self.assertEqual(metadata["eligibility_source_split"], "validation")
            self.assertEqual(
                metadata["audited_source_eligibility_manifest_sha256"],
                manifest_sha,
            )


if __name__ == "__main__":
    unittest.main()
