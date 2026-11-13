from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

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
        source_name="fixture/resumable-admission",
        source_revision="fixture-v1",
        repository_identity="fixture/repository",
        upstream_url="https://example.invalid/fixture",
    )


class ResumableCurrentCorpusAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()
        cls.pipeline = BCNLTSourcePipeline(cls.binding)

    def test_commits_each_document_and_resumes_without_reprocessing(self):
        candidates = (
            _candidate("resume-a", "x = 1\n"),
            _candidate("resume-b", "def add(a, b):\n    return a + b\n"),
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "admission.sqlite"
            admission = ResumableCurrentCorpusAdmission(
                self.pipeline,
                database,
                source_catalog_sha256="a" * 64,
                review_manifest_sha256="b" * 64,
                host_code_identity_sha256="c" * 64,
            )
            first = admission.process(candidates, max_documents=1)
            self.assertEqual(first.processed_this_run, 1)
            self.assertEqual(first.processed_total, 1)
            self.assertFalse(first.catalog_exhausted)

            second = admission.process(candidates)
            self.assertEqual(second.processed_this_run, 1)
            self.assertEqual(second.processed_total, 2)
            self.assertEqual(second.admitted_count, 2)
            self.assertEqual(second.rejected_count, 0)
            self.assertTrue(second.catalog_exhausted)

            connection = sqlite3.connect(database)
            try:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(documents)")
                }
                rows = connection.execute(
                    "SELECT document_id, status, semantic_token_count, length(semantic_ids_zlib) "
                    "FROM documents ORDER BY document_id"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual([row[:2] for row in rows], [("resume-a", "admitted"), ("resume-b", "admitted")])
            self.assertTrue(all(row[2] > 0 and row[3] > 0 for row in rows))
            expected_runtime = self.pipeline.parser_runtime_identity_sha256
            self.assertEqual(metadata.get("parser_runtime_identity_sha256"), expected_runtime)
            self.assertEqual(metadata.get("sidecar_timeout_seconds"), "30")
            self.assertEqual(metadata.get("host_code_identity_sha256"), "c" * 64)
            self.assertIn("parser_runtime_identity_sha256", columns)
            connection = sqlite3.connect(database)
            try:
                runtime_rows = connection.execute(
                    "SELECT DISTINCT parser_runtime_identity_sha256 FROM documents "
                    "WHERE status='admitted'"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(runtime_rows, [(expected_runtime,)])

    def test_resume_fails_closed_if_frozen_lineage_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "admission.sqlite"
            ResumableCurrentCorpusAdmission(
                self.pipeline,
                database,
                source_catalog_sha256="a" * 64,
                review_manifest_sha256="b" * 64,
                host_code_identity_sha256="c" * 64,
            ).process((_candidate("resume-a", "x = 1\n"),))
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'tokenizer_sha256'",
                    ("0" * 64,),
                )
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(ValueError, "resumable admission lineage mismatch"):
                ResumableCurrentCorpusAdmission(
                    self.pipeline,
                    database,
                    source_catalog_sha256="a" * 64,
                    review_manifest_sha256="b" * 64,
                    host_code_identity_sha256="c" * 64,
                )

            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'tokenizer_sha256'",
                    (self.binding.tokenizer_sha256,),
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(ValueError, "resumable admission lineage mismatch"):
                ResumableCurrentCorpusAdmission(
                    self.pipeline,
                    database,
                    source_catalog_sha256="a" * 64,
                    review_manifest_sha256="b" * 64,
                    host_code_identity_sha256="d" * 64,
                )

            different_timeout = BCNLTSourcePipeline(
                self.binding,
                timeout_seconds=20,
            )
            with self.assertRaisesRegex(ValueError, "resumable admission lineage mismatch"):
                ResumableCurrentCorpusAdmission(
                    different_timeout,
                    database,
                    source_catalog_sha256="a" * 64,
                    review_manifest_sha256="b" * 64,
                    host_code_identity_sha256="c" * 64,
                )

    def test_optional_source_freeze_lineage_is_persisted_and_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "admission.sqlite"
            ResumableCurrentCorpusAdmission(
                self.pipeline,
                database,
                source_catalog_sha256="a" * 64,
                review_manifest_sha256="b" * 64,
                host_code_identity_sha256="c" * 64,
                source_freeze_manifest_sha256="d" * 64,
                semantic_audit_qualification_sha256="e" * 64,
                expected_source_document_count=7,
            )
            connection = sqlite3.connect(database)
            try:
                metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            finally:
                connection.close()
            self.assertEqual(metadata["source_freeze_manifest_sha256"], "d" * 64)
            self.assertEqual(
                metadata["semantic_audit_qualification_sha256"], "e" * 64
            )
            self.assertEqual(metadata["expected_source_document_count"], "7")

            with self.assertRaisesRegex(ValueError, "resumable admission lineage mismatch"):
                ResumableCurrentCorpusAdmission(
                    self.pipeline,
                    database,
                    source_catalog_sha256="a" * 64,
                    review_manifest_sha256="b" * 64,
                    host_code_identity_sha256="c" * 64,
                    source_freeze_manifest_sha256="f" * 64,
                    semantic_audit_qualification_sha256="e" * 64,
                    expected_source_document_count=7,
                )

            with self.assertRaisesRegex(ValueError, "expected_source_document_count"):
                ResumableCurrentCorpusAdmission(
                    self.pipeline,
                    Path(directory) / "invalid.sqlite",
                    source_catalog_sha256="a" * 64,
                    review_manifest_sha256="b" * 64,
                    host_code_identity_sha256="c" * 64,
                    source_freeze_manifest_sha256="d" * 64,
                    semantic_audit_qualification_sha256="e" * 64,
                    expected_source_document_count=0,
                )

    def test_parallel_workers_preserve_serial_rows_and_duplicate_order(self):
        candidates = (
            _candidate("parallel-a", "x = 1\n"),
            _candidate("parallel-b", "def add(a, b):\n    return a + b\n"),
            _candidate("parallel-duplicate", "x = 1\n"),
            _candidate("parallel-c", "for value in range(3):\n    print(value)\n"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            databases = {
                "serial": root / "serial.sqlite",
                "parallel": root / "parallel.sqlite",
                "processes": root / "processes.sqlite",
            }
            statuses = {}
            for name, database in databases.items():
                admission = ResumableCurrentCorpusAdmission(
                    self.pipeline,
                    database,
                    source_catalog_sha256="a" * 64,
                    review_manifest_sha256="b" * 64,
                    host_code_identity_sha256="c" * 64,
                )
                statuses[name] = admission.process(
                    candidates,
                    workers=1 if name == "serial" else 4,
                    worker_mode="processes" if name == "processes" else "threads",
                )
            self.assertEqual(statuses["serial"].processed_total, 4)
            self.assertEqual(statuses["parallel"].processed_total, 4)
            self.assertEqual(statuses["parallel"].admitted_count, 3)
            self.assertEqual(statuses["parallel"].rejected_count, 1)
            self.assertEqual(statuses["processes"], statuses["parallel"].__class__(
                database_path=databases["processes"].resolve(),
                processed_this_run=statuses["parallel"].processed_this_run,
                processed_total=statuses["parallel"].processed_total,
                admitted_count=statuses["parallel"].admitted_count,
                rejected_count=statuses["parallel"].rejected_count,
                semantic_token_count=statuses["parallel"].semantic_token_count,
                raw_token_count=statuses["parallel"].raw_token_count,
                catalog_exhausted=statuses["parallel"].catalog_exhausted,
            ))
            rows = {}
            for name, database in databases.items():
                connection = sqlite3.connect(database)
                try:
                    rows[name] = connection.execute(
                        "SELECT * FROM documents ORDER BY document_id"
                    ).fetchall()
                finally:
                    connection.close()
            self.assertEqual(rows["parallel"], rows["serial"])
            self.assertEqual(rows["processes"], rows["serial"])
            duplicate = next(
                row for row in rows["parallel"] if row[0] == "parallel-duplicate"
            )
            self.assertEqual(duplicate[4:6], ("rejected", "duplicate_content_sha256"))

            with self.assertRaisesRegex(ValueError, "workers"):
                ResumableCurrentCorpusAdmission(
                    self.pipeline,
                    root / "invalid-workers.sqlite",
                    source_catalog_sha256="a" * 64,
                    review_manifest_sha256="b" * 64,
                    host_code_identity_sha256="c" * 64,
                ).process(candidates, workers=0)
            with self.assertRaisesRegex(ValueError, "worker_mode"):
                ResumableCurrentCorpusAdmission(
                    self.pipeline,
                    root / "invalid-worker-mode.sqlite",
                    source_catalog_sha256="a" * 64,
                    review_manifest_sha256="b" * 64,
                    host_code_identity_sha256="c" * 64,
                ).process(candidates, worker_mode="invalid")

    def test_rejected_content_does_not_suppress_later_valid_provenance(self):
        source = "approved = True\n"
        rejected = replace(
            _candidate("rejected-license", source),
            license_spdx="LicenseRef-Unapproved",
        )
        admitted = _candidate("approved-license", source)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "admission.sqlite"
            status = ResumableCurrentCorpusAdmission(
                self.pipeline,
                database,
                source_catalog_sha256="a" * 64,
                review_manifest_sha256="b" * 64,
                host_code_identity_sha256="c" * 64,
            ).process((rejected, admitted), workers=2)
            self.assertEqual(status.admitted_count, 1)
            self.assertEqual(status.rejected_count, 1)
            connection = sqlite3.connect(database)
            try:
                rows = dict(
                    connection.execute(
                        "SELECT document_id, status FROM documents ORDER BY document_id"
                    )
                )
            finally:
                connection.close()
            self.assertEqual(rows["rejected-license"], "rejected")
            self.assertEqual(rows["approved-license"], "admitted")

    def test_full_admission_script_runs_from_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "run_full_resumable_admission.py"), "--help"],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--max-documents", completed.stdout)
        self.assertIn("--workers", completed.stdout)

    def test_v10_frozen_admission_script_runs_from_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "run_v10_frozen_training_admission.py"),
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
        self.assertIn("--max-documents", completed.stdout)
        self.assertIn("--workers", completed.stdout)


if __name__ == "__main__":
    unittest.main()
