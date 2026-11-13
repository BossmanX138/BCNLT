from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bosscrafts_model.balanced_curriculum import build_balanced_curriculum
from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.compact_semantic_wire import COMPACT_SEMANTIC_WIRE_SCHEMA
from bosscrafts_model.current_corpus import SourceCandidate
from bosscrafts_model.current_task_packing import CurrentCorpusTaskPacker
from bosscrafts_model.resumable_current_corpus import ResumableCurrentCorpusAdmission
from bosscrafts_model.resumable_task_packing import audit_task_database
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _document_id_for_split(split: str) -> str:
    for index in range(10_000):
        document_id = hashlib.sha256(f"curriculum-{split}-{index}".encode()).hexdigest()
        if CurrentCorpusTaskPacker._split(document_id) == split:
            return document_id
    raise AssertionError(f"could not find a fixture ID for {split}")


def _candidate(split: str) -> SourceCandidate:
    document_id = _document_id_for_split(split)
    suffix = document_id[:8]
    source = (
        f"first_{suffix} = 1\n"
        f"second_{suffix} = 2\n"
        f"total_{suffix} = first_{suffix} + second_{suffix}\n"
    )
    return SourceCandidate(
        document_id=document_id,
        language="python",
        source=source,
        relative_path=f"{split}.py",
        content_sha256=hashlib.sha256(source.encode()).hexdigest(),
        license_spdx="MIT",
        attribution="fixture-author",
        ownership="third_party",
        source_provider="fixture",
        source_name="fixture/balanced-curriculum",
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


class BalancedCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()
        cls.pipeline = BCNLTSourcePipeline(cls.binding)

    def test_builds_exact_deterministic_split_language_family_quotas(self):
        candidates = tuple(_candidate(split) for split in ("train", "validation", "test"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.sqlite"
            admission = root / "admission.sqlite"
            first_database = root / "first.sqlite"
            second_database = root / "second.sqlite"
            _write_catalog(catalog, candidates)
            admission_writer = ResumableCurrentCorpusAdmission(
                self.pipeline,
                admission,
                source_catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
                review_manifest_sha256="a" * 64,
                host_code_identity_sha256="b" * 64,
            )
            status = admission_writer.process(candidates)
            self.assertTrue(status.catalog_exhausted)

            first = build_balanced_curriculum(
                self.binding,
                admission,
                catalog,
                first_database,
                languages=("python",),
                quotas={"train": 1, "validation": 1, "test": 1},
            )
            second = build_balanced_curriculum(
                self.binding,
                admission,
                catalog,
                second_database,
                languages=("python",),
                quotas={"train": 1, "validation": 1, "test": 1},
            )
            self.assertEqual(first.example_count, 9)
            self.assertEqual(first.cell_counts, second.cell_counts)
            self.assertEqual(
                first.database_identity_sha256,
                second.database_identity_sha256,
            )
            self.assertEqual(
                first.cell_counts,
                {
                    "test|python|code_to_semantic": 1,
                    "test|python|semantic_continuation": 1,
                    "test|python|semantic_to_code": 1,
                    "train|python|code_to_semantic": 1,
                    "train|python|semantic_continuation": 1,
                    "train|python|semantic_to_code": 1,
                    "validation|python|code_to_semantic": 1,
                    "validation|python|semantic_continuation": 1,
                    "validation|python|semantic_to_code": 1,
                },
            )
            audit = audit_task_database(self.binding, first_database)
            self.assertTrue(audit.audit_passed)
            self.assertEqual(audit.example_count, 9)
            self.assertEqual(audit.split_counts, {"test": 3, "train": 3, "validation": 3})
            connection = sqlite3.connect(first_database)
            try:
                metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            finally:
                connection.close()
            self.assertEqual(
                metadata["semantic_wire_schema_version"],
                COMPACT_SEMANTIC_WIRE_SCHEMA,
            )

    def test_rejects_unknown_semantic_wire(self):
        with self.assertRaisesRegex(ValueError, "semantic_wire"):
            build_balanced_curriculum(
                self.binding,
                "missing-admission.sqlite",
                "missing-catalog.sqlite",
                "missing-output.sqlite",
                languages=("python",),
                quotas={"train": 1, "validation": 1, "test": 1},
                semantic_wire="unknown",
            )

    def test_refuses_incomplete_admission(self):
        candidates = tuple(_candidate(split) for split in ("train", "validation", "test"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.sqlite"
            admission = root / "admission.sqlite"
            _write_catalog(catalog, candidates)
            admission_writer = ResumableCurrentCorpusAdmission(
                self.pipeline,
                admission,
                source_catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
                review_manifest_sha256="a" * 64,
                host_code_identity_sha256="b" * 64,
            )
            status = admission_writer.process(candidates, max_documents=1)
            self.assertFalse(status.catalog_exhausted)
            with self.assertRaisesRegex(ValueError, "full admission is not complete"):
                build_balanced_curriculum(
                    self.binding,
                    admission,
                    catalog,
                    root / "tasks.sqlite",
                    languages=("python",),
                    quotas={"train": 1, "validation": 1, "test": 1},
                )

    def test_builder_script_runs_from_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "build_balanced_curriculum.py"),
                "--help",
            ],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--train-per-cell", completed.stdout)
        self.assertIn("--validation-per-cell", completed.stdout)
        self.assertIn("--test-per-cell", completed.stdout)


if __name__ == "__main__":
    unittest.main()
