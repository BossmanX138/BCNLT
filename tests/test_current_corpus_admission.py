from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from bosscrafts_model.bcnlt_source_pipeline import (
    BCNLTSourcePipeline,
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.current_corpus import (
    CurrentCorpusAdmission,
    RECORD_SCHEMA,
    SQLiteSourceCatalog,
    SourceCandidate,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _candidate(
    document_id: str,
    language: str,
    source: str,
    path: str,
    *,
    content_sha256: str | None = None,
) -> SourceCandidate:
    return SourceCandidate(
        document_id=document_id,
        language=language,
        source=source,
        relative_path=path,
        content_sha256=content_sha256 or hashlib.sha256(source.encode("utf-8")).hexdigest(),
        license_spdx="MIT",
        attribution="fixture-author",
        ownership="third_party",
        source_provider="fixture",
        source_name="fixture/current-corpus",
        source_revision="fixture-v1",
        repository_identity="fixture/repository",
        upstream_url="https://example.invalid/fixture",
        dialect=None,
    )


class CurrentCorpusAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()
        cls.pipeline = BCNLTSourcePipeline(cls.binding)

    def test_materializes_deterministic_current_lineage_with_raw_and_semantic_ids(self):
        candidates = (
            _candidate("doc-python", "python", "def add(a, b):\n    return a + b\n", "add.py"),
            _candidate(
                "doc-javascript",
                "javascript",
                "function add(a, b) { return a + b; }\n",
                "add.js",
            ),
        )
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            catalog_sha256 = "a" * 64
            first = CurrentCorpusAdmission(
                self.pipeline, source_catalog_sha256=catalog_sha256
            ).materialize(candidates, first_dir)
            second = CurrentCorpusAdmission(
                self.pipeline, source_catalog_sha256=catalog_sha256
            ).materialize(candidates, second_dir)

            self.assertEqual(first.admitted_count, 2)
            self.assertEqual(first.rejected_count, 0)
            self.assertFalse(first.training_ready)
            self.assertEqual(first.records_sha256, second.records_sha256)
            self.assertEqual(first.manifest_sha256, second.manifest_sha256)
            self.assertEqual(first.tokenizer_sha256, self.binding.tokenizer_sha256)
            self.assertEqual(first.binding_sha256, self.binding.binding_sha256)
            self.assertEqual(first.semantic_spec_sha256, {
                "javascript": CURRENT_RUNTIME_SPEC_SHA256["javascript"],
                "python": CURRENT_RUNTIME_SPEC_SHA256["python"],
            })

            records_path = Path(first_dir) / "admitted.jsonl"
            rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["document_id"] for row in rows], ["doc-javascript", "doc-python"])
            for row in rows:
                self.assertEqual(
                    row["parser_runtime_identity_sha256"],
                    PARSER_RUNTIME_IDENTITY_SHA256,
                )
                self.assertEqual(row["schema_version"], RECORD_SCHEMA)
                self.assertEqual(
                    hashlib.sha256(row["source"].encode("utf-8")).hexdigest(),
                    row["source_sha256"],
                )
                self.assertGreater(row["semantic_record_count"], 0)
                self.assertGreater(len(row["semantic_token_ids"]), 2)
                self.assertGreater(len(row["raw_token_ids"]), 2)
                self.assertTrue(
                    all(0 <= token_id < self.binding.vocab_size for token_id in row["semantic_token_ids"])
                )
                self.assertTrue(
                    all(0 <= token_id < self.binding.vocab_size for token_id in row["raw_token_ids"])
                )
            manifest = json.loads((Path(first_dir) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_catalog_sha256"], catalog_sha256)
            self.assertEqual(
                manifest["parser_runtime_identity_sha256"],
                PARSER_RUNTIME_IDENTITY_SHA256,
            )
            self.assertEqual(
                manifest["token_statistics"]["raw_token_count"],
                sum(len(row["raw_token_ids"]) for row in rows),
            )
            self.assertEqual(
                manifest["token_statistics"]["semantic_token_count"],
                sum(len(row["semantic_token_ids"]) for row in rows),
            )
            self.assertEqual(
                manifest["token_statistics"]["semantic_documents_over_2048"],
                sum(len(row["semantic_token_ids"]) > 2048 for row in rows),
            )

    def test_rejections_are_logged_and_never_silently_fallback_to_raw(self):
        good = _candidate("same-id", "python", "value = 1\n", "good.py")
        candidates = (
            good,
            _candidate("same-id", "python", "value = 2\n", "duplicate.py"),
            _candidate("bad-hash", "python", "value = 3\n", "hash.py", content_sha256="0" * 64),
            _candidate("bad-language", "brainfuck", "+", "x.bf"),
            _candidate("bad-parse", "javascript", "function {", "broken.js"),
        )
        with tempfile.TemporaryDirectory() as output_dir:
            result = CurrentCorpusAdmission(self.pipeline).materialize(candidates, output_dir)
            self.assertEqual(result.admitted_count, 1)
            self.assertEqual(result.rejected_count, 4)
            rows = [
                json.loads(line)
                for line in (Path(output_dir) / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {row["reason_code"] for row in rows},
                {"duplicate_document_id", "content_hash_mismatch", "unsupported_language", "semantic_parse_failure"},
            )
            self.assertTrue(all("raw_token_ids" not in row for row in rows))
            parse_rejection = next(
                row for row in rows if row["reason_code"] == "semantic_parse_failure"
            )
            self.assertIn("BCNLT parser", parse_rejection["detail"])

    def test_invalid_unicode_source_is_a_logged_rejection_not_a_crash(self):
        malformed = replace(
            _candidate("bad-unicode", "python", "value = 1\n", "bad.py"),
            source="value = '\ud800'\n",
        )
        with tempfile.TemporaryDirectory() as output_dir:
            result = CurrentCorpusAdmission(self.pipeline).materialize(
                (malformed,),
                output_dir,
            )
            self.assertEqual(result.admitted_count, 0)
            self.assertEqual(result.rejected_count, 1)
            rejection = json.loads(
                (Path(output_dir) / "rejected.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(rejection["reason_code"], "invalid_source_unicode")

    def test_sqlite_catalog_is_read_only_and_selects_each_language_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "corpus.sqlite"
            connection = sqlite3.connect(database)
            connection.execute(
                """CREATE TABLE documents (
                    document_id TEXT PRIMARY KEY, source_provider TEXT, source_name TEXT,
                    source_revision TEXT, canonical_language TEXT, repository_identity TEXT,
                    relative_path TEXT, content_sha256 TEXT, content TEXT, license_spdx TEXT,
                    ownership TEXT, attribution TEXT, upstream_url TEXT, dialect TEXT,
                    registry_entry_sha256 TEXT, parser_bundle_sha256 TEXT,
                    grammar_revision TEXT, semantic_spec_sha256 TEXT,
                    extraction_rule_version TEXT
                )"""
            )
            for document_id, language, source in (
                ("p2", "python", "p2 = 2\n"),
                ("j1", "javascript", "let j = 1;\n"),
                ("p1", "python", "p1 = 1\n"),
            ):
                connection.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        document_id, "fixture", "fixture/source", "rev", language,
                        "fixture/repo", f"{document_id}.txt",
                        hashlib.sha256(source.encode()).hexdigest(), source, "MIT",
                        "third_party", "fixture-author", "https://example.invalid", None,
                        "r", "p", "g", "s", "e",
                    ),
                )
            connection.commit()
            connection.close()

            selected = tuple(SQLiteSourceCatalog(database).iter_candidates(limit_per_language=1))
            self.assertEqual(
                [(item.language, item.document_id) for item in selected],
                [("javascript", "j1"), ("python", "p1")],
            )
            verification = sqlite3.connect(database)
            try:
                self.assertEqual(
                    verification.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                    3,
                )
            finally:
                verification.close()
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                SQLiteSourceCatalog(database, expected_sha256="0" * 64)

    @unittest.skipUnless(sys.platform == "win32", "Windows UNC URI behavior")
    def test_sqlite_catalog_builds_read_only_uri_for_unc_paths(self):
        uri = SQLiteSourceCatalog._read_only_uri(
            Path(r"\\server\share\catalog.sqlite")
        )
        self.assertTrue(uri.startswith("file:////server/share/"), uri)
        self.assertTrue(uri.endswith("catalog.sqlite?mode=ro"), uri)

    def test_builder_script_runs_directly_from_the_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "build_current_65k_corpus.py"), "--help"],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--limit-per-language", completed.stdout)


if __name__ == "__main__":
    unittest.main()
