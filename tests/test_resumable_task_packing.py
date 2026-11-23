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
            contention_probe = packer._connect()
            try:
                busy_timeout_ms = int(
                    contention_probe.execute("PRAGMA busy_timeout").fetchone()[0]
                )
            finally:
                contention_probe.close()
            self.assertGreaterEqual(busy_timeout_ms, 300_000)
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
            self.assertEqual(
                set(weighted_dataset.cell_sampling_weight_totals(split=weighted_split)),
                set(weighted_cells),
            )
            for cell, total in weighted_dataset.cell_sampling_weight_totals(
                split=weighted_split
            ).items():
                self.assertAlmostEqual(total, weighted_totals[cell], places=10)
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
            ledger_tamper = sqlite3.connect(tasks_path)
            try:
                ledger_tamper.execute(
                    "UPDATE documents SET task_count=task_count+1 WHERE document_id=?",
                    (rows[0]["document_id"],),
                )
                ledger_tamper.commit()
            finally:
                ledger_tamper.close()
            with self.assertRaisesRegex(ValueError, "ledger count mismatch"):
                audit_task_database(self.binding, tasks_path)
            ledger_restore = sqlite3.connect(tasks_path)
            try:
                ledger_restore.execute(
                    "UPDATE documents SET task_count=task_count-1 WHERE document_id=?",
                    (rows[0]["document_id"],),
                )
                ledger_restore.commit()
            finally:
                ledger_restore.close()
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

    def test_multiworker_document_write_retries_database_lock_atomically(self):
        module = importlib.import_module("bosscrafts_model.resumable_task_packing")

        class LockOnceConnection:
            def __init__(self) -> None:
                self.task_attempts = 0
                self.rollback_count = 0
                self.commit_count = 0
                self.document_rows: list[tuple[object, ...]] = []

            def executemany(
                self, statement: str, rows: object
            ) -> None:
                if "INSERT INTO tasks" in statement:
                    self.task_attempts += 1
                    if self.task_attempts == 1:
                        raise sqlite3.OperationalError("database is locked")

            def execute(
                self, statement: str, row: tuple[object, ...]
            ) -> None:
                self.document_rows.append(row)

            def rollback(self) -> None:
                self.rollback_count += 1

            def commit(self) -> None:
                self.commit_count += 1

        connection = LockOnceConnection()
        with patch.object(module.time, "sleep") as sleep:
            module.ResumableCurrentTaskPacking._insert_document_rows(
                connection,
                (),
                (),
                ("retry-document", 0, 0),
                retry_locked=True,
            )
        self.assertEqual(connection.task_attempts, 2)
        self.assertEqual(connection.rollback_count, 1)
        self.assertEqual(connection.commit_count, 1)
        self.assertEqual(connection.document_rows, [("retry-document", 0, 0)])
        sleep.assert_called_once_with(
            module.TASK_DATABASE_LOCK_RETRY_INITIAL_SECONDS
        )

    def test_connection_setup_retries_locked_wal_initialization(self) -> None:
        module = importlib.import_module("bosscrafts_model.resumable_task_packing")

        class SetupCursor:
            def __init__(self, row: tuple[str] | None = None) -> None:
                self.row = row

            def fetchone(self) -> tuple[str] | None:
                return self.row

        class SetupConnection:
            def __init__(self, *, lock_journal: bool) -> None:
                self.lock_journal = lock_journal
                self.closed = False
                self.statements: list[str] = []

            def execute(self, statement: str) -> SetupCursor:
                self.statements.append(statement)
                if statement == "PRAGMA journal_mode":
                    return SetupCursor(("delete",))
                if self.lock_journal and statement == "PRAGMA journal_mode=WAL":
                    raise sqlite3.OperationalError("database is locked")
                return SetupCursor()

            def close(self) -> None:
                self.closed = True

        locked = SetupConnection(lock_journal=True)
        ready = SetupConnection(lock_journal=False)
        packing = object.__new__(module.ResumableCurrentTaskPacking)
        packing.database_path = Path("task-packing.sqlite")

        with (
            patch.object(module.sqlite3, "connect", side_effect=(locked, ready)) as connect,
            patch.object(module.time, "sleep") as sleep,
        ):
            connection = packing._connect()

        self.assertIs(connection, ready)
        self.assertTrue(locked.closed)
        self.assertFalse(ready.closed)
        self.assertEqual(connect.call_count, 2)
        self.assertEqual(
            ready.statements,
            [
                f"PRAGMA busy_timeout={module.TASK_DATABASE_BUSY_TIMEOUT_MS}",
                "PRAGMA journal_mode",
                "PRAGMA journal_mode=WAL",
                "PRAGMA synchronous=FULL",
            ],
        )
        sleep.assert_called_once_with(
            module.TASK_DATABASE_LOCK_RETRY_INITIAL_SECONDS
        )

    def test_connection_setup_does_not_reassert_existing_wal_mode(self) -> None:
        module = importlib.import_module("bosscrafts_model.resumable_task_packing")

        class SetupCursor:
            def __init__(self, row: tuple[str] | None = None) -> None:
                self.row = row

            def fetchone(self) -> tuple[str] | None:
                return self.row

        class ExistingWalConnection:
            def __init__(self) -> None:
                self.statements: list[str] = []

            def execute(self, statement: str) -> SetupCursor:
                self.statements.append(statement)
                if statement == "PRAGMA journal_mode":
                    return SetupCursor(("wal",))
                return SetupCursor()

            def close(self) -> None:
                raise AssertionError("ready connection must remain open")

        ready = ExistingWalConnection()
        packing = object.__new__(module.ResumableCurrentTaskPacking)
        packing.database_path = Path("task-packing.sqlite")

        with patch.object(module.sqlite3, "connect", return_value=ready):
            connection = packing._connect()

        self.assertIs(connection, ready)
        self.assertEqual(
            ready.statements,
            [
                f"PRAGMA busy_timeout={module.TASK_DATABASE_BUSY_TIMEOUT_MS}",
                "PRAGMA journal_mode",
                "PRAGMA synchronous=FULL",
            ],
        )

    def test_existing_complete_schema_initializes_without_ddl(self) -> None:
        module = importlib.import_module("bosscrafts_model.resumable_task_packing")
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tasks.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.executescript(
                    """
                    CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE documents(
                        document_id TEXT PRIMARY KEY,
                        task_count INTEGER NOT NULL,
                        rejected_count INTEGER NOT NULL
                    );
                    CREATE TABLE tasks(example_id TEXT PRIMARY KEY);
                    CREATE TABLE rejections(rejection_id TEXT PRIMARY KEY);
                    INSERT INTO metadata VALUES('fixture_key', 'fixture_value');
                    """
                )
                connection.commit()
            finally:
                connection.close()

            statements: list[str] = []

            def traced_connection() -> sqlite3.Connection:
                traced = sqlite3.connect(database)
                traced.set_trace_callback(statements.append)
                return traced

            packing = object.__new__(module.ResumableCurrentTaskPacking)
            packing.database_path = database
            packing._defer_cell_index = True
            packing._expected_metadata = {"fixture_key": "fixture_value"}

            with patch.object(
                module.ResumableCurrentTaskPacking,
                "_connect",
                autospec=True,
                side_effect=lambda _packing: traced_connection(),
            ):
                packing._initialize()

        self.assertFalse(
            any(statement.lstrip().upper().startswith("CREATE") for statement in statements)
        )

    def test_pending_scan_fetches_heavy_payload_only_after_filters(self) -> None:
        module = importlib.import_module("bosscrafts_model.resumable_task_packing")
        connection = sqlite3.connect(":memory:")
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        try:
            connection.execute(
                """CREATE TABLE documents(
                       document_id TEXT PRIMARY KEY,
                       content_sha256 TEXT NOT NULL,
                       language TEXT NOT NULL,
                       relative_path TEXT NOT NULL,
                       status TEXT NOT NULL,
                       semantic_spec_sha256 TEXT NOT NULL,
                       parser_runtime_identity_sha256 TEXT NOT NULL,
                       canonical_sha256 TEXT NOT NULL,
                       canonical_jsonl_zlib BLOB NOT NULL,
                       semantic_ids_zlib BLOB NOT NULL,
                       semantic_record_count INTEGER NOT NULL,
                       semantic_token_count INTEGER NOT NULL
                   )"""
            )
            for document_id in ("chosen", "complete", "ineligible"):
                connection.execute(
                    "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        document_id,
                        "c" * 64,
                        "python",
                        f"{document_id}.py",
                        "admitted",
                        "s" * 64,
                        "p" * 64,
                        "j" * 64,
                        b"heavy-canonical-payload",
                        b"heavy-semantic-payload",
                        1,
                        1,
                    ),
                )
            connection.commit()
            statements.clear()

            packing = object.__new__(module.ResumableCurrentTaskPacking)
            packing._eligible_ids = {"chosen", "complete"}
            rows = list(
                packing._iter_pending_admission_rows(
                    connection,
                    {"complete"},
                    worker_index=0,
                    worker_count=1,
                )
            )
        finally:
            connection.close()

        self.assertEqual([row[0] for row in rows], ["chosen"])
        payload_queries = [
            statement
            for statement in statements
            if "canonical_jsonl_zlib" in statement.casefold()
        ]
        self.assertEqual(len(payload_queries), 1)
        self.assertIn("document_id='chosen'", payload_queries[0])
        self.assertNotIn("complete", payload_queries[0])
        self.assertNotIn("ineligible", payload_queries[0])
        self.assertFalse(
            any(
                "select document_id from documents" in statement.casefold()
                and "order by document_id" in statement.casefold()
                for statement in statements
            )
        )

    def test_eligibility_admission_check_uses_indexed_id_lookups(self) -> None:
        module = importlib.import_module("bosscrafts_model.resumable_task_packing")
        connection = sqlite3.connect(":memory:")
        statements: list[str] = []
        try:
            connection.execute(
                "CREATE TABLE documents(document_id TEXT PRIMARY KEY, status TEXT NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO documents VALUES(?,?)",
                (("alpha", "admitted"), ("beta", "admitted"), ("gamma", "rejected")),
            )
            connection.commit()
            connection.set_trace_callback(statements.append)

            self.assertTrue(
                module.ResumableCurrentTaskPacking._eligible_ids_are_admitted(
                    connection, {"alpha", "beta"}
                )
            )
            self.assertFalse(
                module.ResumableCurrentTaskPacking._eligible_ids_are_admitted(
                    connection, {"alpha", "gamma"}
                )
            )
        finally:
            connection.close()

        self.assertTrue(statements)
        self.assertTrue(all(" IN (" in statement for statement in statements))
        self.assertTrue(all("status='admitted'" in statement for statement in statements))

    def test_worker_partitions_produce_the_same_complete_database_identity(self):
        module = importlib.import_module("bosscrafts_model.resumable_task_packing")
        candidates = tuple(
            _candidate(
                f"partition-doc-{index}",
                f"value_{index} = {index}\nprint(value_{index})\n",
            )
            for index in range(8)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.sqlite"
            admission_path = root / "admission.sqlite"
            partitioned_path = root / "partitioned.sqlite"
            serial_path = root / "serial.sqlite"
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
            partitioned = module.ResumableCurrentTaskPacking(
                self.binding,
                admission_path,
                catalog_path,
                partitioned_path,
            )
            first = partitioned.process(worker_index=0, worker_count=2)
            second = partitioned.process(worker_index=1, worker_count=2)
            self.assertEqual(
                first.processed_this_run + second.processed_this_run,
                len(candidates),
            )
            self.assertTrue(second.packing_complete)
            serial = module.ResumableCurrentTaskPacking(
                self.binding,
                admission_path,
                catalog_path,
                serial_path,
                defer_cell_index=True,
            )
            self.assertTrue(serial.process(commit_interval=4).packing_complete)
            partitioned_audit = module.audit_task_database(
                self.binding, partitioned_path
            )
            serial_audit = module.audit_task_database(self.binding, serial_path)
            self.assertEqual(
                partitioned_audit.database_identity_sha256,
                serial_audit.database_identity_sha256,
            )
            serial_connection = sqlite3.connect(serial_path)
            try:
                indexes = {
                    str(row[1])
                    for row in serial_connection.execute("PRAGMA index_list(tasks)")
                }
            finally:
                serial_connection.close()
            self.assertIn("tasks_cell", indexes)
            with self.assertRaisesRegex(ValueError, "worker index and count"):
                partitioned.process(worker_index=2, worker_count=2)
            with self.assertRaisesRegex(ValueError, "per-document commits"):
                partitioned.process(worker_count=2, commit_interval=2)

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
            dataset = module.ResumableTaskDataset(self.binding, tasks_path)
            evaluation_rows = list(dataset.iter_examples(split="validation"))
            self.assertGreaterEqual(len(evaluation_rows), 2)
            requested_ids = [
                str(evaluation_rows[-1]["example_id"]),
                str(evaluation_rows[0]["example_id"]),
            ]
            selected_rows = list(
                dataset.iter_examples_by_id(
                    requested_ids,
                    split="validation",
                )
            )
            self.assertEqual(
                [row["example_id"] for row in selected_rows],
                requested_ids,
            )
            frozen_evaluation = importlib.import_module(
                "bosscrafts_model.frozen_evaluation"
            )
            acceptance = importlib.import_module("bosscrafts_model.model_acceptance")
            cells: dict[str, list[str]] = {}
            for row in evaluation_rows:
                cell = f'{row["language"]}|{row["family"]}'
                cells.setdefault(cell, []).append(str(row["example_id"]))
            frozen_cells = {
                cell: tuple(sorted(example_ids))
                for cell, example_ids in cells.items()
            }
            selected_cells = dataset.select_frozen_example_ids_by_cell(
                split="validation",
                expected_cells={cell: 1 for cell in frozen_cells},
            )
            self.assertEqual(set(selected_cells), set(frozen_cells))
            self.assertTrue(
                all(len(example_ids) == 1 for example_ids in selected_cells.values())
            )
            self.assertEqual(
                selected_cells,
                dataset.select_frozen_example_ids_by_cell(
                    split="validation",
                    expected_cells={cell: 1 for cell in frozen_cells},
                ),
            )
            with self.assertRaisesRegex(ValueError, "complete"):
                dataset.select_frozen_example_ids_by_cell(
                    split="validation",
                    expected_cells={next(iter(frozen_cells)): 10_000},
                )
            suite = acceptance.GroundedHeldoutSuite(
                manifest_sha256="a" * 64,
                task_database_identity_sha256="b" * 64,
                evaluation_task_database_identities_sha256={
                    "validation": "c" * 64,
                    "test": "d" * 64,
                },
                evaluation_example_ids={
                    "validation": frozen_cells,
                    "test": {},
                },
                evaluation_example_ids_sha256={"validation": {}, "test": {}},
                category_example_ids={},
                category_example_ids_sha256={},
                category_example_counts={},
            )
            frozen_rows = list(
                frozen_evaluation.iter_frozen_evaluation_examples(
                    dataset,
                    suite,
                    split="validation",
                )
            )
            expected_frozen_ids = [
                example_id
                for cell in sorted(frozen_cells)
                for example_id in frozen_cells[cell]
            ]
            self.assertEqual(
                [row["example_id"] for row in frozen_rows],
                expected_frozen_ids,
            )
            first_cell = next(iter(sorted(frozen_cells)))
            wrong_cell = f'rust|{first_cell.split("|", 1)[1]}'
            if wrong_cell == first_cell:
                wrong_cell = f'go|{first_cell.split("|", 1)[1]}'
            forged_suite = acceptance.GroundedHeldoutSuite(
                manifest_sha256=suite.manifest_sha256,
                task_database_identity_sha256=suite.task_database_identity_sha256,
                evaluation_task_database_identities_sha256=(
                    suite.evaluation_task_database_identities_sha256
                ),
                evaluation_example_ids={
                    "validation": {
                        **frozen_cells,
                        wrong_cell: frozen_cells[first_cell],
                        first_cell: (),
                    },
                    "test": {},
                },
                evaluation_example_ids_sha256=suite.evaluation_example_ids_sha256,
                category_example_ids=suite.category_example_ids,
                category_example_ids_sha256=suite.category_example_ids_sha256,
                category_example_counts=suite.category_example_counts,
            )
            with self.assertRaisesRegex(ValueError, "cell"):
                list(
                    frozen_evaluation.iter_frozen_evaluation_examples(
                        dataset,
                        forged_suite,
                        split="validation",
                    )
                )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                list(
                    dataset.iter_examples_by_id(
                        [requested_ids[0], requested_ids[0]],
                        split="validation",
                    )
                )
            with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                list(
                    dataset.iter_examples_by_id(
                        ["not-an-example-id"],
                        split="validation",
                    )
                )
            missing_id = next(
                candidate
                for candidate in ("e" * 64, "f" * 64)
                if candidate not in requested_ids
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                list(
                    dataset.iter_examples_by_id(
                        [missing_id],
                        split="validation",
                    )
                )
            with self.assertRaisesRegex(ValueError, "split"):
                list(dataset.iter_examples_by_id(requested_ids, split="test"))


if __name__ == "__main__":
    unittest.main()
