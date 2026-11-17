from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
import zlib
from collections import Counter
from pathlib import Path
from unittest import mock

from bosscrafts_model.bcnlt_source_pipeline import (
    PARSER_RUNTIME_IDENTITY_SHA256,
    PARSER_RUNTIME_NOT_APPLICABLE,
)
from bosscrafts_model.codex_replay_curriculum import build_codex_replay_curriculum
from bosscrafts_model.resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    ResumableTaskDataset,
    TaskDatabaseAuditResult,
    _canonical_json,
    _encode_u16,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _example_id(document_id: str, family: str, record_index: int, token_ids: list[int]) -> str:
    token_sha256 = hashlib.sha256(_canonical_json(token_ids).encode("utf-8")).hexdigest()
    return hashlib.sha256(
        _canonical_json(
            {
                "document_id": document_id,
                "family": family,
                "record_index": record_index,
                "input_ids_sha256": token_sha256,
            }
        ).encode("utf-8")
    ).hexdigest()


def _create_source(
    binding: NativeTokenizerBinding,
    path: Path,
    rows: list[tuple[str, str, str, int]],
    *,
    codex: bool,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            sorted(
                {
                    "schema_version": TASK_DATABASE_SCHEMA,
                    "binding_sha256": binding.binding_sha256,
                    "tokenizer_sha256": binding.tokenizer_sha256,
                    "vocabulary_size": str(binding.vocab_size),
                    "maximum_sequence_tokens": "2048",
                    "packing_complete": "true",
                    "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                    "curriculum_policy_sha256": "a" * 64,
                }.items()
            ),
        )
        connection.execute(
            """CREATE TABLE tasks(
                   example_id TEXT PRIMARY KEY,document_id TEXT NOT NULL,
                   language TEXT NOT NULL,family TEXT NOT NULL,
                   record_index INTEGER NOT NULL,split TEXT NOT NULL,
                   input_ids_zlib BLOB NOT NULL,token_count INTEGER NOT NULL,
                   loss_start_index INTEGER NOT NULL,
                   semantic_spec_sha256 TEXT NOT NULL,
                   parser_runtime_identity_sha256 TEXT NOT NULL,
                   source_sha256 TEXT NOT NULL
               )"""
        )
        if codex:
            connection.execute(
                """CREATE TABLE codex_provenance(
                       example_id TEXT PRIMARY KEY,content_sha256 TEXT NOT NULL UNIQUE,
                       source_parquet TEXT NOT NULL,source_parquet_sha256 TEXT NOT NULL,
                       source_row_index INTEGER NOT NULL,input_sha256 TEXT NOT NULL,
                       full_output_sha256 TEXT NOT NULL,final_code_sha256 TEXT NOT NULL,
                       fence_label TEXT NOT NULL,license_spdx TEXT NOT NULL
                   )"""
            )
            connection.execute(
                """CREATE TABLE source_shards(
                       name TEXT PRIMARY KEY,sha256 TEXT NOT NULL,size_bytes INTEGER NOT NULL,
                       source_rows INTEGER NOT NULL,admitted_rows INTEGER NOT NULL
                   )"""
            )
            connection.execute(
                "CREATE TABLE rejection_counts(reason_code TEXT PRIMARY KEY,count INTEGER NOT NULL)"
            )
            connection.execute("INSERT INTO source_shards VALUES('tiny.parquet',?,1,?,?)", ("b" * 64, len(rows), len(rows)))
        for index, (split, language, family, target_length) in enumerate(rows):
            document_id = hashlib.sha256(f"{path.name}|document|{index}".encode()).hexdigest()
            token_ids = [1, *([100 + index % 97] * target_length), 2]
            record_index = index
            example_id = _example_id(document_id, family, record_index, token_ids)
            connection.execute(
                "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    example_id,
                    document_id,
                    language,
                    family,
                    record_index,
                    split,
                    zlib.compress(_encode_u16(token_ids), level=9),
                    len(token_ids),
                    1,
                    "c" * 64,
                    (
                        PARSER_RUNTIME_NOT_APPLICABLE
                        if codex
                        else PARSER_RUNTIME_IDENTITY_SHA256
                    ),
                    "d" * 64,
                ),
            )
            if codex:
                connection.execute(
                    "INSERT INTO codex_provenance VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        example_id,
                        hashlib.sha256(f"content|{index}".encode()).hexdigest(),
                        "tiny.parquet",
                        "b" * 64,
                        index,
                        hashlib.sha256(f"input|{index}".encode()).hexdigest(),
                        hashlib.sha256(f"output|{index}".encode()).hexdigest(),
                        hashlib.sha256(f"code|{index}".encode()).hexdigest(),
                        language,
                        "Apache-2.0",
                    ),
                )
        connection.commit()
    finally:
        connection.close()


class CodeXReplayCurriculumTests(unittest.TestCase):
    @staticmethod
    def _minimal_sources(
        binding: NativeTokenizerBinding, root: Path
    ) -> tuple[Path, Path]:
        codex = root / "codex.sqlite"
        replay = root / "replay.sqlite"
        _create_source(
            binding,
            codex,
            [("train", "python", "instruction_to_code", 4)] * 8,
            codex=True,
        )
        _create_source(
            binding,
            replay,
            [("train", "python", "semantic_to_code", 4)] * 8,
            codex=False,
        )
        return codex, replay

    def test_builds_weighted_provenance_preserving_curriculum(self) -> None:
        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codex = root / "codex.sqlite"
            replay = root / "replay.sqlite"
            output = root / "mixed.sqlite"
            codex_rows = (
                [("train", "python", "instruction_to_code", 2 + index % 11) for index in range(256)]
                + [("train", "javascript", "instruction_to_code", 2 + index % 7) for index in range(64)]
                + [("validation", "python", "instruction_to_code", 4) for _ in range(4)]
                + [("test", "python", "instruction_to_code", 4) for _ in range(4)]
            )
            replay_rows: list[tuple[str, str, str, int]] = []
            for language in ("python", "javascript"):
                for family in ("code_to_semantic", "semantic_to_code", "semantic_continuation"):
                    replay_rows.extend(
                        ("train", language, family, 2 + index % 8) for index in range(16)
                    )
                    replay_rows.extend(("validation", language, family, 4) for _ in range(2))
                    replay_rows.extend(("test", language, family, 4) for _ in range(2))
            _create_source(binding, codex, codex_rows, codex=True)
            _create_source(binding, replay, replay_rows, codex=False)
            codex_identity = audit_task_database(binding, codex).database_identity_sha256
            replay_identity = audit_task_database(binding, replay).database_identity_sha256
            report = build_codex_replay_curriculum(
                binding,
                codex_database=codex,
                replay_database=replay,
                destination=output,
                replay_per_train_cell=8,
                codex_weight_fraction=0.80,
                order_window_size=40,
                minimum_replay_order_fraction=0.0,
                maximum_replay_order_fraction=1.0,
                expected_codex_identity=codex_identity,
                expected_replay_identity=replay_identity,
            )
            self.assertEqual(report["example_count"], len(codex_rows) + 6 * (8 + 2 + 2))
            self.assertAlmostEqual(report["codex_train_weight_fraction"], 0.80, places=12)
            self.assertAlmostEqual(report["replay_train_weight_fraction"], 0.20, places=12)
            self.assertEqual(report["replay_per_train_cell"], 8)
            dataset = ResumableTaskDataset(binding, output)
            weighted: Counter[str] = Counter()
            for row in dataset.iter_examples(split="train"):
                kind = "instruction" if row["family"] == "instruction_to_code" else "replay"
                weighted[kind] += float(row["sampling_weight"])
            self.assertAlmostEqual(weighted["instruction"] / sum(weighted.values()), 0.80, places=12)
            connection = sqlite3.connect(output)
            try:
                provenance_count = connection.execute("SELECT COUNT(*) FROM codex_provenance").fetchone()[0]
                source_counts = dict(
                    connection.execute(
                        "SELECT source_kind,COUNT(*) FROM source_example_identity GROUP BY source_kind"
                    )
                )
                replay_train_cells = connection.execute(
                    "SELECT language,family,COUNT(*) FROM tasks "
                    "WHERE split='train' AND family!='instruction_to_code' "
                    "GROUP BY language,family"
                ).fetchall()
                runtime_rows = dict(
                    connection.execute(
                        "SELECT source_kind,MIN(t.parser_runtime_identity_sha256) "
                        "FROM tasks AS t JOIN source_example_identity AS s USING(example_id) "
                        "GROUP BY source_kind"
                    )
                )
            finally:
                connection.close()
            self.assertEqual(provenance_count, len(codex_rows))
            self.assertEqual(source_counts["instruction"], len(codex_rows))
            self.assertTrue(all(count == 8 for _, _, count in replay_train_cells))
            self.assertEqual(
                runtime_rows,
                {
                    "instruction": PARSER_RUNTIME_NOT_APPLICABLE,
                    "replay": PARSER_RUNTIME_IDENTITY_SHA256,
                },
            )
            connection = sqlite3.connect(output)
            try:
                connection.execute(
                    "UPDATE tasks SET parser_runtime_identity_sha256=? "
                    "WHERE example_id=(SELECT example_id FROM tasks "
                    "WHERE family!='instruction_to_code' LIMIT 1)",
                    (PARSER_RUNTIME_NOT_APPLICABLE,),
                )
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(ValueError, "parser runtime lineage"):
                audit_task_database(binding, output)

    def test_rejects_sparse_replay_cells(self) -> None:
        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codex = root / "codex.sqlite"
            replay = root / "replay.sqlite"
            _create_source(
                binding,
                codex,
                [("train", "python", "instruction_to_code", 4)],
                codex=True,
            )
            _create_source(
                binding,
                replay,
                [("train", "python", "semantic_to_code", 4)] * 7,
                codex=False,
            )
            with self.assertRaisesRegex(ValueError, "insufficient training cells"):
                build_codex_replay_curriculum(
                    binding,
                    codex_database=codex,
                    replay_database=replay,
                    destination=root / "mixed.sqlite",
                    replay_per_train_cell=8,
                    minimum_replay_order_fraction=0.0,
                    maximum_replay_order_fraction=1.0,
                )

    def test_preserves_stale_temp_and_never_clobbers_racing_destination(self) -> None:
        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codex, replay = self._minimal_sources(binding, root)
            output = root / "mixed.sqlite"
            stale = root / ".mixed.sqlite.tmp"
            stale.write_bytes(b"unrelated-stale-state")

            real_audit = audit_task_database

            def audit_then_create_destination(
                audit_binding: NativeTokenizerBinding, database: str | Path
            ) -> TaskDatabaseAuditResult:
                result = real_audit(audit_binding, database)
                database_path = Path(database).resolve()
                if database_path not in {codex.resolve(), replay.resolve()}:
                    output.write_bytes(b"racing-owner")
                return result

            with mock.patch(
                "bosscrafts_model.codex_replay_curriculum.audit_task_database",
                side_effect=audit_then_create_destination,
            ):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    build_codex_replay_curriculum(
                        binding,
                        codex_database=codex,
                        replay_database=replay,
                        destination=output,
                        replay_per_train_cell=8,
                        minimum_replay_order_fraction=0.0,
                        maximum_replay_order_fraction=1.0,
                    )

            self.assertEqual(output.read_bytes(), b"racing-owner")
            self.assertEqual(stale.read_bytes(), b"unrelated-stale-state")
            self.assertEqual(list(root.glob(".mixed.sqlite.*.tmp")), [])

    def test_rejects_source_database_changed_during_build(self) -> None:
        binding = NativeTokenizerBinding.load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codex, replay = self._minimal_sources(binding, root)
            output = root / "mixed.sqlite"
            real_audit = audit_task_database
            source_changed = False

            def audit_then_change_source(
                audit_binding: NativeTokenizerBinding, database: str | Path
            ) -> TaskDatabaseAuditResult:
                nonlocal source_changed
                result = real_audit(audit_binding, database)
                database_path = Path(database).resolve()
                if (
                    database_path not in {codex.resolve(), replay.resolve()}
                    and not source_changed
                ):
                    connection = sqlite3.connect(codex)
                    try:
                        connection.execute(
                            "UPDATE metadata SET value=? WHERE key=?",
                            ("f" * 64, "curriculum_policy_sha256"),
                        )
                        connection.commit()
                    finally:
                        connection.close()
                    source_changed = True
                return result

            with mock.patch(
                "bosscrafts_model.codex_replay_curriculum.audit_task_database",
                side_effect=audit_then_change_source,
            ):
                with self.assertRaisesRegex(ValueError, "changed during build"):
                    build_codex_replay_curriculum(
                        binding,
                        codex_database=codex,
                        replay_database=replay,
                        destination=output,
                        replay_per_train_cell=8,
                        minimum_replay_order_fraction=0.0,
                        maximum_replay_order_fraction=1.0,
                    )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
