from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from array import array
from pathlib import Path
from unittest.mock import patch

import pytest

from bosscrafts_model.bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from bosscrafts_model.candidate_snapshot import freeze_candidate_snapshot
from bosscrafts_model.resumable_task_packing import TASK_DATABASE_SCHEMA
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_source_database(path: Path, binding: NativeTokenizerBinding) -> str:
    document_id = "document-0001"
    input_ids = [100, 101, 102, 103]
    input_sha256 = hashlib.sha256(_canonical_json(input_ids).encode()).hexdigest()
    example_id = hashlib.sha256(
        _canonical_json(
            {
                "document_id": document_id,
                "family": "code_to_semantic",
                "record_index": 0,
                "input_ids_sha256": input_sha256,
            }
        ).encode()
    ).hexdigest()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            sorted(
                {
                    "schema_version": TASK_DATABASE_SCHEMA,
                    "binding_sha256": binding.binding_sha256,
                    "tokenizer_sha256": binding.tokenizer_sha256,
                    "vocabulary_size": str(binding.vocab_size),
                    "maximum_sequence_tokens": "4",
                    "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                    "packing_complete": "false",
                }.items()
            ),
        )
        connection.execute(
            "CREATE TABLE documents(document_id TEXT PRIMARY KEY,task_count INTEGER NOT NULL,rejected_count INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO documents VALUES(?,?,?)", (document_id, 1, 0))
        connection.execute(
            "CREATE TABLE rejections(rejection_id TEXT PRIMARY KEY,document_id TEXT NOT NULL,language TEXT NOT NULL,record_index INTEGER NOT NULL,family TEXT,reason_code TEXT NOT NULL)"
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
                   source_sha256 TEXT NOT NULL)"""
        )
        connection.execute(
            "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                example_id,
                document_id,
                "python",
                "code_to_semantic",
                0,
                "train",
                zlib.compress(array("H", input_ids).tobytes()),
                len(input_ids),
                2,
                "a" * 64,
                PARSER_RUNTIME_IDENTITY_SHA256,
                hashlib.sha256(b"source").hexdigest(),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return document_id


def _write_admission_database(path: Path, document_id: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE documents(document_id TEXT PRIMARY KEY,status TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO documents VALUES(?, 'admitted')", (document_id,))
        connection.commit()
    finally:
        connection.close()


def test_snapshot_is_audited_before_unique_atomic_publication(tmp_path: Path) -> None:
    binding = NativeTokenizerBinding.load()
    source = tmp_path / "source.sqlite"
    admission = tmp_path / "admission.sqlite"
    document_id = _write_source_database(source, binding)
    _write_admission_database(admission, document_id)

    failed = tmp_path / "failed.sqlite"
    with (
        patch(
            "bosscrafts_model.candidate_snapshot.audit_task_database",
            side_effect=ValueError("injected snapshot audit failure"),
        ),
        pytest.raises(ValueError, match="injected snapshot audit failure"),
    ):
        freeze_candidate_snapshot(binding, source, admission, failed)
    assert not failed.exists()

    destination = tmp_path / "snapshot.sqlite"
    stale_temporary = tmp_path / ".snapshot.sqlite.tmp"
    stale_temporary.write_bytes(b"unrelated sentinel")
    status = freeze_candidate_snapshot(binding, source, admission, destination)

    assert destination.is_file()
    assert status.task_count == 1
    assert status.processed_document_count == 1
    assert stale_temporary.read_bytes() == b"unrelated sentinel"
