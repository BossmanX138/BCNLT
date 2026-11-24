from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import zlib
from array import array
from pathlib import Path

import pytest

from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.model_acceptance import (
    EXPECTED_LANGUAGES,
    MINIMUM_CATEGORY_EXAMPLES,
)
from bosscrafts_model.packed_category_case_sources import (
    PACKED_CATEGORIES,
    freeze_packed_category_cases,
)
from bosscrafts_model.resumable_task_packing import TASK_DATABASE_SCHEMA
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _u16(values: list[int]) -> bytes:
    encoded = array("H", values)
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def _task_database(path: Path, binding: NativeTokenizerBinding) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        connection.execute(
            """CREATE TABLE tasks (
                example_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                language TEXT NOT NULL, family TEXT NOT NULL,
                record_index INTEGER NOT NULL, split TEXT NOT NULL,
                input_ids_zlib BLOB NOT NULL, token_count INTEGER NOT NULL,
                loss_start_index INTEGER NOT NULL,
                semantic_spec_sha256 TEXT NOT NULL,
                parser_runtime_identity_sha256 TEXT NOT NULL,
                source_sha256 TEXT NOT NULL
            )"""
        )
        metadata = {
            "schema_version": TASK_DATABASE_SCHEMA,
            "binding_sha256": binding.binding_sha256,
            "tokenizer_sha256": binding.tokenizer_sha256,
            "vocabulary_size": str(binding.vocab_size),
            "maximum_sequence_tokens": "64",
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "packing_complete": "true",
        }
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)", sorted(metadata.items())
        )
        rows = []
        families = (
            "code_to_semantic",
            "semantic_continuation",
            "semantic_to_code",
        )
        for language in EXPECTED_LANGUAGES:
            for family in families:
                for ordinal in range(2):
                    document_id = hashlib.sha256(
                        f"{language}:{family}:{ordinal}".encode()
                    ).hexdigest()
                    input_ids = [1, 2, 3 + ordinal]
                    input_hash = hashlib.sha256(
                        _canonical(input_ids).encode()
                    ).hexdigest()
                    identity = {
                        "document_id": document_id,
                        "family": family,
                        "record_index": ordinal,
                        "input_ids_sha256": input_hash,
                    }
                    example_id = hashlib.sha256(
                        _canonical(identity).encode()
                    ).hexdigest()
                    rows.append(
                        (
                            example_id,
                            document_id,
                            language,
                            family,
                            ordinal,
                            "test",
                            zlib.compress(_u16(input_ids), 6),
                            len(input_ids),
                            1,
                            CURRENT_RUNTIME_SPEC_SHA256[language],
                            PARSER_RUNTIME_IDENTITY_SHA256,
                            hashlib.sha256(document_id.encode()).hexdigest(),
                        )
                    )
        connection.executemany("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        connection.commit()
    finally:
        connection.close()


def test_freezes_balanced_disjoint_packed_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = NativeTokenizerBinding.load()
    database = tmp_path / "tasks.sqlite"
    _task_database(database, binding)
    for category in PACKED_CATEGORIES:
        monkeypatch.setitem(
            MINIMUM_CATEGORY_EXAMPLES, category, len(EXPECTED_LANGUAGES)
        )
    output_root = tmp_path / "cases"
    outputs = {
        category: output_root / f"{category}.jsonl"
        for category in PACKED_CATEGORIES
    }

    result = freeze_packed_category_cases(
        binding,
        database,
        outputs,
        output_root / "audit.json",
    )

    all_task_ids: set[str] = set()
    for category in PACKED_CATEGORIES:
        records = [
            json.loads(line)
            for line in outputs[category].read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == len(EXPECTED_LANGUAGES)
        assert [row["example_id"] for row in records] == sorted(
            row["example_id"] for row in records
        )
        assert {row["case"]["language"] for row in records} == set(
            EXPECTED_LANGUAGES
        )
        task_ids = {row["case"]["task_example_id"] for row in records}
        assert not all_task_ids.intersection(task_ids)
        all_task_ids.update(task_ids)
    audit = json.loads(result.audit_output.read_text(encoding="utf-8"))
    assert audit["audit_sha256"] == result.audit_sha256
    assert audit["record_counts"] == {
        category: len(EXPECTED_LANGUAGES) for category in PACKED_CATEGORIES
    }
    original = {category: path.read_bytes() for category, path in outputs.items()}
    with pytest.raises(ValueError, match="already exists"):
        freeze_packed_category_cases(
            binding,
            database,
            outputs,
            output_root / "audit.json",
        )
    assert {category: path.read_bytes() for category, path in outputs.items()} == original
