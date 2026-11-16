from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
    PARSER_RUNTIME_NOT_APPLICABLE,
)
from bosscrafts_model.mixed_curriculum import build_mixed_curriculum
from bosscrafts_model.resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    ResumableTaskDataset,
    _canonical_json,
    _encode_u16,
    audit_task_database,
)
from bosscrafts_model.reviewed_instruction_curriculum import (
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_RESPONSE_FAMILY,
    INSTRUCTION_SEMANTIC_SPEC_SHA256,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _write_database(
    path: Path,
    binding: NativeTokenizerBinding,
    *,
    instruction: bool,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
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
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            sorted(
                {
                    "schema_version": TASK_DATABASE_SCHEMA,
                    "binding_sha256": binding.binding_sha256,
                    "tokenizer_sha256": binding.tokenizer_sha256,
                    "vocabulary_size": str(binding.vocab_size),
                    "parser_runtime_identity_sha256": (
                        PARSER_RUNTIME_IDENTITY_SHA256
                    ),
                    "maximum_sequence_tokens": "16",
                    "packing_complete": "true",
                }.items()
            ),
        )
        for index, split in enumerate(("train", "validation", "test")):
            family = (
                INSTRUCTION_RESPONSE_FAMILY if instruction else "semantic_to_code"
            )
            language = INSTRUCTION_LANGUAGE if instruction else "python"
            document_id = f"{'instruction' if instruction else 'semantic'}-{split}"
            input_ids = [100 + index, 200 + index, 300 + index, 400 + index]
            identity = {
                "document_id": document_id,
                "family": family,
                "record_index": 0,
                "input_ids_sha256": hashlib.sha256(
                    _canonical_json(input_ids).encode()
                ).hexdigest(),
            }
            example_id = hashlib.sha256(
                _canonical_json(identity).encode()
            ).hexdigest()
            connection.execute(
                "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    example_id,
                    document_id,
                    language,
                    family,
                    0,
                    split,
                    zlib.compress(_encode_u16(input_ids), 6),
                    len(input_ids),
                    2,
                    (
                        INSTRUCTION_SEMANTIC_SPEC_SHA256
                        if instruction
                        else CURRENT_RUNTIME_SPEC_SHA256["python"]
                    ),
                    (
                        PARSER_RUNTIME_NOT_APPLICABLE
                        if instruction
                        else PARSER_RUNTIME_IDENTITY_SHA256
                    ),
                    hashlib.sha256(document_id.encode()).hexdigest(),
                ),
            )
        connection.commit()
    finally:
        connection.close()


def test_cli_runs_from_project_root() -> None:
    project_root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_v10_mixed_curriculum.py"),
            "--help",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--instruction-weight-fraction" in result.stdout


def test_merges_sources_with_exact_lineage_and_half_instruction_weight(
    tmp_path: Path,
) -> None:
    binding = NativeTokenizerBinding.load()
    semantic = tmp_path / "semantic.sqlite"
    instruction = tmp_path / "instruction.sqlite"
    output = tmp_path / "mixed.sqlite"
    _write_database(semantic, binding, instruction=False)
    _write_database(instruction, binding, instruction=True)
    semantic_identity = audit_task_database(
        binding, semantic
    ).database_identity_sha256
    instruction_identity = audit_task_database(
        binding, instruction
    ).database_identity_sha256

    result = build_mixed_curriculum(
        binding,
        semantic_database=semantic,
        instruction_database=instruction,
        output_database=output,
        expected_semantic_identity_sha256=semantic_identity,
        expected_instruction_identity_sha256=instruction_identity,
        instruction_training_weight_fraction=0.5,
    )

    assert result.example_count == 6
    assert result.semantic_example_count == 3
    assert result.instruction_example_count == 3
    assert result.instruction_training_weight_fraction == pytest.approx(0.5)
    dataset = ResumableTaskDataset(binding, output)
    train = list(dataset.iter_examples(split="train"))
    instruction_weight = sum(
        float(row["sampling_weight"])
        for row in train
        if row["family"] == INSTRUCTION_RESPONSE_FAMILY
    )
    assert instruction_weight / sum(float(row["sampling_weight"]) for row in train) == (
        pytest.approx(0.5)
    )
    connection = sqlite3.connect(output)
    try:
        kinds = dict(
            connection.execute(
                "SELECT source_kind,COUNT(*) FROM source_example_identity "
                "GROUP BY source_kind"
            )
        )
    finally:
        connection.close()
    assert kinds == {"instruction": 3, "replay": 3}


def test_rejects_wrong_expected_source_identity(tmp_path: Path) -> None:
    binding = NativeTokenizerBinding.load()
    semantic = tmp_path / "semantic.sqlite"
    instruction = tmp_path / "instruction.sqlite"
    _write_database(semantic, binding, instruction=False)
    _write_database(instruction, binding, instruction=True)
    instruction_identity = audit_task_database(
        binding, instruction
    ).database_identity_sha256

    with pytest.raises(ValueError, match="source identity mismatch"):
        build_mixed_curriculum(
            binding,
            semantic_database=semantic,
            instruction_database=instruction,
            output_database=tmp_path / "mixed.sqlite",
            expected_semantic_identity_sha256="f" * 64,
            expected_instruction_identity_sha256=instruction_identity,
            instruction_training_weight_fraction=0.5,
        )
