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
from bosscrafts_model.current_task_packing import FAMILIES
from bosscrafts_model.mixed_curriculum import (
    SPLIT_SAFE_MIXED_TRAINING_SCHEMA,
    build_mixed_curriculum,
    build_split_safe_mixed_training_curriculum,
)
from bosscrafts_model.mixed_task_loss_mask_audit import (
    MixedTaskLossMaskAuditor,
    SplitSafeMixedTrainingLossMaskAuditor,
)
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
from bosscrafts_model.task_database_audit_manifest import (
    load_verified_task_database_audit_manifest,
    materialize_task_database_audit_manifest,
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
                    "parser_runtime_identity_sha256": (PARSER_RUNTIME_IDENTITY_SHA256),
                    "maximum_sequence_tokens": "16",
                    "packing_complete": "true",
                }.items()
            ),
        )
        for index, split in enumerate(("train", "validation", "test")):
            family = INSTRUCTION_RESPONSE_FAMILY if instruction else "semantic_to_code"
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
            example_id = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
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


def _write_complete_semantic_train_database(
    path: Path, binding: NativeTokenizerBinding
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
                    "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                    "maximum_sequence_tokens": "16",
                    "packing_complete": "true",
                }.items()
            ),
        )
        for index, (language, family) in enumerate(
            (language, family)
            for language in sorted(CURRENT_RUNTIME_SPEC_SHA256)
            for family in FAMILIES
        ):
            document_id = f"semantic-{language}-{family}"
            input_ids = [100 + index, 300 + index, 500 + index, 700 + index]
            identity = {
                "document_id": document_id,
                "family": family,
                "record_index": 0,
                "input_ids_sha256": hashlib.sha256(
                    _canonical_json(input_ids).encode()
                ).hexdigest(),
            }
            example_id = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
            connection.execute(
                "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    example_id,
                    document_id,
                    language,
                    family,
                    0,
                    "train",
                    zlib.compress(_encode_u16(input_ids), 6),
                    len(input_ids),
                    2,
                    CURRENT_RUNTIME_SPEC_SHA256[language],
                    PARSER_RUNTIME_IDENTITY_SHA256,
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

    split_safe_result = subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "scripts"
                / "build_v10_split_safe_mixed_training_curriculum.py"
            ),
            "--help",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert split_safe_result.returncode == 0, split_safe_result.stderr
    assert "--semantic-train-database" in split_safe_result.stdout
    assert "--semantic-train-identity-sha256" in split_safe_result.stdout

    split_safe_audit_result = subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "scripts"
                / "audit_v10_split_safe_mixed_training_loss_masks.py"
            ),
            "--help",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert split_safe_audit_result.returncode == 0, split_safe_audit_result.stderr
    assert "--database" in split_safe_audit_result.stdout
    assert "--output" in split_safe_audit_result.stdout

    audit_result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "audit_v10_mixed_task_loss_masks.py"),
            "--help",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert audit_result.returncode == 0, audit_result.stderr
    assert "--database" in audit_result.stdout


def test_merges_sources_with_exact_lineage_and_half_instruction_weight(
    tmp_path: Path,
) -> None:
    binding = NativeTokenizerBinding.load()
    semantic = tmp_path / "semantic.sqlite"
    instruction = tmp_path / "instruction.sqlite"
    output = tmp_path / "mixed.sqlite"
    _write_database(semantic, binding, instruction=False)
    _write_database(instruction, binding, instruction=True)
    semantic_identity = audit_task_database(binding, semantic).database_identity_sha256
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

    loss_mask_result = MixedTaskLossMaskAuditor(
        binding,
        expected_languages=("python",),
        expected_semantic_families=("semantic_to_code",),
    ).materialize(output, tmp_path / "mixed-loss-mask.json")
    assert loss_mask_result.audit_passed is True
    assert loss_mask_result.training_ready is False
    assert loss_mask_result.semantic_example_count == 3
    assert loss_mask_result.instruction_example_count == 3
    assert loss_mask_result.split_example_counts == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert loss_mask_result.observed_instruction_training_weight_fraction == (
        pytest.approx(0.5)
    )


def test_builds_train_only_split_safe_mix_and_excludes_both_heldout_splits(
    tmp_path: Path,
) -> None:
    binding = NativeTokenizerBinding.load()
    semantic = tmp_path / "semantic-train.sqlite"
    instruction = tmp_path / "instruction.sqlite"
    output = tmp_path / "mixed-train.sqlite"
    _write_complete_semantic_train_database(semantic, binding)
    _write_database(instruction, binding, instruction=True)
    semantic_manifest = tmp_path / "semantic-task-audit.json"
    instruction_manifest = tmp_path / "instruction-task-audit.json"
    output_manifest = tmp_path / "mixed-task-audit.json"
    semantic_identity = materialize_task_database_audit_manifest(
        binding, semantic, semantic_manifest
    ).audit.database_identity_sha256
    instruction_identity = materialize_task_database_audit_manifest(
        binding, instruction, instruction_manifest
    ).audit.database_identity_sha256

    result = build_split_safe_mixed_training_curriculum(
        binding,
        semantic_train_database=semantic,
        instruction_database=instruction,
        output_database=output,
        expected_semantic_train_identity_sha256=semantic_identity,
        expected_instruction_identity_sha256=instruction_identity,
        instruction_training_weight_fraction=0.5,
        semantic_train_audit_manifest=semantic_manifest,
        instruction_audit_manifest=instruction_manifest,
        output_audit_manifest=output_manifest,
    )

    assert result.semantic_example_count == len(CURRENT_RUNTIME_SPEC_SHA256) * len(
        FAMILIES
    )
    assert result.instruction_train_example_count == 1
    assert result.excluded_instruction_validation_example_count == 1
    assert result.excluded_instruction_test_example_count == 1
    assert result.instruction_training_weight_fraction == pytest.approx(0.5)
    assert result.audit_manifest_sha256 is not None
    verified_output = load_verified_task_database_audit_manifest(
        binding,
        output,
        output_manifest,
        expected_database_identity_sha256=result.database_identity_sha256,
    )
    assert verified_output.manifest_sha256 == result.audit_manifest_sha256
    audit = audit_task_database(binding, output)
    assert audit.split_counts == {"train": result.example_count}
    dataset = ResumableTaskDataset(binding, output)
    metadata = dataset.metadata
    assert metadata["curriculum_schema_version"] == SPLIT_SAFE_MIXED_TRAINING_SCHEMA
    assert metadata["curriculum_source_split"] == "train"
    assert metadata["excluded_instruction_validation_example_count"] == "1"
    assert metadata["excluded_instruction_test_example_count"] == "1"
    connection = sqlite3.connect(output)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE split != 'train'"
        ).fetchone()[0] == 0
        assert dict(
            connection.execute(
                "SELECT source_kind,COUNT(*) FROM source_example_identity "
                "GROUP BY source_kind"
            )
        ) == {
            "instruction": 1,
            "replay": result.semantic_example_count,
        }
    finally:
        connection.close()
    receipt = SplitSafeMixedTrainingLossMaskAuditor(binding).materialize(
        output,
        tmp_path / "split-safe-loss-mask.json",
        audit_manifest=output_manifest,
    )
    assert receipt.audit_passed is True
    assert receipt.training_ready is False
    assert receipt.semantic_example_count == result.semantic_example_count
    assert receipt.instruction_train_example_count == 1
    assert receipt.excluded_instruction_validation_example_count == 1
    assert receipt.excluded_instruction_test_example_count == 1
    assert receipt.observed_instruction_training_weight_fraction == pytest.approx(0.5)


def test_mixed_loss_mask_audit_rejects_weight_policy_mismatch(
    tmp_path: Path,
) -> None:
    binding = NativeTokenizerBinding.load()
    semantic = tmp_path / "semantic.sqlite"
    instruction = tmp_path / "instruction.sqlite"
    output = tmp_path / "mixed.sqlite"
    _write_database(semantic, binding, instruction=False)
    _write_database(instruction, binding, instruction=True)
    build_mixed_curriculum(
        binding,
        semantic_database=semantic,
        instruction_database=instruction,
        output_database=output,
        expected_semantic_identity_sha256=(
            audit_task_database(binding, semantic).database_identity_sha256
        ),
        expected_instruction_identity_sha256=(
            audit_task_database(binding, instruction).database_identity_sha256
        ),
        instruction_training_weight_fraction=0.5,
    )
    connection = sqlite3.connect(output)
    try:
        connection.execute(
            "UPDATE metadata SET value='0.6' "
            "WHERE key='instruction_training_weight_fraction'"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="does not match policy"):
        MixedTaskLossMaskAuditor(
            binding,
            expected_languages=("python",),
            expected_semantic_families=("semantic_to_code",),
        ).materialize(output, tmp_path / "should-not-exist.json")


def test_task_audit_streams_and_rejects_truncated_source_identity_map(
    tmp_path: Path,
) -> None:
    binding = NativeTokenizerBinding.load()
    semantic = tmp_path / "semantic.sqlite"
    instruction = tmp_path / "instruction.sqlite"
    output = tmp_path / "mixed.sqlite"
    _write_database(semantic, binding, instruction=False)
    _write_database(instruction, binding, instruction=True)
    build_mixed_curriculum(
        binding,
        semantic_database=semantic,
        instruction_database=instruction,
        output_database=output,
        expected_semantic_identity_sha256=(
            audit_task_database(binding, semantic).database_identity_sha256
        ),
        expected_instruction_identity_sha256=(
            audit_task_database(binding, instruction).database_identity_sha256
        ),
        instruction_training_weight_fraction=0.5,
    )
    connection = sqlite3.connect(output)
    try:
        connection.execute(
            "DELETE FROM source_example_identity WHERE example_id=("
            "SELECT MAX(example_id) FROM source_example_identity)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="identity map count mismatch"):
        audit_task_database(binding, output)


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
