from __future__ import annotations

import re
from pathlib import Path

import pytest
from test_frozen_evaluation_contract import (
    TEST_DATABASE_ID,
    VALIDATION_DATABASE_ID,
    _plural_manifest,
)

from scripts import run_bosscrafts_v10_instruction_training_chunk as v10_chunk
from scripts import (
    run_bosscrafts_v10_instruction_training_until_complete as v10_controller,
)
from scripts import run_bounded_training_trial as bounded_training


def test_v10_contract_verifier_rejects_manifest_database_substitution() -> None:
    manifest = _plural_manifest()
    database_identities = {
        "validation": VALIDATION_DATABASE_ID,
        "test": TEST_DATABASE_ID,
    }
    contract = v10_chunk.verify_v10_evaluation_contract(
        task_database_identities_sha256=database_identities,
        heldout_manifest=manifest,
    )
    assert contract == (
        database_identities,
        manifest["manifest_sha256"],
    )
    with pytest.raises(ValueError, match="lineage"):
        v10_chunk.verify_v10_evaluation_contract(
            task_database_identities_sha256={
                "validation": "f" * 64,
                "test": TEST_DATABASE_ID,
            },
            heldout_manifest=manifest,
        )


def test_training_contract_uses_the_preverified_manifest_without_databases() -> None:
    manifest = _plural_manifest()
    identities, manifest_sha256 = (
        bounded_training._verify_self_contained_heldout_manifest(manifest)
    )

    assert identities == {
        "validation": VALIDATION_DATABASE_ID,
        "test": TEST_DATABASE_ID,
    }
    assert manifest_sha256 == manifest["manifest_sha256"]


def test_v10_manifest_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "heldout.json"
    path.write_text('{"schema_version":"one","schema_version":"two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        v10_chunk.read_v10_heldout_manifest(path)


def test_v10_plan_hash_binds_frozen_evaluation_contract() -> None:
    evaluation_database_identities = {
        "validation": "a" * 64,
        "test": "b" * 64,
    }
    heldout_suite_identity = "b" * 64

    expected = v10_chunk.expected_training_plan_sha256(
        evaluation_database_identities,
        heldout_suite_identity,
    )

    assert re.fullmatch(r"[0-9a-f]{64}", expected)
    assert expected != v10_chunk.expected_training_plan_sha256(
        {"validation": "c" * 64, "test": "b" * 64},
        heldout_suite_identity,
    )
    assert expected != v10_chunk.expected_training_plan_sha256(
        evaluation_database_identities,
        "d" * 64,
    )
    with pytest.raises(ValueError, match="evaluation contract"):
        v10_chunk.expected_training_plan_sha256(
            {"validation": "invalid", "test": "b" * 64},
            heldout_suite_identity,
        )


def test_v10_bounded_command_carries_frozen_evaluation_artifacts() -> None:
    heldout_manifest = Path("heldout-suite-manifest.json")

    command = v10_chunk.build_bounded_training_command(
        database=Path("training.sqlite"),
        checkpoint=Path("checkpoint.pt"),
        status_output=Path("progress.json"),
        parent_database=Path("parent.sqlite"),
        parent_checkpoint=Path("parent.pt"),
        heldout_suite_manifest=heldout_manifest,
        resume=False,
    )

    assert command[command.index("--heldout-suite-manifest") + 1] == str(
        heldout_manifest
    )
    assert "--validation-database" not in command
    assert "--test-database" not in command
    assert "--resume" not in command
    assert command[command.index("--initial-model-checkpoint") + 1] == "parent.pt"


def test_v10_controller_propagates_and_checks_the_same_contract() -> None:
    heldout_manifest = Path("heldout-suite-manifest.json")
    evaluation_identities = {"validation": "a" * 64, "test": "b" * 64}
    heldout_identity = "b" * 64

    command = v10_controller.build_v10_chunk_command(
        status_path=Path("progress.json"),
        heldout_suite_manifest=heldout_manifest,
    )
    fields = v10_controller.expected_v10_status_fields(
        evaluation_task_database_identities_sha256=evaluation_identities,
        heldout_suite_manifest_sha256=heldout_identity,
    )

    assert command[command.index("--heldout-suite-manifest") + 1] == str(
        heldout_manifest
    )
    assert "--validation-database" not in command
    assert "--test-database" not in command
    assert fields["evaluation_task_database_identities_sha256"] == evaluation_identities
    assert fields["heldout_suite_manifest_sha256"] == heldout_identity
    assert fields["training_plan_sha256"] == v10_chunk.expected_training_plan_sha256(
        evaluation_identities,
        heldout_identity,
    )
