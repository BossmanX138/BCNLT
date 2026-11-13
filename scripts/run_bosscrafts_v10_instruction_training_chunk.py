"""Run one guarded v10 instruction-to-code chunk from preserved v8."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.balanced_curriculum import TASK_FAMILIES
from bosscrafts_model.bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from bosscrafts_model.instruction_curriculum import INSTRUCTION_FAMILY
from bosscrafts_model.model_acceptance import verify_heldout_suite_manifest
from bosscrafts_model.resumable_task_packing import (
    CELL_WEIGHT_MULTIPLIERS_METADATA_KEY,
    TASK_DATABASE_SCHEMA,
)

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-instruction-replay-balanced-v10"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v10-instruction"
    / "checkpoint.pt"
)
DEFAULT_PARENT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-deterministic-clean-v5"
    / "tasks.sqlite"
)
DEFAULT_PARENT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v8-consolidation"
    / "checkpoint.pt"
)
EXPECTED_DATABASE_IDENTITY = "14be957f8fa947ede8d8a3cf84bc2e023550c8fe8b976ccb34748c66a72b05bd"
EXPECTED_POLICY_SHA256 = "e06e8456e2f0ad7ae855cf07b03bbfcf83050f673b35ded399b416312ee3c0b4"
EXPECTED_INSTRUCTION_SOURCE_IDENTITY = (
    "d1d94a21744cc8857ce32d3b98ea2459741e7e9cda9d62548e7e362f5da80e5d"
)
EXPECTED_MODEL_FINGERPRINT = "383bc3acb9bea01ed9a40fb6293a5ac81ee7ffb27ffbc4fdcfeafd96358b3bca"
PARENT_CHECKPOINT_SHA256 = "1c367f2440415639ed1ce52e2fc85f2862e198e62f179095450674f7f734b519"
PARENT_DATABASE_IDENTITY_SHA256 = "626186864cf4444db997bea5bd8ea4a18f18b5042509033a0d6939d0bd35afd2"
REPLAY_PER_CELL = 256
INSTRUCTION_TRAIN_EXAMPLES = 4096
TRAIN_EXAMPLES = 14_848
GRADIENT_ACCUMULATION = 8
PLANNED_OPTIMIZER_STEPS = math.ceil(TRAIN_EXAMPLES / GRADIENT_ACCUMULATION)
LEARNING_RATE = 0.00002
WARMUP_RATIO = 0.01
MINIMUM_LEARNING_RATE_RATIO = 0.1
MAX_HELDOUT_MANIFEST_BYTES = 16 * 1024 * 1024


def expected_initialization() -> dict[str, object]:
    return {
        "type": "verified_completed_checkpoint",
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "task_database_identity_sha256": PARENT_DATABASE_IDENTITY_SHA256,
    }


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate held-out manifest key: {key}")
        payload[key] = value
    return payload


def read_v10_heldout_manifest(path: Path) -> dict[str, object]:
    try:
        content = path.resolve().read_bytes()
        if not content or len(content) > MAX_HELDOUT_MANIFEST_BYTES:
            raise ValueError("v10 held-out manifest size is invalid")
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v10 held-out manifest is unreadable") from error
    if type(payload) is not dict:
        raise ValueError("v10 held-out manifest must be a JSON object")
    return payload


def verify_v10_evaluation_contract(
    *,
    task_database_identities_sha256: Mapping[str, str],
    heldout_manifest: dict[str, object],
) -> tuple[dict[str, str], str]:
    suite = verify_heldout_suite_manifest(
        heldout_manifest,
        task_database_identities_sha256=task_database_identities_sha256,
    )
    return suite.evaluation_task_database_identities_sha256, suite.manifest_sha256


def v10_evaluation_identities_from_manifest(
    heldout_manifest: dict[str, object],
) -> dict[str, str]:
    value = heldout_manifest.get("evaluation_task_database_identities_sha256")
    if (
        type(value) is not dict
        or set(value) != {"validation", "test"}
        or any(not _is_sha256(identity) for identity in value.values())
    ):
        raise ValueError("v10 held-out manifest database identities are invalid")
    return {split: str(identity) for split, identity in value.items()}


def expected_training_plan_sha256(
    evaluation_task_database_identities_sha256: Mapping[str, str],
    heldout_suite_manifest_sha256: str,
) -> str:
    if (
        set(evaluation_task_database_identities_sha256) != {"validation", "test"}
        or any(
            not _is_sha256(value)
            for value in evaluation_task_database_identities_sha256.values()
        )
        or not _is_sha256(heldout_suite_manifest_sha256)
    ):
        raise ValueError("v10 frozen evaluation contract is invalid")
    warmup_steps = min(
        PLANNED_OPTIMIZER_STEPS - 1,
        int(PLANNED_OPTIMIZER_STEPS * WARMUP_RATIO),
    )
    plan = {
        "dtype": "bfloat16",
        "epochs": 1,
        "full_model": True,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "gradient_checkpointing": True,
        "optimizer": {
            "amsgrad": False,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "learning_rate": LEARNING_RATE,
            "name": "AdamW",
            "weight_decay": 0.01,
        },
        "schedule": {
            "minimum_learning_rate_ratio": MINIMUM_LEARNING_RATE_RATIO,
            "name": "linear_warmup_cosine_decay",
            "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
            "warmup_ratio": WARMUP_RATIO,
            "warmup_steps": warmup_steps,
        },
        "schema_version": "bosscrafts-training-plan-v4",
        "seed": 7,
        "split": "train",
        "initialization": expected_initialization(),
        "evaluation_contract": {
            "evaluation_task_database_identities_sha256": (
                dict(evaluation_task_database_identities_sha256)
            ),
            "heldout_suite_manifest_sha256": heldout_suite_manifest_sha256,
        },
    }
    return hashlib.sha256(
        json.dumps(plan, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_bounded_training_command(
    *,
    database: Path,
    checkpoint: Path,
    status_output: Path,
    parent_database: Path,
    parent_checkpoint: Path,
    validation_database: Path,
    test_database: Path,
    heldout_suite_manifest: Path,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_bounded_training_trial.py"),
        "--database",
        str(database),
        "--validation-database",
        str(validation_database),
        "--test-database",
        str(test_database),
        "--heldout-suite-manifest",
        str(heldout_suite_manifest),
        "--checkpoint",
        str(checkpoint),
        "--status-output",
        str(status_output),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--full-model",
        "--gradient-checkpointing",
        "--gradient-accumulation",
        str(GRADIENT_ACCUMULATION),
        "--learning-rate",
        str(LEARNING_RATE),
        "--warmup-ratio",
        str(WARMUP_RATIO),
        "--minimum-learning-rate-ratio",
        str(MINIMUM_LEARNING_RATE_RATIO),
        "--max-steps",
        "100",
        "--epochs",
        "1",
        "--initialization-checkpoint-sha256",
        PARENT_CHECKPOINT_SHA256,
        "--initialization-database-identity-sha256",
        PARENT_DATABASE_IDENTITY_SHA256,
    ]
    if resume:
        command.append("--resume")
    else:
        command.extend(
            [
                "--initial-model-checkpoint",
                str(parent_checkpoint),
                "--initial-checkpoint-database",
                str(parent_database),
            ]
        )
    return command


def _verify_database(database: Path) -> None:
    if not database.is_file():
        raise ValueError("v10 instruction replay curriculum does not exist")
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        observed = {
            (str(split), str(language), str(family)): int(count)
            for split, language, family, count in connection.execute(
                "SELECT split,language,family,COUNT(*) FROM tasks "
                "GROUP BY split,language,family"
            )
        }
        train_count, maximum_sequence = connection.execute(
            "SELECT COUNT(*),MAX(token_count) FROM tasks WHERE split='train'"
        ).fetchone()
    finally:
        connection.close()
    try:
        multipliers = json.loads(metadata[CELL_WEIGHT_MULTIPLIERS_METADATA_KEY])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("v10 instruction replay multipliers are invalid") from error
    if (
        metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or metadata.get("packing_complete") != "true"
        or metadata.get("curriculum_schema_version")
        != "bosscrafts-instruction-replay-curriculum-v1"
        or metadata.get("maximum_sequence_tokens") != "2048"
        or metadata.get("instruction_replay_policy_sha256")
        != EXPECTED_POLICY_SHA256
        or metadata.get("replay_source_database_identity_sha256")
        != PARENT_DATABASE_IDENTITY_SHA256
        or metadata.get("instruction_source_database_identity_sha256")
        != EXPECTED_INSTRUCTION_SOURCE_IDENTITY
        or metadata.get("instruction_replay_target_train_weight_fraction") != "0.5"
        or metadata.get("instruction_replay_replay_mode") != "stratified"
        or metadata.get("instruction_replay_replay_per_cell") != str(REPLAY_PER_CELL)
        or multipliers != {f"train|python|{INSTRUCTION_FAMILY}": 42.0}
    ):
        raise ValueError("v10 instruction replay curriculum metadata is invalid")
    languages = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))
    expected = {
        ("train", language, family): REPLAY_PER_CELL
        for language in languages
        for family in TASK_FAMILIES
    }
    expected.update(
        {
            (split, language, family): 128
            for split in ("validation", "test")
            for language in languages
            for family in TASK_FAMILIES
        }
    )
    expected.update(
        {
            ("train", "python", INSTRUCTION_FAMILY): INSTRUCTION_TRAIN_EXAMPLES,
            ("validation", "python", INSTRUCTION_FAMILY): 256,
            ("test", "python", INSTRUCTION_FAMILY): 256,
        }
    )
    if observed != expected:
        raise ValueError("v10 instruction replay curriculum cell counts are invalid")
    if int(train_count) != TRAIN_EXAMPLES or int(maximum_sequence) > 2048:
        raise ValueError("v10 instruction replay curriculum bounds are invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--parent-database", type=Path, default=DEFAULT_PARENT_DATABASE)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT)
    parser.add_argument("--validation-database", type=Path, required=True)
    parser.add_argument("--test-database", type=Path, required=True)
    parser.add_argument("--heldout-suite-manifest", type=Path, required=True)
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    database = arguments.database.resolve()
    checkpoint = arguments.checkpoint.resolve()
    parent_database = arguments.parent_database.resolve()
    parent_checkpoint = arguments.parent_checkpoint.resolve()
    validation_database = arguments.validation_database.resolve()
    test_database = arguments.test_database.resolve()
    heldout_suite_manifest = arguments.heldout_suite_manifest.resolve()
    status_output = (
        arguments.status_output.resolve()
        if arguments.status_output is not None
        else checkpoint.with_name("training-progress.json")
    )
    _verify_database(database)
    if not parent_database.is_file() or not parent_checkpoint.is_file():
        raise ValueError("verified v8 parent artifacts are required")
    manifest = read_v10_heldout_manifest(heldout_suite_manifest)
    (
        evaluation_database_identities_sha256,
        heldout_suite_manifest_sha256,
    ) = verify_v10_evaluation_contract(
        task_database_identities_sha256=(
            v10_evaluation_identities_from_manifest(manifest)
        ),
        heldout_manifest=manifest,
    )
    resume = checkpoint.is_file()
    command = build_bounded_training_command(
        database=database,
        checkpoint=checkpoint,
        status_output=status_output,
        parent_database=parent_database,
        parent_checkpoint=parent_checkpoint,
        validation_database=validation_database,
        test_database=test_database,
        heldout_suite_manifest=heldout_suite_manifest,
        resume=resume,
    )
    contract = {
        "schema_version": "bosscrafts-v10-instruction-training-chunk-contract-v2",
        "database": str(database),
        "database_identity_sha256": EXPECTED_DATABASE_IDENTITY,
        "checkpoint": str(checkpoint),
        "parent_database": str(parent_database),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "evaluation_databases": {
            "validation": str(validation_database),
            "test": str(test_database),
        },
        "evaluation_database_identities_sha256": (
            evaluation_database_identities_sha256
        ),
        "heldout_suite_manifest": str(heldout_suite_manifest),
        "heldout_suite_manifest_sha256": heldout_suite_manifest_sha256,
        "expected_training_plan_sha256": expected_training_plan_sha256(
            evaluation_database_identities_sha256,
            heldout_suite_manifest_sha256,
        ),
        "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
        "train_examples": TRAIN_EXAMPLES,
        "instruction_train_examples": INSTRUCTION_TRAIN_EXAMPLES,
        "device": "cuda",
        "dtype": "bfloat16",
        "epochs": 1,
        "full_model": True,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "gradient_checkpointing": True,
        "learning_rate": LEARNING_RATE,
        "maximum_optimizer_steps_this_chunk": 100,
        "minimum_learning_rate_ratio": MINIMUM_LEARNING_RATE_RATIO,
        "resume": resume,
        "status_output": str(status_output),
        "warmup_ratio": WARMUP_RATIO,
        "command": command,
    }
    if arguments.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
