"""Traceable, weighted merge of semantic and reviewed instruction curricula."""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from .current_task_packing import FAMILIES
from .resumable_task_packing import (
    CELL_WEIGHT_MULTIPLIERS_METADATA_KEY,
    SALTED_SOURCE_EXAMPLE_ID_POLICY,
    TASK_DATABASE_SCHEMA,
    ResumableTaskDataset,
    _canonical_json,
    audit_task_database,
    salted_source_example_id,
)
from .reviewed_instruction_curriculum import (
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_RESPONSE_FAMILY,
)
from .task_database_audit_manifest import (
    load_verified_task_database_audit_manifest,
    publish_task_database_audit_manifest_from_result,
)
from .tokenizer import NativeTokenizerBinding

MIXED_CURRICULUM_SCHEMA = "bosscrafts-semantic-reviewed-instruction-mix-v1"
SPLIT_SAFE_MIXED_TRAINING_SCHEMA = (
    "bosscrafts-split-safe-semantic-reviewed-instruction-train-mix-v1"
)
_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class MixedCurriculumResult:
    database_path: Path
    database_identity_sha256: str
    content_sha256: str
    policy_sha256: str
    example_count: int
    semantic_example_count: int
    instruction_example_count: int
    split_counts: dict[str, int]
    family_counts: dict[str, int]
    instruction_training_weight_fraction: float


@dataclass(frozen=True, slots=True)
class SplitSafeMixedTrainingResult:
    database_path: Path
    database_identity_sha256: str
    content_sha256: str
    policy_sha256: str
    example_count: int
    semantic_example_count: int
    instruction_train_example_count: int
    excluded_instruction_validation_example_count: int
    excluded_instruction_test_example_count: int
    instruction_training_weight_fraction: float
    audit_manifest_sha256: str | None = None


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ro_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _metadata(path: Path) -> dict[str, str]:
    connection = _ro_connect(path)
    try:
        return dict(connection.execute("SELECT key,value FROM metadata"))
    finally:
        connection.close()


def _cell_counts(path: Path) -> dict[tuple[str, str, str], int]:
    connection = _ro_connect(path)
    try:
        return {
            (str(split), str(language), str(family)): int(count)
            for split, language, family, count in connection.execute(
                "SELECT split,language,family,COUNT(*) FROM tasks "
                "GROUP BY split,language,family"
            )
        }
    finally:
        connection.close()


def _source_identity_map_sha256(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    count = 0
    for example_id, source_example_id, source_kind in connection.execute(
        "SELECT example_id,source_example_id,source_kind "
        "FROM source_example_identity ORDER BY example_id"
    ):
        digest.update(
            _canonical_json(
                {
                    "example_id": str(example_id),
                    "source_example_id": str(source_example_id),
                    "source_kind": str(source_kind),
                }
            ).encode("utf-8")
        )
        count += 1
    if count < 1:
        raise ValueError("mixed curriculum source identity map is empty")
    return digest.hexdigest()


def _validate_sources(
    semantic_cells: dict[tuple[str, str, str], int],
    instruction_cells: dict[tuple[str, str, str], int],
) -> int:
    semantic_train_cells = {
        (language, family)
        for split, language, family in semantic_cells
        if split == "train"
    }
    if not semantic_train_cells:
        raise ValueError("semantic curriculum has no training cells")
    if any(
        split not in _SPLITS
        or family not in FAMILIES
        or count < 1
        or (language, family) not in semantic_train_cells
        for (split, language, family), count in semantic_cells.items()
    ):
        raise ValueError("semantic curriculum cells are invalid")
    for split in _SPLITS:
        if {
            (language, family)
            for current, language, family in semantic_cells
            if current == split
        } != semantic_train_cells:
            raise ValueError("semantic curriculum cells differ across splits")
    expected_instruction = {
        (split, INSTRUCTION_LANGUAGE, INSTRUCTION_RESPONSE_FAMILY)
        for split in _SPLITS
    }
    if set(instruction_cells) != expected_instruction or any(
        count < 1 for count in instruction_cells.values()
    ):
        raise ValueError("reviewed instruction curriculum cells are invalid")
    return len(semantic_train_cells)


def _validate_split_safe_sources(
    semantic_cells: dict[tuple[str, str, str], int],
    instruction_cells: dict[tuple[str, str, str], int],
) -> tuple[int, int, int, int]:
    expected_semantic = {
        ("train", language, family)
        for language in CURRENT_RUNTIME_SPEC_SHA256
        for family in FAMILIES
    }
    if set(semantic_cells) != expected_semantic or any(
        count < 1 for count in semantic_cells.values()
    ):
        raise ValueError(
            "split-safe semantic curriculum must contain only complete train cells"
        )
    expected_instruction = {
        (split, INSTRUCTION_LANGUAGE, INSTRUCTION_RESPONSE_FAMILY)
        for split in _SPLITS
    }
    if set(instruction_cells) != expected_instruction or any(
        count < 1 for count in instruction_cells.values()
    ):
        raise ValueError(
            "split-safe reviewed instruction curriculum cells are invalid"
        )
    return (
        len(expected_semantic),
        instruction_cells[("train", INSTRUCTION_LANGUAGE, INSTRUCTION_RESPONSE_FAMILY)],
        instruction_cells[
            ("validation", INSTRUCTION_LANGUAGE, INSTRUCTION_RESPONSE_FAMILY)
        ],
        instruction_cells[("test", INSTRUCTION_LANGUAGE, INSTRUCTION_RESPONSE_FAMILY)],
    )


def build_mixed_curriculum(
    binding: NativeTokenizerBinding,
    *,
    semantic_database: str | Path,
    instruction_database: str | Path,
    output_database: str | Path,
    expected_semantic_identity_sha256: str,
    expected_instruction_identity_sha256: str,
    instruction_training_weight_fraction: float = 0.5,
) -> MixedCurriculumResult:
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if not _is_sha256(expected_semantic_identity_sha256) or not _is_sha256(
        expected_instruction_identity_sha256
    ):
        raise ValueError("mixed curriculum expected source identity is invalid")
    if (
        type(instruction_training_weight_fraction) is not float
        or not math.isfinite(instruction_training_weight_fraction)
        or not 0.05 <= instruction_training_weight_fraction <= 0.95
    ):
        raise ValueError("instruction training weight fraction must be 0.05 through 0.95")
    semantic_path = Path(semantic_database).resolve()
    instruction_path = Path(instruction_database).resolve()
    destination = Path(output_database).resolve()
    if destination.exists():
        raise ValueError("mixed curriculum output already exists")
    if not semantic_path.is_file() or not instruction_path.is_file():
        raise ValueError("mixed curriculum source database does not exist")
    semantic_audit = audit_task_database(binding, semantic_path)
    instruction_audit = audit_task_database(binding, instruction_path)
    if (
        semantic_audit.database_identity_sha256
        != expected_semantic_identity_sha256
        or instruction_audit.database_identity_sha256
        != expected_instruction_identity_sha256
    ):
        raise ValueError("mixed curriculum source identity mismatch")
    semantic_metadata = _metadata(semantic_path)
    instruction_metadata = _metadata(instruction_path)
    if (
        semantic_metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or instruction_metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or semantic_metadata.get("packing_complete") != "true"
        or instruction_metadata.get("packing_complete") != "true"
        or semantic_metadata.get("binding_sha256") != binding.binding_sha256
        or instruction_metadata.get("binding_sha256") != binding.binding_sha256
    ):
        raise ValueError("mixed curriculum source metadata is invalid")
    semantic_cells = _cell_counts(semantic_path)
    instruction_cells = _cell_counts(instruction_path)
    semantic_train_cell_count = _validate_sources(
        semantic_cells, instruction_cells
    )
    multiplier = (
        semantic_train_cell_count
        * instruction_training_weight_fraction
        / (1.0 - instruction_training_weight_fraction)
    )
    multiplier_key = (
        f"train|{INSTRUCTION_LANGUAGE}|{INSTRUCTION_RESPONSE_FAMILY}"
    )
    multipliers = {multiplier_key: multiplier}
    policy = {
        "schema_version": MIXED_CURRICULUM_SCHEMA,
        "semantic_source_database_identity_sha256": (
            semantic_audit.database_identity_sha256
        ),
        "instruction_source_database_identity_sha256": (
            instruction_audit.database_identity_sha256
        ),
        "row_policy": "copy every source row without token modification",
        "example_id_policy": SALTED_SOURCE_EXAMPLE_ID_POLICY,
        "heldout_policy": "preserve source validation and test rows and document splits",
        "instruction_training_weight_fraction": instruction_training_weight_fraction,
        "semantic_training_cell_count": semantic_train_cell_count,
        "training_cell_weight_multipliers": multipliers,
    }
    policy_sha256 = hashlib.sha256(
        _canonical_json(policy).encode("utf-8")
    ).hexdigest()
    output_metadata = dict(semantic_metadata)
    output_metadata.update(
        {
            "curriculum_schema_version": MIXED_CURRICULUM_SCHEMA,
            "curriculum_policy_sha256": policy_sha256,
            "semantic_source_database_identity_sha256": (
                semantic_audit.database_identity_sha256
            ),
            "instruction_source_database_identity_sha256": (
                instruction_audit.database_identity_sha256
            ),
            "instruction_training_weight_fraction": repr(
                instruction_training_weight_fraction
            ),
            CELL_WEIGHT_MULTIPLIERS_METADATA_KEY: _canonical_json(multipliers),
            "training_example_id_policy": SALTED_SOURCE_EXAMPLE_ID_POLICY,
            "packing_complete": "true",
        }
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        output = sqlite3.connect(temporary)
        try:
            output.create_function(
                "salt_source_id", 2, salted_source_example_id, deterministic=True
            )
            output.execute("PRAGMA journal_mode=DELETE")
            output.execute("PRAGMA synchronous=FULL")
            output.execute(
                "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
            output.execute(
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
            output.execute(
                """CREATE TABLE source_example_identity(
                       example_id TEXT PRIMARY KEY,source_example_id TEXT NOT NULL,
                       source_kind TEXT NOT NULL CHECK(source_kind IN ('replay','instruction')),
                       UNIQUE(source_kind,source_example_id))"""
            )
            output.execute("ATTACH DATABASE ? AS semantic", (str(semantic_path),))
            output.execute(
                "ATTACH DATABASE ? AS instruction", (str(instruction_path),)
            )
            for schema, kind in (("semantic", "replay"), ("instruction", "instruction")):
                output.execute(
                    f"""INSERT INTO tasks
                        SELECT salt_source_id(?,example_id),document_id,language,family,
                               record_index,split,input_ids_zlib,token_count,
                               loss_start_index,semantic_spec_sha256,
                               parser_runtime_identity_sha256,source_sha256
                          FROM {schema}.tasks""",
                    (kind,),
                )
                output.execute(
                    f"""INSERT INTO source_example_identity
                        SELECT salt_source_id(?,example_id),example_id,?
                          FROM {schema}.tasks""",
                    (kind, kind),
                )
            output.execute("CREATE INDEX tasks_cell ON tasks(split,language,family)")
            map_sha256 = _source_identity_map_sha256(output)
            output_metadata["source_example_identity_map_sha256"] = map_sha256
            output.executemany(
                "INSERT INTO metadata VALUES(?,?)", sorted(output_metadata.items())
            )
            output.commit()
            output.execute("VACUUM")
        finally:
            output.close()
        mixed_audit = audit_task_database(binding, temporary)
        expected_examples = semantic_audit.example_count + instruction_audit.example_count
        if mixed_audit.example_count != expected_examples:
            raise ValueError("mixed curriculum example count mismatch")
        dataset = ResumableTaskDataset(binding, temporary)
        training_weight = {"instruction": 0.0, "semantic": 0.0}
        for row in dataset.iter_examples(split="train"):
            kind = (
                "instruction"
                if row["family"] == INSTRUCTION_RESPONSE_FAMILY
                else "semantic"
            )
            training_weight[kind] += float(row["sampling_weight"])
        total_weight = training_weight["instruction"] + training_weight["semantic"]
        observed_fraction = training_weight["instruction"] / total_weight
        if not math.isclose(
            observed_fraction,
            instruction_training_weight_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("mixed curriculum training weight fraction mismatch")
        final_semantic = audit_task_database(binding, semantic_path)
        final_instruction = audit_task_database(binding, instruction_path)
        if (
            final_semantic.database_identity_sha256
            != semantic_audit.database_identity_sha256
            or final_instruction.database_identity_sha256
            != instruction_audit.database_identity_sha256
        ):
            raise ValueError("mixed curriculum source changed during build")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if destination.exists():
            raise ValueError("mixed curriculum output already exists")
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError("mixed curriculum output already exists") from error
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.unlink()
    return MixedCurriculumResult(
        database_path=destination,
        database_identity_sha256=mixed_audit.database_identity_sha256,
        content_sha256=mixed_audit.content_sha256,
        policy_sha256=policy_sha256,
        example_count=mixed_audit.example_count,
        semantic_example_count=semantic_audit.example_count,
        instruction_example_count=instruction_audit.example_count,
        split_counts=mixed_audit.split_counts,
        family_counts=mixed_audit.family_counts,
        instruction_training_weight_fraction=observed_fraction,
    )


def build_split_safe_mixed_training_curriculum(
    binding: NativeTokenizerBinding,
    *,
    semantic_train_database: str | Path,
    instruction_database: str | Path,
    output_database: str | Path,
    expected_semantic_train_identity_sha256: str,
    expected_instruction_identity_sha256: str,
    instruction_training_weight_fraction: float = 0.5,
    semantic_train_audit_manifest: str | Path | None = None,
    instruction_audit_manifest: str | Path | None = None,
    output_audit_manifest: str | Path | None = None,
) -> SplitSafeMixedTrainingResult:
    """Mix only training rows while retaining held-out source lineage."""
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if not _is_sha256(expected_semantic_train_identity_sha256) or not _is_sha256(
        expected_instruction_identity_sha256
    ):
        raise ValueError("split-safe mixed curriculum expected identity is invalid")
    if (
        type(instruction_training_weight_fraction) is not float
        or not math.isfinite(instruction_training_weight_fraction)
        or not 0.05 <= instruction_training_weight_fraction <= 0.95
    ):
        raise ValueError("instruction training weight fraction must be 0.05 through 0.95")
    semantic_path = Path(semantic_train_database).resolve()
    instruction_path = Path(instruction_database).resolve()
    destination = Path(output_database).resolve()
    audit_manifest_destination = (
        None
        if output_audit_manifest is None
        else Path(output_audit_manifest).resolve()
    )
    if destination.exists():
        raise ValueError("split-safe mixed curriculum output already exists")
    if audit_manifest_destination == destination:
        raise ValueError("split-safe mixed audit manifest must differ from its database")
    if audit_manifest_destination is not None and audit_manifest_destination.exists():
        raise ValueError("split-safe mixed audit manifest output already exists")
    if not semantic_path.is_file() or not instruction_path.is_file():
        raise ValueError("split-safe mixed curriculum source database does not exist")
    if (semantic_train_audit_manifest is None) != (
        instruction_audit_manifest is None
    ):
        raise ValueError("split-safe mixed curriculum requires both audit manifests")

    if semantic_train_audit_manifest is None:
        semantic_audit = audit_task_database(binding, semantic_path)
        instruction_audit = audit_task_database(binding, instruction_path)
    else:
        assert instruction_audit_manifest is not None
        semantic_audit = load_verified_task_database_audit_manifest(
            binding,
            semantic_path,
            semantic_train_audit_manifest,
            expected_database_identity_sha256=(
                expected_semantic_train_identity_sha256
            ),
        ).audit
        instruction_audit = load_verified_task_database_audit_manifest(
            binding,
            instruction_path,
            instruction_audit_manifest,
            expected_database_identity_sha256=expected_instruction_identity_sha256,
        ).audit
    if (
        semantic_audit.database_identity_sha256
        != expected_semantic_train_identity_sha256
        or instruction_audit.database_identity_sha256
        != expected_instruction_identity_sha256
    ):
        raise ValueError("split-safe mixed curriculum source identity mismatch")
    if set(semantic_audit.split_counts) != {"train"} or set(
        instruction_audit.split_counts
    ) != set(_SPLITS):
        raise ValueError("split-safe mixed curriculum source split contract is invalid")

    semantic_metadata = _metadata(semantic_path)
    instruction_metadata = _metadata(instruction_path)
    if (
        semantic_metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or instruction_metadata.get("schema_version") != TASK_DATABASE_SCHEMA
        or semantic_metadata.get("packing_complete") != "true"
        or instruction_metadata.get("packing_complete") != "true"
        or semantic_metadata.get("binding_sha256") != binding.binding_sha256
        or instruction_metadata.get("binding_sha256") != binding.binding_sha256
    ):
        raise ValueError("split-safe mixed curriculum source metadata is invalid")
    try:
        maximum_sequence = max(
            int(semantic_metadata["maximum_sequence_tokens"]),
            int(instruction_metadata["maximum_sequence_tokens"]),
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            "split-safe mixed curriculum maximum sequence metadata is invalid"
        ) from error

    semantic_cells = _cell_counts(semantic_path)
    instruction_cells = _cell_counts(instruction_path)
    (
        semantic_train_cell_count,
        instruction_train_count,
        excluded_instruction_validation_count,
        excluded_instruction_test_count,
    ) = _validate_split_safe_sources(semantic_cells, instruction_cells)
    multiplier = (
        semantic_train_cell_count
        * instruction_training_weight_fraction
        / (1.0 - instruction_training_weight_fraction)
    )
    multiplier_key = (
        f"train|{INSTRUCTION_LANGUAGE}|{INSTRUCTION_RESPONSE_FAMILY}"
    )
    multipliers = {multiplier_key: multiplier}
    policy = {
        "schema_version": SPLIT_SAFE_MIXED_TRAINING_SCHEMA,
        "semantic_train_source_database_identity_sha256": (
            semantic_audit.database_identity_sha256
        ),
        "instruction_source_database_identity_sha256": (
            instruction_audit.database_identity_sha256
        ),
        "row_policy": (
            "copy every semantic train row and only reviewed instruction train rows "
            "without token modification"
        ),
        "example_id_policy": SALTED_SOURCE_EXAMPLE_ID_POLICY,
        "heldout_policy": (
            "exclude semantic validation/test databases and reviewed instruction "
            "validation/test rows from the training artifact"
        ),
        "excluded_instruction_validation_example_count": (
            excluded_instruction_validation_count
        ),
        "excluded_instruction_test_example_count": excluded_instruction_test_count,
        "instruction_training_weight_fraction": instruction_training_weight_fraction,
        "semantic_training_cell_count": semantic_train_cell_count,
        "training_cell_weight_multipliers": multipliers,
    }
    policy_sha256 = hashlib.sha256(
        _canonical_json(policy).encode("utf-8")
    ).hexdigest()
    output_metadata = dict(semantic_metadata)
    output_metadata.update(
        {
            "curriculum_schema_version": SPLIT_SAFE_MIXED_TRAINING_SCHEMA,
            "curriculum_policy_sha256": policy_sha256,
            "curriculum_source_split": "train",
            "semantic_train_source_database_identity_sha256": (
                semantic_audit.database_identity_sha256
            ),
            "instruction_source_database_identity_sha256": (
                instruction_audit.database_identity_sha256
            ),
            "instruction_source_train_example_count": str(instruction_train_count),
            "excluded_instruction_validation_example_count": str(
                excluded_instruction_validation_count
            ),
            "excluded_instruction_test_example_count": str(
                excluded_instruction_test_count
            ),
            "instruction_training_weight_fraction": repr(
                instruction_training_weight_fraction
            ),
            CELL_WEIGHT_MULTIPLIERS_METADATA_KEY: _canonical_json(multipliers),
            "training_example_id_policy": SALTED_SOURCE_EXAMPLE_ID_POLICY,
            "maximum_sequence_tokens": str(maximum_sequence),
            "packing_complete": "true",
        }
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        output = sqlite3.connect(temporary)
        try:
            output.create_function(
                "salt_source_id", 2, salted_source_example_id, deterministic=True
            )
            output.execute("PRAGMA journal_mode=DELETE")
            output.execute("PRAGMA synchronous=FULL")
            output.execute(
                "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
            output.execute(
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
            output.execute(
                """CREATE TABLE source_example_identity(
                       example_id TEXT PRIMARY KEY,source_example_id TEXT NOT NULL,
                       source_kind TEXT NOT NULL CHECK(source_kind IN ('replay','instruction')),
                       UNIQUE(source_kind,source_example_id))"""
            )
            output.execute("ATTACH DATABASE ? AS semantic", (str(semantic_path),))
            output.execute(
                "ATTACH DATABASE ? AS instruction", (str(instruction_path),)
            )
            for schema, kind in (("semantic", "replay"), ("instruction", "instruction")):
                output.execute(
                    f"""INSERT INTO tasks
                        SELECT salt_source_id(?,example_id),document_id,language,family,
                               record_index,split,input_ids_zlib,token_count,
                               loss_start_index,semantic_spec_sha256,
                               parser_runtime_identity_sha256,source_sha256
                          FROM {schema}.tasks WHERE split='train'""",
                    (kind,),
                )
                output.execute(
                    f"""INSERT INTO source_example_identity
                        SELECT salt_source_id(?,example_id),example_id,?
                          FROM {schema}.tasks WHERE split='train'""",
                    (kind, kind),
                )
            output.execute("CREATE INDEX tasks_cell ON tasks(split,language,family)")
            map_sha256 = _source_identity_map_sha256(output)
            output_metadata["source_example_identity_map_sha256"] = map_sha256
            output.executemany(
                "INSERT INTO metadata VALUES(?,?)", sorted(output_metadata.items())
            )
            output.commit()
        finally:
            output.close()

        mixed_audit = audit_task_database(binding, temporary)
        expected_examples = semantic_audit.example_count + instruction_train_count
        if (
            mixed_audit.example_count != expected_examples
            or mixed_audit.split_counts != {"train": expected_examples}
        ):
            raise ValueError("split-safe mixed curriculum example count mismatch")
        dataset = ResumableTaskDataset(binding, temporary)
        cell_weight_totals = dataset.cell_sampling_weight_totals(split="train")
        instruction_weight = sum(
            weight
            for (_, _, family), weight in cell_weight_totals.items()
            if family == INSTRUCTION_RESPONSE_FAMILY
        )
        semantic_weight = sum(cell_weight_totals.values()) - instruction_weight
        total_weight = instruction_weight + semantic_weight
        observed_fraction = instruction_weight / total_weight
        if not math.isclose(
            observed_fraction,
            instruction_training_weight_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "split-safe mixed curriculum training weight fraction mismatch"
            )
        if semantic_train_audit_manifest is None:
            final_semantic = audit_task_database(binding, semantic_path)
            final_instruction = audit_task_database(binding, instruction_path)
        else:
            assert instruction_audit_manifest is not None
            final_semantic = load_verified_task_database_audit_manifest(
                binding,
                semantic_path,
                semantic_train_audit_manifest,
                expected_database_identity_sha256=(
                    expected_semantic_train_identity_sha256
                ),
            ).audit
            final_instruction = load_verified_task_database_audit_manifest(
                binding,
                instruction_path,
                instruction_audit_manifest,
                expected_database_identity_sha256=(
                    expected_instruction_identity_sha256
                ),
            ).audit
        if (
            final_semantic.database_identity_sha256
            != semantic_audit.database_identity_sha256
            or final_instruction.database_identity_sha256
            != instruction_audit.database_identity_sha256
        ):
            raise ValueError("split-safe mixed curriculum source changed during build")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if destination.exists():
            raise ValueError("split-safe mixed curriculum output already exists")
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError(
                "split-safe mixed curriculum output already exists"
            ) from error
        try:
            audit_manifest_sha256 = (
                None
                if audit_manifest_destination is None
                else publish_task_database_audit_manifest_from_result(
                    binding,
                    mixed_audit,
                    destination,
                    audit_manifest_destination,
                ).manifest_sha256
            )
        except BaseException:
            try:
                if destination.exists() and destination.samefile(temporary):
                    destination.unlink()
            except OSError:
                pass
            raise
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.unlink()
    return SplitSafeMixedTrainingResult(
        database_path=destination,
        database_identity_sha256=mixed_audit.database_identity_sha256,
        content_sha256=mixed_audit.content_sha256,
        policy_sha256=policy_sha256,
        example_count=mixed_audit.example_count,
        semantic_example_count=semantic_audit.example_count,
        instruction_train_example_count=instruction_train_count,
        excluded_instruction_validation_example_count=(
            excluded_instruction_validation_count
        ),
        excluded_instruction_test_example_count=excluded_instruction_test_count,
        instruction_training_weight_fraction=observed_fraction,
        audit_manifest_sha256=audit_manifest_sha256,
    )
