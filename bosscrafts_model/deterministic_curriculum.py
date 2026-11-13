"""Derive a balanced curriculum with globally deterministic, isolated prompts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3

from .bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from .compression import decompress_bounded
from .current_task_packing import FAMILIES
from .resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    _canonical_json,
    _decode_u16,
    audit_task_database,
)
from .tokenizer import NativeTokenizerBinding


DETERMINISTIC_CURRICULUM_SCHEMA = "bosscrafts-deterministic-curriculum-v1"
SPLITS = ("train", "validation", "test")
_SPLIT_BITS = {"train": 1, "validation": 2, "test": 4}


@dataclass(slots=True)
class _GroupState:
    first_target: tuple[str, int]
    other_targets: set[tuple[str, int]] | None
    split_bits: int
    row_count: int


@dataclass(frozen=True)
class DeterministicCurriculumStatus:
    database_path: Path
    source_database_identity_sha256: str
    database_identity_sha256: str
    curriculum_policy_sha256: str
    source_example_count: int
    selected_example_count: int
    total_sequence_tokens: int
    unique_source_prompt_count: int
    conflicting_prompt_group_count: int
    conflicting_prompt_row_count: int
    cross_split_prompt_group_count: int
    cross_split_prompt_row_count: int
    excluded_prompt_group_count: int
    excluded_prompt_row_count: int
    consistent_duplicate_rows_suppressed: int
    quota_surplus_rows_skipped: int
    cell_counts: dict[str, int]


def _validate_policy(
    languages: tuple[str, ...], quotas: dict[str, int]
) -> tuple[str, ...]:
    if (
        type(languages) is not tuple
        or not languages
        or any(type(language) is not str for language in languages)
        or len(set(languages)) != len(languages)
    ):
        raise ValueError("languages must be a nonempty unique tuple")
    unsupported = sorted(set(languages) - set(CURRENT_RUNTIME_SPEC_SHA256))
    if unsupported:
        raise ValueError(f"unsupported curriculum languages: {unsupported}")
    if (
        type(quotas) is not dict
        or set(quotas) != set(SPLITS)
        or any(type(value) is not int or not 1 <= value <= 100_000 for value in quotas.values())
    ):
        raise ValueError("quotas must define positive train, validation and test limits")
    return tuple(sorted(languages))


def _keys(
    family: str,
    compressed_ids: bytes,
    token_count: int,
    loss_start: int,
) -> tuple[tuple[str, str, int], tuple[str, int], bytes]:
    raw_ids = decompress_bounded(
        compressed_ids,
        maximum_output_bytes=token_count * 2,
    )
    input_ids = _decode_u16(raw_ids)
    if len(input_ids) != token_count:
        raise ValueError("task token count mismatch")
    if not 0 < loss_start < token_count:
        raise ValueError("task loss boundary is invalid")
    boundary = loss_start * 2
    prompt_sha256 = hashlib.sha256(raw_ids[:boundary]).hexdigest()
    target_sha256 = hashlib.sha256(raw_ids[boundary:]).hexdigest()
    return (
        (family, prompt_sha256, loss_start),
        (target_sha256, token_count - loss_start),
        raw_ids,
    )


def derive_deterministic_curriculum(
    binding: NativeTokenizerBinding,
    source_database: str | Path,
    destination_database: str | Path,
    *,
    languages: tuple[str, ...] = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256)),
    quotas: dict[str, int] | None = None,
) -> DeterministicCurriculumStatus:
    """Filter contradictions/leakage, globally deduplicate, then balance cells."""
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if quotas is None:
        quotas = {"train": 2048, "validation": 128, "test": 128}
    languages = _validate_policy(languages, quotas)
    source_path = Path(source_database).resolve()
    destination = Path(destination_database).resolve()
    if not source_path.is_file():
        raise ValueError("source task database does not exist")
    if destination.exists():
        raise ValueError("destination task database already exists")

    source_audit = audit_task_database(binding, source_path)
    source = sqlite3.connect(f"{source_path.as_uri()}?mode=ro", uri=True, timeout=30)
    source.execute("PRAGMA query_only=ON")
    source_metadata = dict(source.execute("SELECT key,value FROM metadata"))

    groups: dict[tuple[str, str, int], _GroupState] = {}
    try:
        rows = source.execute(
            """SELECT family,split,input_ids_zlib,token_count,loss_start_index
               FROM tasks ORDER BY example_id"""
        )
        for family_value, split_value, compressed, count_value, loss_value in rows:
            family = str(family_value)
            split = str(split_value)
            if family not in FAMILIES:
                raise ValueError(f"unsupported task family: {family}")
            if split not in _SPLIT_BITS:
                raise ValueError(f"unsupported task split: {split}")
            prompt_key, target_key, _ = _keys(
                family, compressed, int(count_value), int(loss_value)
            )
            state = groups.get(prompt_key)
            if state is None:
                groups[prompt_key] = _GroupState(
                    first_target=target_key,
                    other_targets=None,
                    split_bits=_SPLIT_BITS[split],
                    row_count=1,
                )
                continue
            state.row_count += 1
            state.split_bits |= _SPLIT_BITS[split]
            if target_key != state.first_target:
                if state.other_targets is None:
                    state.other_targets = set()
                state.other_targets.add(target_key)

        conflicting_keys = {
            key for key, state in groups.items() if state.other_targets
        }
        cross_split_keys = {
            key
            for key, state in groups.items()
            if state.split_bits not in _SPLIT_BITS.values()
        }
        excluded_keys = conflicting_keys | cross_split_keys
        conflicting_rows = sum(groups[key].row_count for key in conflicting_keys)
        cross_split_rows = sum(groups[key].row_count for key in cross_split_keys)
        excluded_rows = sum(groups[key].row_count for key in excluded_keys)

        policy = {
            "schema_version": DETERMINISTIC_CURRICULUM_SCHEMA,
            "source_task_database_identity_sha256": source_audit.database_identity_sha256,
            "languages": languages,
            "task_families": FAMILIES,
            "quotas_per_language_family": dict(sorted(quotas.items())),
            "prompt_identity": "exact model-visible uint16 tokens within task family",
            "conflict_policy": "exclude every row in any prompt group with multiple targets",
            "heldout_policy": "exclude every row in any prompt group spanning splits",
            "duplicate_policy": "keep at most one row per remaining global prompt",
            "selection": "ascending example_id subject to exact cell quotas",
        }
        policy_sha256 = hashlib.sha256(
            _canonical_json(policy).encode("utf-8")
        ).hexdigest()
        expected_cells = {
            (split, language, family): quotas[split]
            for split in SPLITS
            for language in languages
            for family in FAMILIES
        }

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        if temporary.exists():
            temporary.unlink()
        output = sqlite3.connect(temporary)
        output.execute("PRAGMA journal_mode=DELETE")
        output.execute("PRAGMA synchronous=FULL")
        cell_counts: Counter[tuple[str, str, str]] = Counter()
        selected_prompts: set[tuple[str, str, int]] = set()
        duplicate_rows_suppressed = 0
        quota_surplus_rows_skipped = 0
        total_sequence_tokens = 0
        try:
            output.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            output.execute(
                """CREATE TABLE tasks (
                       example_id TEXT PRIMARY KEY,
                       document_id TEXT NOT NULL,
                       language TEXT NOT NULL,
                       family TEXT NOT NULL,
                       record_index INTEGER NOT NULL,
                       split TEXT NOT NULL,
                       input_ids_zlib BLOB NOT NULL,
                       token_count INTEGER NOT NULL,
                       loss_start_index INTEGER NOT NULL,
                       semantic_spec_sha256 TEXT NOT NULL,
                       parser_runtime_identity_sha256 TEXT NOT NULL,
                       source_sha256 TEXT NOT NULL
                   )"""
            )
            output.execute("CREATE INDEX tasks_cell ON tasks(split,language,family)")
            metadata = {
                key: value
                for key, value in source_metadata.items()
                if key
                in {
                    "binding_sha256",
                    "tokenizer_sha256",
                    "vocabulary_size",
                    "maximum_sequence_tokens",
                    "semantic_spec_sha256",
                    "parser_runtime_identity_sha256",
                    "source_catalog_sha256",
                    "source_admission_identity_sha256",
                    "review_manifest_sha256",
                }
            }
            metadata.update(
                {
                    "schema_version": TASK_DATABASE_SCHEMA,
                    "packing_complete": "true",
                    "curriculum_schema_version": DETERMINISTIC_CURRICULUM_SCHEMA,
                    "curriculum_policy_sha256": policy_sha256,
                    "source_task_database_identity_sha256": source_audit.database_identity_sha256,
                }
            )
            output.executemany(
                "INSERT INTO metadata(key,value) VALUES(?,?)", sorted(metadata.items())
            )

            rows = source.execute(
                """SELECT example_id,document_id,language,family,record_index,split,
                          input_ids_zlib,token_count,loss_start_index,
                           semantic_spec_sha256,parser_runtime_identity_sha256,
                           source_sha256
                   FROM tasks ORDER BY example_id"""
            )
            for row in rows:
                language = str(row[2])
                family = str(row[3])
                split = str(row[5])
                if language not in languages:
                    continue
                prompt_key, _, _ = _keys(
                    family, row[6], int(row[7]), int(row[8])
                )
                if prompt_key in excluded_keys:
                    continue
                if prompt_key in selected_prompts:
                    duplicate_rows_suppressed += 1
                    continue
                cell = (split, language, family)
                target = expected_cells.get(cell)
                if target is None or cell_counts[cell] >= target:
                    quota_surplus_rows_skipped += 1
                    continue
                output.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", row)
                selected_prompts.add(prompt_key)
                cell_counts[cell] += 1
                total_sequence_tokens += int(row[7])

            missing = {
                "|".join(cell): target - cell_counts[cell]
                for cell, target in expected_cells.items()
                if cell_counts[cell] != target
            }
            if missing:
                raise ValueError(
                    f"deterministic curriculum quotas could not be filled: {missing}"
                )
            output.commit()
        except BaseException:
            output.close()
            if temporary.exists():
                temporary.unlink()
            raise
        else:
            output.close()
            temporary.replace(destination)
    finally:
        source.close()

    output_audit = audit_task_database(binding, destination)
    rendered_cells = {
        "|".join(cell): count for cell, count in sorted(cell_counts.items())
    }
    return DeterministicCurriculumStatus(
        database_path=destination,
        source_database_identity_sha256=source_audit.database_identity_sha256,
        database_identity_sha256=output_audit.database_identity_sha256,
        curriculum_policy_sha256=policy_sha256,
        source_example_count=source_audit.example_count,
        selected_example_count=output_audit.example_count,
        total_sequence_tokens=total_sequence_tokens,
        unique_source_prompt_count=len(groups),
        conflicting_prompt_group_count=len(conflicting_keys),
        conflicting_prompt_row_count=conflicting_rows,
        cross_split_prompt_group_count=len(cross_split_keys),
        cross_split_prompt_row_count=cross_split_rows,
        excluded_prompt_group_count=len(excluded_keys),
        excluded_prompt_row_count=excluded_rows,
        consistent_duplicate_rows_suppressed=duplicate_rows_suppressed,
        quota_surplus_rows_skipped=quota_surplus_rows_skipped,
        cell_counts=rendered_cells,
    )
