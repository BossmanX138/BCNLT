"""Global prompt-to-target determinism audit for packed BossCrafts tasks."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3

from .current_task_packing import FAMILIES
from .compression import decompress_bounded
from .resumable_task_packing import _decode_u16, audit_task_database
from .tokenizer import NativeTokenizerBinding


AUDIT_SCHEMA = "bosscrafts-task-prompt-determinism-audit-v1"
_SPLIT_BITS = {"train": 1, "validation": 2, "test": 4}


@dataclass(slots=True)
class _PromptGroup:
    family: str
    prompt_sha256: str
    prompt_token_count: int
    row_count: int
    first_target: tuple[str, int]
    additional_targets: set[tuple[str, int]] | None
    split_bits: int
    first_language: str
    crosses_languages: bool
    first_representative: tuple[str, str, str, str, str]
    conflict_representatives: list[tuple[str, str, str, str, str]] | None
    prompt_token_ids: list[int] | None


def _representative(row: tuple[object, ...], target_sha256: str) -> tuple[str, str, str, str, str]:
    return (
        str(row[0]),
        str(row[1]),
        str(row[2]),
        str(row[4]),
        target_sha256,
    )


def _render_representative(
    representative: tuple[str, str, str, str, str]
) -> dict[str, str]:
    return {
        "example_id": representative[0],
        "document_id": representative[1],
        "language": representative[2],
        "split": representative[3],
        "target_sha256": representative[4],
    }


def audit_task_prompt_determinism(
    binding: NativeTokenizerBinding,
    database_path: str | Path,
    *,
    sample_limit: int = 20,
) -> dict[str, object]:
    """Prove that each model-visible prompt has one target within each family.

    Grouping is global rather than document-local because document IDs, source paths,
    and language labels are not model inputs. Consistent duplicates are reported too,
    especially when they cross splits and could leak held-out examples.
    """
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if type(sample_limit) is not int or not 1 <= sample_limit <= 100:
        raise ValueError("sample_limit must be an integer between 1 and 100")

    database = Path(database_path).resolve()
    database_audit = audit_task_database(binding, database)
    groups: dict[tuple[str, str, int], _PromptGroup] = {}
    family_examples: Counter[str] = Counter()

    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    try:
        rows = connection.execute(
            """SELECT example_id,document_id,language,family,split,
                      input_ids_zlib,token_count,loss_start_index
               FROM tasks ORDER BY example_id"""
        )
        for row in rows:
            family = str(row[3])
            if family not in FAMILIES:
                raise ValueError(f"unsupported task family: {family}")
            split = str(row[4])
            if split not in _SPLIT_BITS:
                raise ValueError(f"unsupported task split: {split}")
            token_count = int(row[6])
            raw_ids = decompress_bounded(
                row[5],
                maximum_output_bytes=token_count * 2,
            )
            input_ids = _decode_u16(raw_ids)
            if len(input_ids) != token_count:
                raise ValueError("task token count mismatch")
            loss_start = int(row[7])
            if not 0 < loss_start < len(input_ids):
                raise ValueError("task loss boundary is invalid")

            boundary = loss_start * 2
            prompt_bytes = raw_ids[:boundary]
            target_bytes = raw_ids[boundary:]
            prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
            target_sha256 = hashlib.sha256(target_bytes).hexdigest()
            prompt_key = (family, prompt_sha256, loss_start)
            target_key = (target_sha256, len(input_ids) - loss_start)
            representative = _representative(row, target_sha256)
            family_examples[family] += 1

            group = groups.get(prompt_key)
            if group is None:
                groups[prompt_key] = _PromptGroup(
                    family=family,
                    prompt_sha256=prompt_sha256,
                    prompt_token_count=loss_start,
                    row_count=1,
                    first_target=target_key,
                    additional_targets=None,
                    split_bits=_SPLIT_BITS[split],
                    first_language=str(row[2]),
                    crosses_languages=False,
                    first_representative=representative,
                    conflict_representatives=None,
                    prompt_token_ids=None,
                )
                continue

            group.row_count += 1
            group.split_bits |= _SPLIT_BITS[split]
            group.crosses_languages = (
                group.crosses_languages or str(row[2]) != group.first_language
            )
            if target_key != group.first_target:
                if group.additional_targets is None:
                    group.additional_targets = set()
                    group.conflict_representatives = [group.first_representative]
                    group.prompt_token_ids = input_ids[:loss_start]
                group.additional_targets.add(target_key)
                if group.conflict_representatives is None:
                    raise RuntimeError("conflicting prompt representatives were not initialized")
                if all(item[4] != target_sha256 for item in group.conflict_representatives):
                    group.conflict_representatives.append(representative)
    finally:
        connection.close()

    family_counters: dict[str, Counter[str]] = {
        family: Counter() for family in FAMILIES
    }
    global_counts: Counter[str] = Counter()
    sample_groups: list[_PromptGroup] = []
    for group in groups.values():
        counters = (global_counts, family_counters[group.family])
        for counter in counters:
            counter["unique_prompt_count"] += 1
            counter["maximum_rows_per_prompt"] = max(
                counter["maximum_rows_per_prompt"], group.row_count
            )
            if group.row_count > 1:
                counter["duplicate_prompt_group_count"] += 1
            if group.split_bits not in _SPLIT_BITS.values():
                counter["cross_split_prompt_group_count"] += 1
            if group.crosses_languages:
                counter["cross_language_prompt_group_count"] += 1
            if group.additional_targets:
                counter["conflicting_prompt_group_count"] += 1
                counter["rows_in_conflicting_prompt_groups"] += group.row_count
            elif group.row_count > 1:
                counter["consistent_duplicate_prompt_group_count"] += 1
        if group.additional_targets:
            sample_groups.append(group)

    global_counts["example_count"] = database_audit.example_count
    for family in FAMILIES:
        family_counters[family]["example_count"] = family_examples[family]

    count_fields = (
        "example_count",
        "unique_prompt_count",
        "duplicate_prompt_group_count",
        "consistent_duplicate_prompt_group_count",
        "conflicting_prompt_group_count",
        "rows_in_conflicting_prompt_groups",
        "cross_split_prompt_group_count",
        "cross_language_prompt_group_count",
        "maximum_rows_per_prompt",
    )

    def render_counts(counter: Counter[str]) -> dict[str, int]:
        return {name: int(counter[name]) for name in count_fields}

    samples: list[dict[str, object]] = []
    for group in sorted(
        sample_groups, key=lambda item: (item.family, item.prompt_sha256)
    )[:sample_limit]:
        if (
            group.additional_targets is None
            or group.conflict_representatives is None
            or group.prompt_token_ids is None
        ):
            raise RuntimeError("conflicting prompt evidence is incomplete")
        samples.append(
            {
                "family": group.family,
                "prompt_sha256": group.prompt_sha256,
                "prompt_token_count": group.prompt_token_count,
                "prompt_text": binding.decode(group.prompt_token_ids),
                "row_count": group.row_count,
                "distinct_target_count": 1 + len(group.additional_targets),
                "representatives": [
                    _render_representative(item)
                    for item in group.conflict_representatives
                ],
            }
        )

    summary = render_counts(global_counts)
    return {
        "schema_version": AUDIT_SCHEMA,
        "database": str(database),
        "task_database_identity_sha256": database_audit.database_identity_sha256,
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "vocabulary_size": binding.vocab_size,
        "grouping_policy": "global exact model-visible prompt tokens within task family",
        "summary": summary,
        "families": {
            family: render_counts(family_counters[family]) for family in FAMILIES
        },
        "samples": samples,
        "prompt_targets_deterministic": summary["conflicting_prompt_group_count"] == 0,
        "heldout_prompt_isolation": summary["cross_split_prompt_group_count"] == 0,
        "audit_complete": True,
    }
