"""Fail-closed, no-optimizer audit of packed causal loss boundaries."""
from __future__ import annotations

import hashlib
import heapq
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from .current_task_packing import FAMILIES
from .immutable_artifacts import publish_new_files
from .resumable_task_packing import audit_task_database
from .tokenizer import NativeTokenizerBinding

LOSS_MASK_AUDIT_SCHEMA = "bosscrafts-task-loss-mask-audit-v1"
_SPLITS = ("train", "validation", "test")


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _assert_cross_split_example_ids_disjoint(
    databases: Mapping[str, Path], expected_counts: Mapping[str, int]
) -> None:
    """Merge indexed ID streams without retaining the corpus in memory."""
    connections: list[sqlite3.Connection] = []
    cursors: list[sqlite3.Cursor] = []
    heap: list[tuple[str, int]] = []
    observed_counts = {split: 0 for split in _SPLITS}
    try:
        for index, split in enumerate(_SPLITS):
            database = databases[split]
            connection = sqlite3.connect(
                f"{database.as_uri()}?mode=ro", uri=True, timeout=30
            )
            connection.execute("PRAGMA query_only=ON")
            connections.append(connection)
            cursor = connection.execute("SELECT example_id FROM tasks ORDER BY example_id")
            cursors.append(cursor)
            first = cursor.fetchone()
            if first is not None:
                heapq.heappush(heap, (str(first[0]), index))

        previous_id: str | None = None
        previous_index: int | None = None
        while heap:
            example_id, index = heapq.heappop(heap)
            split = _SPLITS[index]
            observed_counts[split] += 1
            if example_id == previous_id and index != previous_index:
                raise ValueError("task example identity crosses frozen splits")
            previous_id = example_id
            previous_index = index
            following = cursors[index].fetchone()
            if following is not None:
                heapq.heappush(heap, (str(following[0]), index))
    finally:
        for cursor in cursors:
            cursor.close()
        for connection in connections:
            connection.close()
    if observed_counts != dict(expected_counts):
        raise ValueError("task database changed during loss-mask audit")


def _read_loss_mask_summary(
    database: Path,
) -> tuple[Counter[tuple[str, str]], set[str], int, int, int, int]:
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    try:
        rows = connection.execute(
            """SELECT split, language, family, COUNT(*),
                      COALESCE(SUM(token_count-loss_start_index), 0),
                      COALESCE(MAX(token_count), 0),
                      COALESCE(MIN(token_count-loss_start_index), 0),
                      SUM(CASE WHEN loss_start_index <= 0
                                    OR loss_start_index >= token_count
                               THEN 1 ELSE 0 END)
               FROM tasks GROUP BY split, language, family"""
        ).fetchall()
        has_document_ledger = (
            connection.execute(
                "SELECT 1 FROM sqlite_schema "
                "WHERE type='table' AND name='documents'"
            ).fetchone()
            is not None
        )
        if has_document_ledger:
            document_rows = list(
                connection.execute(
                    "SELECT document_id,task_count FROM documents "
                    "WHERE task_count>0 ORDER BY document_id"
                )
            )
        else:
            document_rows = list(
                connection.execute(
                    "SELECT DISTINCT document_id,NULL FROM tasks ORDER BY document_id"
                )
            )
    finally:
        connection.close()
    cells: Counter[tuple[str, str]] = Counter()
    example_count = 0
    target_total = 0
    maximum_sequence = 0
    minimum_target: int | None = None
    for (
        split,
        language,
        family,
        count,
        target_count,
        cell_maximum,
        cell_minimum,
        invalid_count,
    ) in rows:
        if split not in _SPLITS or int(invalid_count) != 0:
            raise ValueError("packed causal loss mask boundary is inconsistent")
        cells[(str(language), str(family))] += int(count)
        example_count += int(count)
        target_total += int(target_count)
        maximum_sequence = max(maximum_sequence, int(cell_maximum))
        minimum_target = (
            int(cell_minimum)
            if minimum_target is None
            else min(minimum_target, int(cell_minimum))
        )
    document_ids = {str(row[0]) for row in document_rows}
    declared_task_count = (
        sum(int(row[1]) for row in document_rows) if has_document_ledger else example_count
    )
    if len(document_ids) != len(document_rows) or declared_task_count != example_count:
        raise ValueError("task document accounting is inconsistent")
    if minimum_target is None or minimum_target < 1:
        raise ValueError("packed causal loss mask boundary is inconsistent")
    return (
        cells,
        document_ids,
        example_count,
        target_total,
        maximum_sequence,
        minimum_target,
    )


@dataclass(frozen=True, slots=True)
class TaskLossMaskAuditResult:
    output_path: Path
    split_example_counts: dict[str, int]
    split_document_counts: dict[str, int]
    split_target_token_counts: dict[str, int]
    maximum_sequence_tokens: int
    minimum_target_tokens: int
    receipt_sha256: str
    audit_passed: bool
    training_ready: bool


class TaskLossMaskAuditor:
    def __init__(
        self,
        binding: NativeTokenizerBinding,
        *,
        expected_languages: tuple[str, ...] | None = None,
    ) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        languages = (
            tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))
            if expected_languages is None
            else expected_languages
        )
        if (
            type(languages) is not tuple
            or not languages
            or any(type(language) is not str or not language for language in languages)
            or len(set(languages)) != len(languages)
        ):
            raise ValueError("expected_languages must be a unique non-empty tuple")
        self._binding = binding
        self._languages = tuple(sorted(languages))

    def materialize(
        self,
        databases: Mapping[str, str | Path],
        output_path: str | Path,
    ) -> TaskLossMaskAuditResult:
        if type(databases) is not dict or set(databases) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError("loss-mask audit requires train, validation and test databases")
        output = Path(output_path).resolve()
        if output.exists():
            raise ValueError("loss-mask audit output already exists")

        database_receipts: dict[str, dict[str, object]] = {}
        split_example_counts: dict[str, int] = {}
        split_document_counts: dict[str, int] = {}
        split_target_counts: dict[str, int] = {}
        all_document_ids: set[str] = set()
        resolved_databases: dict[str, Path] = {}
        maximum_sequence = 0
        minimum_target: int | None = None

        for split in _SPLITS:
            database = Path(databases[split]).resolve()
            resolved_databases[split] = database
            database_audit = audit_task_database(self._binding, database)
            (
                cells,
                document_ids,
                example_count,
                target_total,
                split_maximum,
                split_minimum,
            ) = _read_loss_mask_summary(database)
            maximum_sequence = max(maximum_sequence, split_maximum)
            minimum_target = (
                split_minimum
                if minimum_target is None
                else min(minimum_target, split_minimum)
            )
            expected_cells = {
                (language, family)
                for language in self._languages
                for family in FAMILIES
            }
            observed_splits = set(database_audit.split_counts)
            if observed_splits != {split}:
                raise ValueError(f"{split} task database contains another split")
            if set(cells) != expected_cells or any(count < 1 for count in cells.values()):
                raise ValueError(f"{split} task database lacks complete language-family cells")
            if example_count != database_audit.example_count:
                raise ValueError("task database changed during loss-mask audit")
            if len(document_ids) != database_audit.document_count:
                raise ValueError("task database changed during loss-mask audit")
            if split_maximum != database_audit.maximum_sequence_tokens:
                raise ValueError("task database changed during loss-mask audit")
            if all_document_ids.intersection(document_ids):
                raise ValueError("source document crosses frozen task splits")
            all_document_ids.update(document_ids)
            split_example_counts[split] = example_count
            split_document_counts[split] = len(document_ids)
            split_target_counts[split] = target_total
            database_receipts[split] = {
                "path": str(database),
                "database_identity_sha256": database_audit.database_identity_sha256,
                "content_sha256": database_audit.content_sha256,
                "example_count": database_audit.example_count,
                "document_count": database_audit.document_count,
                "maximum_sequence_tokens": database_audit.maximum_sequence_tokens,
                "cell_counts": {
                    f"{language}|{family}": count
                    for (language, family), count in sorted(cells.items())
                },
            }

        _assert_cross_split_example_ids_disjoint(
            resolved_databases, split_example_counts
        )

        if minimum_target is None:
            raise ValueError("loss-mask audit found no examples")
        unsigned = {
            "schema_version": LOSS_MASK_AUDIT_SCHEMA,
            "binding_sha256": self._binding.binding_sha256,
            "tokenizer_sha256": self._binding.tokenizer_sha256,
            "vocabulary_size": self._binding.vocab_size,
            "expected_languages": list(self._languages),
            "expected_families": list(FAMILIES),
            "databases": database_receipts,
            "split_example_counts": split_example_counts,
            "split_document_counts": split_document_counts,
            "split_target_token_counts": split_target_counts,
            "maximum_sequence_tokens": maximum_sequence,
            "minimum_target_tokens": minimum_target,
            "cross_split_example_count": 0,
            "cross_split_document_count": 0,
            "audit_passed": True,
            "training_ready": False,
            "training_blockers": [
                "semantic uncertainty calibration is not promoted",
                "frozen model initialization and resource plan are not approved",
                "model training remains explicitly held",
            ],
        }
        receipt_sha = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
        receipt = {**unsigned, "receipt_sha256": receipt_sha}
        publish_new_files(
            ((output, f"{_canonical(receipt)}\n".encode("utf-8")),),
            artifact_label="loss-mask audit",
        )
        return TaskLossMaskAuditResult(
            output_path=output,
            split_example_counts=split_example_counts,
            split_document_counts=split_document_counts,
            split_target_token_counts=split_target_counts,
            maximum_sequence_tokens=maximum_sequence,
            minimum_target_tokens=minimum_target,
            receipt_sha256=receipt_sha,
            audit_passed=True,
            training_ready=False,
        )
