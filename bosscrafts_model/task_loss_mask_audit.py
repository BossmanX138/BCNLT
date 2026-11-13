"""Fail-closed, no-optimizer audit of packed causal loss boundaries."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from .current_task_packing import FAMILIES
from .immutable_artifacts import publish_new_files
from .resumable_task_packing import (
    ResumableTaskDataset,
    audit_task_database,
)
from .tokenizer import NativeTokenizerBinding

LOSS_MASK_AUDIT_SCHEMA = "bosscrafts-task-loss-mask-audit-v1"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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
        all_example_ids: set[str] = set()
        all_document_ids: set[str] = set()
        maximum_sequence = 0
        minimum_target: int | None = None

        for split in ("train", "validation", "test"):
            database = Path(databases[split]).resolve()
            database_audit = audit_task_database(self._binding, database)
            dataset = ResumableTaskDataset(self._binding, database)
            cells: Counter[tuple[str, str]] = Counter()
            example_ids: set[str] = set()
            document_ids: set[str] = set()
            target_total = 0
            for row in dataset.iter_examples():
                if row["split"] != split:
                    raise ValueError(f"{split} task database contains another split")
                example_id = str(row["example_id"])
                document_id = str(row["document_id"])
                if example_id in example_ids:
                    raise ValueError("task database contains duplicate example identity")
                example_ids.add(example_id)
                document_ids.add(document_id)
                language = str(row["language"])
                family = str(row["family"])
                cells[(language, family)] += 1
                input_ids = row["input_ids"]
                loss_start = int(row["loss_start_index"])
                sequence_tokens = len(input_ids)
                target_tokens = sequence_tokens - loss_start
                causal_targets = sum(
                    position >= loss_start for position in range(1, sequence_tokens)
                )
                if target_tokens < 1 or causal_targets != target_tokens:
                    raise ValueError("packed causal loss mask boundary is inconsistent")
                target_total += target_tokens
                maximum_sequence = max(maximum_sequence, sequence_tokens)
                minimum_target = (
                    target_tokens
                    if minimum_target is None
                    else min(minimum_target, target_tokens)
                )
            expected_cells = {
                (language, family)
                for language in self._languages
                for family in FAMILIES
            }
            if set(cells) != expected_cells or any(count < 1 for count in cells.values()):
                raise ValueError(f"{split} task database lacks complete language-family cells")
            if all_example_ids.intersection(example_ids):
                raise ValueError("task example identity crosses frozen splits")
            if all_document_ids.intersection(document_ids):
                raise ValueError("source document crosses frozen task splits")
            all_example_ids.update(example_ids)
            all_document_ids.update(document_ids)
            split_example_counts[split] = len(example_ids)
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
