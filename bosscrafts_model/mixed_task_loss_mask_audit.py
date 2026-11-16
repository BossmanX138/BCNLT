"""Fail-closed loss-mask audit for a mixed semantic and instruction curriculum."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from .current_task_packing import FAMILIES
from .immutable_artifacts import publish_new_files
from .mixed_curriculum import MIXED_CURRICULUM_SCHEMA
from .resumable_task_packing import (
    SALTED_SOURCE_EXAMPLE_ID_POLICY,
    ResumableTaskDataset,
    audit_task_database,
)
from .reviewed_instruction_curriculum import (
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_RESPONSE_FAMILY,
)
from .tokenizer import NativeTokenizerBinding

MIXED_LOSS_MASK_AUDIT_SCHEMA = "bosscrafts-mixed-task-loss-mask-audit-v1"
_SPLITS = ("train", "validation", "test")


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class MixedTaskLossMaskAuditResult:
    output_path: Path
    database_identity_sha256: str
    split_example_counts: dict[str, int]
    split_document_counts: dict[str, int]
    split_target_token_counts: dict[str, int]
    semantic_example_count: int
    instruction_example_count: int
    observed_instruction_training_weight_fraction: float
    maximum_sequence_tokens: int
    minimum_target_tokens: int
    receipt_sha256: str
    audit_passed: bool
    training_ready: bool


class MixedTaskLossMaskAuditor:
    """Verify every packed loss boundary and the complete mixed cell contract."""

    def __init__(
        self,
        binding: NativeTokenizerBinding,
        *,
        expected_languages: tuple[str, ...] | None = None,
        expected_semantic_families: tuple[str, ...] | None = None,
    ) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        languages = (
            tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))
            if expected_languages is None
            else expected_languages
        )
        families = (
            FAMILIES
            if expected_semantic_families is None
            else expected_semantic_families
        )
        if (
            type(languages) is not tuple
            or not languages
            or any(type(language) is not str or not language for language in languages)
            or len(set(languages)) != len(languages)
        ):
            raise ValueError("expected_languages must be a unique non-empty tuple")
        if (
            type(families) is not tuple
            or not families
            or any(family not in FAMILIES for family in families)
            or len(set(families)) != len(families)
        ):
            raise ValueError(
                "expected_semantic_families must be a unique non-empty subset"
            )
        self._binding = binding
        self._languages = tuple(sorted(languages))
        self._families = tuple(sorted(families))

    def materialize(
        self,
        database_path: str | Path,
        output_path: str | Path,
    ) -> MixedTaskLossMaskAuditResult:
        database = Path(database_path).resolve()
        output = Path(output_path).resolve()
        if not database.is_file():
            raise ValueError("mixed task database does not exist")
        if output.exists():
            raise ValueError("mixed loss-mask audit output already exists")

        database_audit = audit_task_database(self._binding, database)
        dataset = ResumableTaskDataset(self._binding, database)
        metadata = dataset.metadata
        if (
            metadata.get("curriculum_schema_version") != MIXED_CURRICULUM_SCHEMA
            or metadata.get("training_example_id_policy")
            != SALTED_SOURCE_EXAMPLE_ID_POLICY
            or not _is_sha256(metadata.get("curriculum_policy_sha256"))
            or not _is_sha256(metadata.get("semantic_source_database_identity_sha256"))
            or not _is_sha256(
                metadata.get("instruction_source_database_identity_sha256")
            )
        ):
            raise ValueError("mixed task database lineage metadata is invalid")
        raw_fraction = metadata.get("instruction_training_weight_fraction", "")
        try:
            configured_fraction = float(raw_fraction)
        except ValueError as error:
            raise ValueError(
                "mixed task instruction weight metadata is invalid"
            ) from error
        if (
            not math.isfinite(configured_fraction)
            or not 0.05 <= configured_fraction <= 0.95
            or raw_fraction != repr(configured_fraction)
        ):
            raise ValueError("mixed task instruction weight metadata is invalid")

        expected_cells = {
            (split, language, family)
            for split in _SPLITS
            for language in self._languages
            for family in self._families
        }
        expected_cells.update(
            (split, INSTRUCTION_LANGUAGE, INSTRUCTION_RESPONSE_FAMILY)
            for split in _SPLITS
        )
        cells: Counter[tuple[str, str, str]] = Counter()
        families: Counter[str] = Counter()
        split_example_counts: Counter[str] = Counter()
        split_target_counts: Counter[str] = Counter()
        split_documents: dict[str, set[str]] = {split: set() for split in _SPLITS}
        document_splits: dict[str, set[str]] = defaultdict(set)
        example_ids: set[str] = set()
        semantic_examples = 0
        instruction_examples = 0
        train_semantic_weight = 0.0
        train_instruction_weight = 0.0
        maximum_sequence = 0
        minimum_target: int | None = None

        for row in dataset.iter_examples():
            split = str(row["split"])
            language = str(row["language"])
            family = str(row["family"])
            example_id = str(row["example_id"])
            document_id = str(row["document_id"])
            if example_id in example_ids:
                raise ValueError("mixed task database contains a duplicate example ID")
            example_ids.add(example_id)
            input_ids = row["input_ids"]
            loss_start = int(row["loss_start_index"])
            sequence_tokens = len(input_ids)
            target_tokens = sequence_tokens - loss_start
            causal_targets = sum(
                position >= loss_start for position in range(1, sequence_tokens)
            )
            if target_tokens < 1 or causal_targets != target_tokens:
                raise ValueError("packed causal loss mask boundary is inconsistent")
            cell = (split, language, family)
            cells[cell] += 1
            families[family] += 1
            split_example_counts[split] += 1
            split_target_counts[split] += target_tokens
            split_documents[split].add(document_id)
            document_splits[document_id].add(split)
            maximum_sequence = max(maximum_sequence, sequence_tokens)
            minimum_target = (
                target_tokens
                if minimum_target is None
                else min(minimum_target, target_tokens)
            )
            weight = float(row["sampling_weight"])
            if family == INSTRUCTION_RESPONSE_FAMILY:
                instruction_examples += 1
                if split == "train":
                    train_instruction_weight += weight
            else:
                semantic_examples += 1
                if split == "train":
                    train_semantic_weight += weight

        if set(cells) != expected_cells or any(count < 1 for count in cells.values()):
            raise ValueError("mixed task database lacks complete expected cells")
        if any(len(splits) != 1 for splits in document_splits.values()):
            raise ValueError("source document crosses mixed task splits")
        if minimum_target is None or semantic_examples < 1 or instruction_examples < 1:
            raise ValueError("mixed loss-mask audit found an empty task family")
        total_train_weight = train_semantic_weight + train_instruction_weight
        if total_train_weight <= 0.0:
            raise ValueError("mixed task database has no positive training weight")
        observed_fraction = train_instruction_weight / total_train_weight
        if not math.isclose(
            observed_fraction,
            configured_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "mixed task instruction training weight does not match policy"
            )

        connection = sqlite3.connect(
            f"{database.as_uri()}?mode=ro", uri=True, timeout=30
        )
        connection.execute("PRAGMA query_only=ON")
        try:
            source_identity_counts = {
                str(kind): int(count)
                for kind, count in connection.execute(
                    "SELECT source_kind,COUNT(*) FROM source_example_identity "
                    "GROUP BY source_kind ORDER BY source_kind"
                )
            }
        except sqlite3.DatabaseError as error:
            raise ValueError("mixed task source identity map is invalid") from error
        finally:
            connection.close()
        if source_identity_counts != {
            "instruction": instruction_examples,
            "replay": semantic_examples,
        }:
            raise ValueError(
                "mixed task source identity counts do not match task families"
            )

        unsigned = {
            "schema_version": MIXED_LOSS_MASK_AUDIT_SCHEMA,
            "binding_sha256": self._binding.binding_sha256,
            "tokenizer_sha256": self._binding.tokenizer_sha256,
            "vocabulary_size": self._binding.vocab_size,
            "database": {
                "path": str(database),
                "database_identity_sha256": database_audit.database_identity_sha256,
                "content_sha256": database_audit.content_sha256,
                "example_count": database_audit.example_count,
                "document_count": database_audit.document_count,
            },
            "curriculum_policy_sha256": metadata["curriculum_policy_sha256"],
            "semantic_source_database_identity_sha256": metadata[
                "semantic_source_database_identity_sha256"
            ],
            "instruction_source_database_identity_sha256": metadata[
                "instruction_source_database_identity_sha256"
            ],
            "expected_semantic_languages": list(self._languages),
            "expected_semantic_families": list(self._families),
            "expected_instruction_cell": (
                f"{INSTRUCTION_LANGUAGE}|{INSTRUCTION_RESPONSE_FAMILY}"
            ),
            "cell_counts": {
                f"{split}|{language}|{family}": count
                for (split, language, family), count in sorted(cells.items())
            },
            "family_counts": dict(sorted(families.items())),
            "source_identity_counts": source_identity_counts,
            "split_example_counts": {
                split: split_example_counts[split] for split in _SPLITS
            },
            "split_document_counts": {
                split: len(split_documents[split]) for split in _SPLITS
            },
            "split_target_token_counts": {
                split: split_target_counts[split] for split in _SPLITS
            },
            "configured_instruction_training_weight_fraction": configured_fraction,
            "observed_instruction_training_weight_fraction": observed_fraction,
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
            artifact_label="mixed loss-mask audit",
        )
        return MixedTaskLossMaskAuditResult(
            output_path=output,
            database_identity_sha256=database_audit.database_identity_sha256,
            split_example_counts={
                split: split_example_counts[split] for split in _SPLITS
            },
            split_document_counts={
                split: len(split_documents[split]) for split in _SPLITS
            },
            split_target_token_counts={
                split: split_target_counts[split] for split in _SPLITS
            },
            semantic_example_count=semantic_examples,
            instruction_example_count=instruction_examples,
            observed_instruction_training_weight_fraction=observed_fraction,
            maximum_sequence_tokens=maximum_sequence,
            minimum_target_tokens=minimum_target,
            receipt_sha256=receipt_sha,
            audit_passed=True,
            training_ready=False,
        )
