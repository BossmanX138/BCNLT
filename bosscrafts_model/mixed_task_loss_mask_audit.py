"""Fail-closed loss-mask audit for a mixed semantic and instruction curriculum."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from .current_task_packing import FAMILIES
from .immutable_artifacts import publish_new_files
from .mixed_curriculum import (
    MIXED_CURRICULUM_SCHEMA,
    SPLIT_SAFE_MIXED_TRAINING_SCHEMA,
)
from .resumable_task_packing import (
    SALTED_SOURCE_EXAMPLE_ID_POLICY,
    ResumableTaskDataset,
    audit_task_database,
)
from .reviewed_instruction_curriculum import (
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_RESPONSE_FAMILY,
)
from .task_database_audit_manifest import (
    load_verified_task_database_audit_manifest,
)
from .task_loss_mask_audit import _read_loss_mask_summary
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


@dataclass(frozen=True, slots=True)
class SplitSafeMixedTrainingLossMaskAuditResult:
    output_path: Path
    database_identity_sha256: str
    example_count: int
    document_count: int
    target_token_count: int
    semantic_example_count: int
    instruction_train_example_count: int
    excluded_instruction_validation_example_count: int
    excluded_instruction_test_example_count: int
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
        split_document_counts: Counter[str] = Counter()
        maximum_sequence = 0
        minimum_target: int | None = None
        connection = sqlite3.connect(
            f"{database.as_uri()}?mode=ro", uri=True, timeout=30
        )
        connection.execute("PRAGMA query_only=ON")
        try:
            rows = connection.execute(
                """SELECT split,language,family,COUNT(*),
                          COALESCE(SUM(token_count-loss_start_index),0),
                          COALESCE(MAX(token_count),0),
                          COALESCE(MIN(token_count-loss_start_index),0),
                          SUM(CASE WHEN loss_start_index <= 0
                                        OR loss_start_index >= token_count
                                   THEN 1 ELSE 0 END)
                   FROM tasks GROUP BY split,language,family"""
            ).fetchall()
            for document_id, minimum_split, maximum_split in connection.execute(
                "SELECT document_id,MIN(split),MAX(split) FROM tasks "
                "GROUP BY document_id ORDER BY document_id"
            ):
                if minimum_split != maximum_split:
                    raise ValueError("source document crosses mixed task splits")
                split_document_counts[str(minimum_split)] += 1
            source_identity_counts = {
                str(kind): int(count)
                for kind, count in connection.execute(
                    "SELECT source_kind,COUNT(*) FROM source_example_identity "
                    "GROUP BY source_kind ORDER BY source_kind"
                )
            }
        except sqlite3.DatabaseError as error:
            raise ValueError("mixed task database aggregate tables are invalid") from error
        finally:
            connection.close()
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
            split_name = str(split)
            family_name = str(family)
            if split_name not in _SPLITS or int(invalid_count) != 0:
                raise ValueError("packed causal loss mask boundary is inconsistent")
            cell = (split_name, str(language), family_name)
            cells[cell] += int(count)
            families[family_name] += int(count)
            split_example_counts[split_name] += int(count)
            split_target_counts[split_name] += int(target_count)
            maximum_sequence = max(maximum_sequence, int(cell_maximum))
            minimum_target = (
                int(cell_minimum)
                if minimum_target is None
                else min(minimum_target, int(cell_minimum))
            )
        instruction_examples = families[INSTRUCTION_RESPONSE_FAMILY]
        semantic_examples = sum(families.values()) - instruction_examples

        if set(cells) != expected_cells or any(count < 1 for count in cells.values()):
            raise ValueError("mixed task database lacks complete expected cells")
        if minimum_target is None or semantic_examples < 1 or instruction_examples < 1:
            raise ValueError("mixed loss-mask audit found an empty task family")
        if (
            sum(split_example_counts.values()) != database_audit.example_count
            or sum(split_document_counts.values()) != database_audit.document_count
            or maximum_sequence != database_audit.maximum_sequence_tokens
        ):
            raise ValueError("mixed task database changed during loss-mask audit")
        cell_weight_totals = dataset.cell_sampling_weight_totals(split="train")
        train_instruction_weight = sum(
            weight
            for (_, _, family), weight in cell_weight_totals.items()
            if family == INSTRUCTION_RESPONSE_FAMILY
        )
        total_train_weight = sum(cell_weight_totals.values())
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
                split: split_document_counts[split] for split in _SPLITS
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
                split: split_document_counts[split] for split in _SPLITS
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


class SplitSafeMixedTrainingLossMaskAuditor:
    """Audit a train-only mix without admitting either held-out split."""

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

    @staticmethod
    def _positive_metadata_count(metadata: dict[str, str], key: str) -> int:
        raw = metadata.get(key, "")
        try:
            count = int(raw)
        except ValueError as error:
            raise ValueError("split-safe mixed held-out count is invalid") from error
        if raw != str(count) or count < 1:
            raise ValueError("split-safe mixed held-out count is invalid")
        return count

    def materialize(
        self,
        database_path: str | Path,
        output_path: str | Path,
        *,
        audit_manifest: str | Path | None = None,
    ) -> SplitSafeMixedTrainingLossMaskAuditResult:
        database = Path(database_path).resolve()
        output = Path(output_path).resolve()
        if not database.is_file():
            raise ValueError("split-safe mixed training database does not exist")
        if output.exists():
            raise ValueError("split-safe mixed loss-mask audit output already exists")

        database_audit = (
            audit_task_database(self._binding, database)
            if audit_manifest is None
            else load_verified_task_database_audit_manifest(
                self._binding, database, audit_manifest
            ).audit
        )
        dataset = ResumableTaskDataset(self._binding, database)
        metadata = dataset.metadata
        if (
            metadata.get("curriculum_schema_version")
            != SPLIT_SAFE_MIXED_TRAINING_SCHEMA
            or metadata.get("curriculum_source_split") != "train"
            or metadata.get("training_example_id_policy")
            != SALTED_SOURCE_EXAMPLE_ID_POLICY
            or not _is_sha256(metadata.get("curriculum_policy_sha256"))
            or not _is_sha256(
                metadata.get("semantic_train_source_database_identity_sha256")
            )
            or not _is_sha256(
                metadata.get("instruction_source_database_identity_sha256")
            )
            or set(database_audit.split_counts) != {"train"}
        ):
            raise ValueError("split-safe mixed training lineage metadata is invalid")
        excluded_validation = self._positive_metadata_count(
            metadata, "excluded_instruction_validation_example_count"
        )
        excluded_test = self._positive_metadata_count(
            metadata, "excluded_instruction_test_example_count"
        )
        expected_instruction_train = self._positive_metadata_count(
            metadata, "instruction_source_train_example_count"
        )
        raw_fraction = metadata.get("instruction_training_weight_fraction", "")
        try:
            configured_fraction = float(raw_fraction)
        except ValueError as error:
            raise ValueError(
                "split-safe mixed instruction weight metadata is invalid"
            ) from error
        if (
            not math.isfinite(configured_fraction)
            or not 0.05 <= configured_fraction <= 0.95
            or raw_fraction != repr(configured_fraction)
        ):
            raise ValueError("split-safe mixed instruction weight metadata is invalid")

        (
            cells,
            document_ids,
            example_count,
            target_total,
            maximum_sequence,
            minimum_target,
        ) = _read_loss_mask_summary(database)
        expected_cells = {
            (language, family)
            for language in self._languages
            for family in self._families
        }
        expected_cells.add((INSTRUCTION_LANGUAGE, INSTRUCTION_RESPONSE_FAMILY))
        if set(cells) != expected_cells or any(count < 1 for count in cells.values()):
            raise ValueError("split-safe mixed training database lacks expected cells")
        if (
            example_count != database_audit.example_count
            or len(document_ids) != database_audit.document_count
            or maximum_sequence != database_audit.maximum_sequence_tokens
        ):
            raise ValueError("split-safe mixed training database changed during audit")
        instruction_examples = cells[
            (INSTRUCTION_LANGUAGE, INSTRUCTION_RESPONSE_FAMILY)
        ]
        semantic_examples = example_count - instruction_examples
        if instruction_examples != expected_instruction_train or semantic_examples < 1:
            raise ValueError("split-safe mixed training source counts are inconsistent")

        cell_weight_totals = dataset.cell_sampling_weight_totals(split="train")
        instruction_weight = sum(
            weight
            for (_, _, family), weight in cell_weight_totals.items()
            if family == INSTRUCTION_RESPONSE_FAMILY
        )
        total_weight = sum(cell_weight_totals.values())
        if total_weight <= 0.0:
            raise ValueError("split-safe mixed training database has no positive weight")
        observed_fraction = instruction_weight / total_weight
        if not math.isclose(
            observed_fraction,
            configured_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "split-safe mixed instruction training weight does not match policy"
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
            raise ValueError("split-safe mixed source identity map is invalid") from error
        finally:
            connection.close()
        if source_identity_counts != {
            "instruction": instruction_examples,
            "replay": semantic_examples,
        }:
            raise ValueError("split-safe mixed source identity counts are inconsistent")
        if audit_manifest is not None:
            reverified = load_verified_task_database_audit_manifest(
                self._binding,
                database,
                audit_manifest,
                expected_database_identity_sha256=(
                    database_audit.database_identity_sha256
                ),
            ).audit
            if reverified.content_sha256 != database_audit.content_sha256:
                raise ValueError(
                    "split-safe mixed training database changed during audit"
                )

        unsigned = {
            "schema_version": "bosscrafts-split-safe-mixed-loss-mask-audit-v1",
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
            "semantic_train_source_database_identity_sha256": metadata[
                "semantic_train_source_database_identity_sha256"
            ],
            "instruction_source_database_identity_sha256": metadata[
                "instruction_source_database_identity_sha256"
            ],
            "expected_semantic_languages": list(self._languages),
            "expected_semantic_families": list(self._families),
            "cell_counts": {
                f"train|{language}|{family}": count
                for (language, family), count in sorted(cells.items())
            },
            "source_identity_counts": source_identity_counts,
            "example_count": example_count,
            "document_count": len(document_ids),
            "target_token_count": target_total,
            "semantic_example_count": semantic_examples,
            "instruction_train_example_count": instruction_examples,
            "excluded_instruction_validation_example_count": excluded_validation,
            "excluded_instruction_test_example_count": excluded_test,
            "configured_instruction_training_weight_fraction": configured_fraction,
            "observed_instruction_training_weight_fraction": observed_fraction,
            "maximum_sequence_tokens": maximum_sequence,
            "minimum_target_tokens": minimum_target,
            "admitted_splits": ["train"],
            "excluded_splits": ["validation", "test"],
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
            artifact_label="split-safe mixed loss-mask audit",
        )
        return SplitSafeMixedTrainingLossMaskAuditResult(
            output_path=output,
            database_identity_sha256=database_audit.database_identity_sha256,
            example_count=example_count,
            document_count=len(document_ids),
            target_token_count=target_total,
            semantic_example_count=semantic_examples,
            instruction_train_example_count=instruction_examples,
            excluded_instruction_validation_example_count=excluded_validation,
            excluded_instruction_test_example_count=excluded_test,
            observed_instruction_training_weight_fraction=observed_fraction,
            maximum_sequence_tokens=maximum_sequence,
            minimum_target_tokens=minimum_target,
            receipt_sha256=receipt_sha,
            audit_passed=True,
            training_ready=False,
        )
