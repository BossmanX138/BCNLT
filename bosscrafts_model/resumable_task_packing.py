"""Crash-resumable current-lineage task packing backed by committed SQLite rows."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
import zlib
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from .bcnlt_adapter import MAX_CANONICAL_OUTPUT_BYTES
from .bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
    PARSER_RUNTIME_NOT_APPLICABLE,
)
from .compact_semantic_wire import COMPACT_SEMANTIC_WIRE_SCHEMA
from .compact_task_packing import CompactCurrentCorpusTaskPacker
from .compression import decompress_bounded
from .current_task_packing import CurrentCorpusTaskPacker
from .jsonl import split_jsonl_records
from .resumable_current_corpus import RESUMABLE_SCHEMA
from .resumable_training_semantic_audit import ResumableTrainingSemanticAudit
from .tokenizer import NativeTokenizerBinding
from .training_eligibility import (
    ELIGIBILITY_MANIFEST_SCHEMA,
    ELIGIBLE_DOCUMENT_SCHEMA,
)

TASK_DATABASE_SCHEMA = "bosscrafts-current-task-resumable-v3"
FULL_SEMANTIC_WIRE_SCHEMA = "bosscrafts-full-semantic-wire-v3"
CELL_WEIGHT_MULTIPLIERS_METADATA_KEY = "training_cell_weight_multipliers_v1_json"
SALTED_SOURCE_EXAMPLE_ID_POLICY = "bosscrafts-v11-salted-source-example-id-v1"
_TASK_FAMILIES = frozenset(
    {
        "instruction_to_code",
        "code_to_semantic",
        "semantic_continuation",
        "semantic_to_code",
    }
)


class TaskExample(TypedDict):
    example_id: str
    document_id: str
    language: str
    family: str
    record_index: int
    split: str
    input_ids: list[int]
    loss_start_index: int
    sampling_weight: float
    semantic_spec_sha256: str
    parser_runtime_identity_sha256: str
    source_sha256: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def salted_source_example_id(source_kind: str, source_example_id: str) -> str:
    if source_kind not in {"replay", "instruction"}:
        raise ValueError("source example kind is invalid")
    if (
        type(source_example_id) is not str
        or len(source_example_id) != 64
        or any(character not in "0123456789abcdef" for character in source_example_id)
    ):
        raise ValueError("source example ID is invalid")
    return hashlib.sha256(
        f"{SALTED_SOURCE_EXAMPLE_ID_POLICY}:{source_kind}:{source_example_id}".encode()
    ).hexdigest()


def _decode_u16(content: bytes) -> list[int]:
    if len(content) % 2:
        raise ValueError("semantic token blob has odd byte length")
    values = array("H")
    values.frombytes(content)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def _encode_u16(values: list[int]) -> bytes:
    if any(type(value) is not int or not 0 <= value <= 65_535 for value in values):
        raise ValueError("packed task token IDs must fit uint16")
    encoded = array("H", values)
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


@dataclass(frozen=True)
class ResumableTaskPackingStatus:
    database_path: Path
    processed_this_run: int
    processed_documents: int
    task_count: int
    rejected_count: int
    total_sequence_tokens: int
    packing_complete: bool


@dataclass(frozen=True)
class TaskDatabaseAuditResult:
    database_path: Path
    content_sha256: str
    database_identity_sha256: str
    example_count: int
    total_sequence_tokens: int
    maximum_sequence_tokens: int
    split_counts: dict[str, int]
    language_counts: dict[str, int]
    family_counts: dict[str, int]
    document_count: int
    audit_passed: bool


class ResumableTaskDataset:
    """Read a completed task database with deterministic balanced weights."""

    def __init__(
        self, binding: NativeTokenizerBinding, database_path: str | Path
    ) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        self._binding = binding
        self.database_path = Path(database_path).resolve()
        if not self.database_path.is_file():
            raise ValueError("task database does not exist")
        connection = ResumableCurrentTaskPacking._ro_connect(self.database_path)
        try:
            self.metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if self.metadata.get("schema_version") != TASK_DATABASE_SCHEMA:
                raise ValueError("task database schema mismatch")
            if (
                self.metadata.get("binding_sha256") != binding.binding_sha256
                or self.metadata.get("tokenizer_sha256") != binding.tokenizer_sha256
                or self.metadata.get("vocabulary_size") != str(binding.vocab_size)
                or self.metadata.get("parser_runtime_identity_sha256")
                != PARSER_RUNTIME_IDENTITY_SHA256
            ):
                raise ValueError("task database tokenizer lineage mismatch")
            if self.metadata.get("packing_complete") != "true":
                raise ValueError("task database packing is incomplete")
            raw_maximum_sequence = self.metadata.get("maximum_sequence_tokens")
            try:
                maximum_sequence = int(raw_maximum_sequence or "")
            except ValueError as error:
                raise ValueError(
                    "task database maximum sequence metadata is invalid"
                ) from error
            if (
                raw_maximum_sequence != str(maximum_sequence)
                or not 2 <= maximum_sequence <= 1_000_000
            ):
                raise ValueError("task database maximum sequence metadata is invalid")
            self._maximum_sequence_tokens = maximum_sequence
            self.identity_sha256 = hashlib.sha256(
                _canonical_json(self.metadata).encode("utf-8")
            ).hexdigest()
            counts = connection.execute(
                """SELECT split, language, family, COUNT(*)
                   FROM tasks GROUP BY split, language, family"""
            ).fetchall()
        finally:
            connection.close()
        if not counts:
            raise ValueError("completed task database is empty")
        by_split: dict[str, list[tuple[str, str, str, int]]] = {}
        for split, language, family, count in counts:
            if (
                type(split) is not str
                or split not in {"train", "validation", "test"}
                or type(language) is not str
                or language not in CURRENT_RUNTIME_SPEC_SHA256
                or type(family) is not str
                or family not in _TASK_FAMILIES
                or type(count) is not int
                or count < 1
            ):
                raise ValueError("task database split, language, or family is invalid")
            by_split.setdefault(split, []).append((split, language, family, count))
        raw_multipliers = self.metadata.get(CELL_WEIGHT_MULTIPLIERS_METADATA_KEY)
        multipliers: dict[tuple[str, str, str], float] = {}
        if raw_multipliers is not None:
            try:
                parsed_multipliers = json.loads(raw_multipliers)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "task database cell weight multipliers are invalid"
                ) from error
            if type(parsed_multipliers) is not dict:
                raise ValueError("task database cell weight multipliers are invalid")
            for key, value in parsed_multipliers.items():
                parts = key.split("|") if type(key) is str else []
                if (
                    len(parts) != 3
                    or parts[0] not in {"train", "validation", "test"}
                    or type(value) not in {int, float}
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                ):
                    raise ValueError(
                        "task database cell weight multipliers are invalid"
                    )
                multipliers[(parts[0], parts[1], parts[2])] = float(value)
        observed_cells = {
            (split, language, family)
            for rows in by_split.values()
            for split, language, family, _ in rows
        }
        if not set(multipliers).issubset(observed_cells):
            raise ValueError(
                "task database cell weight multiplier names an absent cell"
            )

        self._weights: dict[tuple[str, str, str], float] = {}
        self._split_counts: Counter[str] = Counter()
        for split_rows in by_split.values():
            mean_count = sum(row[3] for row in split_rows) / len(split_rows)
            unnormalized: dict[tuple[str, str, str], float] = {}
            for split, language, family, count in split_rows:
                cell = (split, language, family)
                unnormalized[cell] = mean_count / count * multipliers.get(cell, 1.0)
                self._split_counts[split] += count
            total_examples = sum(row[3] for row in split_rows)
            total_weight = sum(
                unnormalized[(split, language, family)] * count
                for split, language, family, count in split_rows
            )
            normalization = total_examples / total_weight
            for cell, weight in unnormalized.items():
                self._weights[cell] = weight * normalization

    def example_count(self, *, split: str | None = None) -> int:
        if split is not None and split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, test or None")
        if split is None:
            return sum(self._split_counts.values())
        return self._split_counts[split]

    def iter_examples(
        self,
        *,
        split: str | None = None,
        after_example_id: str | None = None,
    ) -> Iterator[TaskExample]:
        if split is not None and split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, test or None")
        if after_example_id is not None and (
            len(after_example_id) != 64
            or any(
                character not in "0123456789abcdef" for character in after_example_id
            )
        ):
            raise ValueError("after_example_id must be a lowercase SHA-256 or None")
        connection = ResumableCurrentTaskPacking._ro_connect(self.database_path)
        try:
            query = (
                "SELECT example_id, document_id, language, family, record_index, split, "
                "input_ids_zlib, token_count, loss_start_index, "
                "semantic_spec_sha256, parser_runtime_identity_sha256, source_sha256 FROM tasks"
            )
            conditions: list[str] = []
            parameters: list[str] = []
            if split is not None:
                conditions.append("split=?")
                parameters.append(split)
            if after_example_id is not None:
                conditions.append("example_id>?")
                parameters.append(after_example_id)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY example_id"
            for row in connection.execute(query, tuple(parameters)):
                yield self._decode_task_row(row)
        finally:
            connection.close()

    def iter_examples_by_id(
        self,
        example_ids: Sequence[str],
        *,
        split: str,
    ) -> Iterator[TaskExample]:
        """Read an exact, bounded frozen subset without scanning the task table."""
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if isinstance(example_ids, (str, bytes)) or not isinstance(
            example_ids, Sequence
        ):
            raise TypeError("example_ids must be a sequence of lowercase SHA-256 IDs")
        requested = list(example_ids)
        if not requested or len(requested) > 100_000:
            raise ValueError("example_ids must contain between 1 and 100000 IDs")
        if any(
            type(example_id) is not str
            or len(example_id) != 64
            or any(character not in "0123456789abcdef" for character in example_id)
            for example_id in requested
        ):
            raise ValueError("example_ids must contain only lowercase SHA-256 IDs")
        if len(set(requested)) != len(requested):
            raise ValueError("example_ids must not contain duplicate IDs")

        selected: dict[str, TaskExample] = {}
        connection = ResumableCurrentTaskPacking._ro_connect(self.database_path)
        try:
            for offset in range(0, len(requested), 500):
                batch = requested[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                query = (
                    "SELECT example_id, document_id, language, family, record_index, split, "
                    "input_ids_zlib, token_count, loss_start_index, "
                    "semantic_spec_sha256, parser_runtime_identity_sha256, source_sha256 "
                    f"FROM tasks WHERE example_id IN ({placeholders})"
                )
                for row in connection.execute(query, batch):
                    decoded = self._decode_task_row(row)
                    example_id = decoded["example_id"]
                    if example_id in selected:
                        raise ValueError("task database returned a duplicate example ID")
                    selected[example_id] = decoded
        finally:
            connection.close()

        missing = [example_id for example_id in requested if example_id not in selected]
        if missing:
            raise ValueError("one or more requested example IDs are missing")
        wrong_split = [
            example_id
            for example_id in requested
            if selected[example_id]["split"] != split
        ]
        if wrong_split:
            raise ValueError("one or more requested example IDs belong to another split")
        for example_id in requested:
            yield selected[example_id]

    def select_frozen_example_ids_by_cell(
        self,
        *,
        split: str,
        expected_cells: Mapping[str, int],
    ) -> dict[str, tuple[str, ...]]:
        """Select the first deterministic IDs per cell without decoding token blobs."""
        if split not in {"validation", "test"}:
            raise ValueError("frozen selection split must be validation or test")
        if not isinstance(expected_cells, Mapping) or not expected_cells:
            raise ValueError("frozen selection cells must be a non-empty mapping")
        if len(expected_cells) > 1_000:
            raise ValueError("frozen selection names too many cells")
        requested: dict[str, int] = {}
        for cell, count in expected_cells.items():
            language, separator, family = (
                cell.partition("|") if type(cell) is str else ("", "", "")
            )
            if (
                not separator
                or language not in CURRENT_RUNTIME_SPEC_SHA256
                or family not in _TASK_FAMILIES
                or type(count) is not int
                or not 1 <= count <= 100_000
            ):
                raise ValueError("frozen selection cell contract is invalid")
            requested[cell] = count

        selected: dict[str, list[str]] = {cell: [] for cell in requested}
        remaining = set(requested)
        connection = ResumableCurrentTaskPacking._ro_connect(self.database_path)
        try:
            rows = connection.execute(
                "SELECT example_id,language,family FROM tasks "
                "WHERE split=? ORDER BY example_id",
                (split,),
            )
            for example_id, language, family in rows:
                cell = f"{language}|{family}"
                if cell not in remaining:
                    continue
                if (
                    type(example_id) is not str
                    or len(example_id) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in example_id
                    )
                ):
                    raise ValueError("frozen selection encountered an invalid example ID")
                selected[cell].append(example_id)
                if len(selected[cell]) == requested[cell]:
                    remaining.remove(cell)
                    if not remaining:
                        break
        finally:
            connection.close()
        if remaining:
            raise ValueError("task database cannot complete the frozen cell inventory")
        return {cell: tuple(example_ids) for cell, example_ids in selected.items()}

    def _decode_task_row(self, row: Sequence[object]) -> TaskExample:
        family = row[3]
        split_value = row[5]
        language = row[2]
        expected_runtime = (
            PARSER_RUNTIME_NOT_APPLICABLE
            if family == "instruction_to_code"
            else PARSER_RUNTIME_IDENTITY_SHA256
        )
        if (
            type(family) is not str
            or family not in _TASK_FAMILIES
            or type(split_value) is not str
            or split_value not in {"train", "validation", "test"}
            or type(language) is not str
            or language not in CURRENT_RUNTIME_SPEC_SHA256
            or str(row[10]) != expected_runtime
        ):
            raise ValueError("task row parser runtime lineage mismatch")
        token_count = row[7]
        if (
            type(token_count) is not int
            or not 2 <= token_count <= self._maximum_sequence_tokens
        ):
            raise ValueError("task token count exceeds configured maximum")
        if type(row[6]) is not bytes:
            raise ValueError("task token blob is invalid")
        expected_input_bytes = token_count * 2
        input_ids = _decode_u16(
            decompress_bounded(
                row[6],
                maximum_output_bytes=expected_input_bytes,
            )
        )
        if len(input_ids) != token_count:
            raise ValueError("task token count mismatch")
        if any(token_id >= self._binding.vocab_size for token_id in input_ids):
            raise ValueError("task token exceeds frozen vocabulary")
        loss_start = row[8]
        if type(loss_start) is not int or not 0 < loss_start < len(input_ids):
            raise ValueError("task loss boundary is invalid")
        record_index = row[4]
        if type(record_index) is not int or record_index < 0:
            raise ValueError("task record index is invalid")
        cell = (split_value, language, family)
        return {
            "example_id": str(row[0]),
            "document_id": str(row[1]),
            "language": cell[1],
            "family": cell[2],
            "record_index": record_index,
            "split": cell[0],
            "input_ids": input_ids,
            "loss_start_index": loss_start,
            "sampling_weight": self._weights[cell],
            "semantic_spec_sha256": str(row[9]),
            "parser_runtime_identity_sha256": str(row[10]),
            "source_sha256": str(row[11]),
        }


def audit_task_database(
    binding: NativeTokenizerBinding, database_path: str | Path
) -> TaskDatabaseAuditResult:
    dataset = ResumableTaskDataset(binding, database_path)
    try:
        maximum_allowed = int(dataset.metadata["maximum_sequence_tokens"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            "task database maximum sequence metadata is invalid"
        ) from error
    example_id_policy = dataset.metadata.get("training_example_id_policy")
    source_identities: dict[str, tuple[str, str]] | None = None
    if example_id_policy is not None:
        if example_id_policy != SALTED_SOURCE_EXAMPLE_ID_POLICY:
            raise ValueError("task database example ID policy is unsupported")
        connection = sqlite3.connect(
            f"{dataset.database_path.as_uri()}?mode=ro", uri=True, timeout=30
        )
        connection.execute("PRAGMA query_only=ON")
        try:
            mapping_rows = connection.execute(
                "SELECT example_id,source_example_id,source_kind "
                "FROM source_example_identity ORDER BY example_id"
            ).fetchall()
        except sqlite3.DatabaseError as error:
            raise ValueError("task database source identity map is invalid") from error
        finally:
            connection.close()
        mapping_digest = hashlib.sha256()
        source_identities = {}
        for example_id, source_example_id, source_kind in mapping_rows:
            new_id = str(example_id)
            source_id = str(source_example_id)
            kind = str(source_kind)
            if new_id in source_identities:
                raise ValueError("task database source identity map is duplicated")
            if salted_source_example_id(kind, source_id) != new_id:
                raise ValueError("task database salted example identity mismatch")
            source_identities[new_id] = (source_id, kind)
            mapping_digest.update(
                _canonical_json(
                    {
                        "example_id": new_id,
                        "source_example_id": source_id,
                        "source_kind": kind,
                    }
                ).encode("utf-8")
            )
        if not source_identities or mapping_digest.hexdigest() != dataset.metadata.get(
            "source_example_identity_map_sha256"
        ):
            raise ValueError("task database source identity map hash mismatch")
    split_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    document_splits: dict[str, set[str]] = defaultdict(set)
    content_digest = hashlib.sha256()
    example_count = 0
    total_sequence_tokens = 0
    maximum_sequence_tokens = 0
    for row in dataset.iter_examples():
        input_ids = row["input_ids"]
        input_ids_sha256 = hashlib.sha256(
            _canonical_json(input_ids).encode("utf-8")
        ).hexdigest()
        identity = {
            "document_id": row["document_id"],
            "family": row["family"],
            "record_index": row["record_index"],
            "input_ids_sha256": input_ids_sha256,
        }
        expected_id = hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()
        if source_identities is None:
            if row["example_id"] != expected_id:
                raise ValueError("task example identity mismatch")
        else:
            mapped = source_identities.get(str(row["example_id"]))
            expected_kind = (
                "instruction" if row["family"] == "instruction_to_code" else "replay"
            )
            if mapped != (expected_id, expected_kind):
                raise ValueError("task source example identity mismatch")
        sequence_length = len(input_ids)
        if sequence_length > maximum_allowed:
            raise ValueError("task sequence exceeds configured maximum")
        split = str(row["split"])
        language = str(row["language"])
        family = str(row["family"])
        split_counts[split] += 1
        language_counts[language] += 1
        family_counts[family] += 1
        document_splits[str(row["document_id"])].add(split)
        content_digest.update(
            _canonical_json(
                {
                    "document_id": row["document_id"],
                    "example_id": row["example_id"],
                    "family": family,
                    "input_ids_sha256": input_ids_sha256,
                    "language": language,
                    "loss_start_index": row["loss_start_index"],
                    "record_index": row["record_index"],
                    "parser_runtime_identity_sha256": row[
                        "parser_runtime_identity_sha256"
                    ],
                    "semantic_spec_sha256": row["semantic_spec_sha256"],
                    "source_sha256": row["source_sha256"],
                    "split": split,
                }
            ).encode("utf-8")
        )
        content_digest.update(b"\n")
        example_count += 1
        total_sequence_tokens += sequence_length
        maximum_sequence_tokens = max(maximum_sequence_tokens, sequence_length)
    if not example_count:
        raise ValueError("task database audit found no examples")
    if source_identities is not None and len(source_identities) != example_count:
        raise ValueError("task database source identity map count mismatch")
    if any(len(splits) != 1 for splits in document_splits.values()):
        raise ValueError("task document appears in multiple splits")
    content_sha256 = content_digest.hexdigest()
    database_identity_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "content_sha256": content_sha256,
                "metadata_sha256": dataset.identity_sha256,
            }
        ).encode("utf-8")
    ).hexdigest()
    return TaskDatabaseAuditResult(
        database_path=dataset.database_path,
        content_sha256=content_sha256,
        database_identity_sha256=database_identity_sha256,
        example_count=example_count,
        total_sequence_tokens=total_sequence_tokens,
        maximum_sequence_tokens=maximum_sequence_tokens,
        split_counts=dict(sorted(split_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        family_counts=dict(sorted(family_counts.items())),
        document_count=len(document_splits),
        audit_passed=True,
    )


class ResumableCurrentTaskPacking:
    def __init__(
        self,
        binding: NativeTokenizerBinding,
        admission_database: str | Path,
        source_catalog: str | Path,
        task_database: str | Path,
        *,
        max_sequence: int = 2048,
        semantic_wire: str = "full_v3",
        eligibility_root: str | Path | None = None,
        expected_eligibility_manifest_sha256: str | None = None,
        forced_split: str | None = None,
    ) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        if type(max_sequence) is not int or max_sequence < 64:
            raise ValueError("max_sequence must be an integer of at least 64")
        if semantic_wire not in {"full_v3", "compact_v2"}:
            raise ValueError("semantic_wire must be full_v3 or compact_v2")
        if forced_split not in {None, "train", "validation", "test"}:
            raise ValueError("forced_split must be train, validation, test or None")
        if (eligibility_root is None) != (expected_eligibility_manifest_sha256 is None):
            raise ValueError(
                "eligibility_root and expected eligibility manifest SHA-256 must be supplied together"
            )
        if expected_eligibility_manifest_sha256 is not None and (
            len(expected_eligibility_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_eligibility_manifest_sha256
            )
        ):
            raise ValueError("expected eligibility manifest SHA-256 is invalid")
        self._binding = binding
        self.admission_database = Path(admission_database).resolve()
        self.source_catalog = Path(source_catalog).resolve()
        self.database_path = Path(task_database).resolve()
        self._forced_split = forced_split
        if not self.admission_database.is_file():
            raise ValueError("admission database does not exist")
        if not self.source_catalog.is_file():
            raise ValueError("source catalog does not exist")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._packer = (
            CompactCurrentCorpusTaskPacker(binding, max_sequence=max_sequence)
            if semantic_wire == "compact_v2"
            else CurrentCorpusTaskPacker(binding, max_sequence=max_sequence)
        )
        self._admission_metadata, admitted_count, rejected_count = (
            self._admission_identity()
        )
        if self._admission_metadata.get("catalog_exhausted") != "true":
            raise ValueError("full admission is not complete")
        expected_catalog_sha256 = self._admission_metadata.get("source_catalog_sha256")
        if _sha256_file(self.source_catalog) != expected_catalog_sha256:
            raise ValueError("source catalog SHA-256 mismatch")
        self._eligible_ids: set[str] | None = None
        eligibility_manifest: dict[str, object] | None = None
        if eligibility_root is not None:
            self._eligible_ids, eligibility_manifest = self._load_eligibility(
                Path(eligibility_root).resolve(),
                str(expected_eligibility_manifest_sha256),
                str(expected_catalog_sha256),
            )
            eligibility_source_split = str(eligibility_manifest["source_split"])
            if (forced_split is None and eligibility_source_split != "train") or (
                forced_split is not None and eligibility_source_split == "train"
            ):
                raise ValueError(
                    "eligibility source split and task split policy mismatch"
                )
        identity_payload = {
            "metadata": self._admission_metadata,
            "admitted_count": admitted_count,
            "rejected_count": rejected_count,
        }
        admission_identity = hashlib.sha256(
            _canonical_json(identity_payload).encode("utf-8")
        ).hexdigest()
        self._expected_metadata = {
            "schema_version": TASK_DATABASE_SCHEMA,
            "binding_sha256": binding.binding_sha256,
            "tokenizer_sha256": binding.tokenizer_sha256,
            "vocabulary_size": str(binding.vocab_size),
            "maximum_sequence_tokens": str(max_sequence),
            "source_admission_identity_sha256": admission_identity,
            "source_catalog_sha256": str(expected_catalog_sha256),
            "review_manifest_sha256": self._admission_metadata[
                "review_manifest_sha256"
            ],
            "semantic_spec_sha256": self._admission_metadata["semantic_spec_sha256"],
            "parser_runtime_identity_sha256": self._admission_metadata[
                "parser_runtime_identity_sha256"
            ],
            "semantic_wire_schema_version": (
                COMPACT_SEMANTIC_WIRE_SCHEMA
                if semantic_wire == "compact_v2"
                else FULL_SEMANTIC_WIRE_SCHEMA
            ),
        }
        if eligibility_manifest is not None:
            self._expected_metadata.update(
                {
                    "audited_source_eligibility_manifest_sha256": str(
                        eligibility_manifest["manifest_sha256"]
                    ),
                    "eligibility_source_split": str(
                        eligibility_manifest["source_split"]
                    ),
                    "source_audit_identity_sha256": str(
                        eligibility_manifest["source_audit_identity_sha256"]
                    ),
                }
            )
            if eligibility_manifest["source_split"] == "train":
                self._expected_metadata.update(
                    {
                        "training_eligibility_manifest_sha256": str(
                            eligibility_manifest["manifest_sha256"]
                        ),
                        "task_split_policy": (
                            "frozen_training_eligibility_forced_train"
                        ),
                    }
                )
        if forced_split is not None:
            self._expected_metadata.update(
                {
                    "task_split_policy": "explicit_frozen_source_split",
                    "forced_task_split": forced_split,
                }
            )
        self._admitted_count = (
            len(self._eligible_ids)
            if self._eligible_ids is not None
            else admitted_count
        )
        self._initialize()

    def _load_eligibility(
        self,
        root: Path,
        expected_manifest_sha256: str,
        expected_catalog_sha256: str,
    ) -> tuple[set[str], dict[str, object]]:
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("training eligibility manifest is invalid") from error
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if (
            manifest.get("schema_version") != ELIGIBILITY_MANIFEST_SCHEMA
            or claimed
            != hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
            or claimed != expected_manifest_sha256
            or manifest.get("eligibility_complete") is not True
            or manifest.get("training_ready") is not False
            or manifest.get("source_catalog_sha256") != expected_catalog_sha256
            or manifest.get("source_split")
            not in {"train", "validation", "test", "reserved_heldout"}
        ):
            raise ValueError("training eligibility manifest lineage mismatch")
        try:
            content = (root / str(manifest["eligible_file"])).read_text(
                encoding="utf-8"
            )
        except OSError as error:
            raise ValueError("training eligibility inventory cannot be read") from error
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != manifest.get(
            "eligible_sha256"
        ):
            raise ValueError("training eligibility inventory SHA-256 mismatch")
        eligible_ids: set[str] = set()
        for line in split_jsonl_records(content):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("training eligibility inventory is invalid") from error
            document_id = row.get("document_id") if type(row) is dict else None
            if (
                type(row) is not dict
                or line != _canonical_json(row)
                or row.get("schema_version") != ELIGIBLE_DOCUMENT_SCHEMA
                or type(document_id) is not str
                or not document_id
                or document_id in eligible_ids
            ):
                raise ValueError("training eligibility inventory is invalid")
            eligible_ids.add(document_id)
        if len(eligible_ids) != manifest.get("eligible_document_count"):
            raise ValueError("training eligibility inventory count mismatch")
        admission = self._ro_connect(self.admission_database)
        try:
            metadata = dict(admission.execute("SELECT key,value FROM metadata"))
            identity = ResumableTrainingSemanticAudit._admission_identity(
                admission, metadata
            )
            admitted_ids = {
                str(row[0])
                for row in admission.execute(
                    "SELECT document_id FROM documents WHERE status='admitted'"
                )
            }
        finally:
            admission.close()
        if manifest.get(
            "source_admission_identity_sha256"
        ) != identity or not eligible_ids.issubset(admitted_ids):
            raise ValueError("training eligibility admission lineage mismatch")
        return eligible_ids, manifest

    @staticmethod
    def _ro_connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _admission_identity(self) -> tuple[dict[str, str], int, int]:
        connection = self._ro_connect(self.admission_database)
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("schema_version") != RESUMABLE_SCHEMA:
                raise ValueError("admission database schema mismatch")
            if (
                metadata.get("binding_sha256") != self._binding.binding_sha256
                or metadata.get("tokenizer_sha256") != self._binding.tokenizer_sha256
                or metadata.get("vocabulary_size") != str(self._binding.vocab_size)
                or metadata.get("parser_runtime_identity_sha256")
                != PARSER_RUNTIME_IDENTITY_SHA256
            ):
                raise ValueError("admission tokenizer lineage mismatch")
            counts = connection.execute(
                """SELECT SUM(CASE WHEN status='admitted' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END)
                   FROM documents"""
            ).fetchone()
        finally:
            connection.close()
        if counts is None:
            raise RuntimeError("admission database count query returned no row")
        return metadata, int(counts[0] or 0), int(counts[1] or 0)

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                       document_id TEXT PRIMARY KEY,
                       task_count INTEGER NOT NULL,
                       rejected_count INTEGER NOT NULL
                   )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
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
            connection.execute(
                "CREATE INDEX IF NOT EXISTS tasks_cell ON tasks(split, language, family)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS rejections (
                       rejection_id TEXT PRIMARY KEY,
                       document_id TEXT NOT NULL,
                       language TEXT NOT NULL,
                       record_index INTEGER NOT NULL,
                       family TEXT,
                       reason_code TEXT NOT NULL
                   )"""
            )
            existing = dict(connection.execute("SELECT key, value FROM metadata"))
            if not existing:
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES(?, ?)",
                    sorted(self._expected_metadata.items()),
                )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('packing_complete', 'false')"
                )
                connection.commit()
            else:
                actual = {key: existing.get(key) for key in self._expected_metadata}
                if actual != self._expected_metadata:
                    raise ValueError("resumable task packing lineage mismatch")
        finally:
            connection.close()

    def status(self, *, processed_this_run: int = 0) -> ResumableTaskPackingStatus:
        connection = self._connect()
        try:
            processed_documents = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            task_row = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM tasks"
            ).fetchone()
            rejected_count = connection.execute(
                "SELECT COUNT(*) FROM rejections"
            ).fetchone()[0]
            complete = connection.execute(
                "SELECT value FROM metadata WHERE key='packing_complete'"
            ).fetchone()[0]
        finally:
            connection.close()
        return ResumableTaskPackingStatus(
            database_path=self.database_path,
            processed_this_run=processed_this_run,
            processed_documents=int(processed_documents),
            task_count=int(task_row[0]),
            rejected_count=int(rejected_count),
            total_sequence_tokens=int(task_row[1]),
            packing_complete=complete == "true",
        )

    def process(
        self, *, max_documents: int | None = None
    ) -> ResumableTaskPackingStatus:
        if max_documents is not None and (
            type(max_documents) is not int or max_documents < 1
        ):
            raise ValueError("max_documents must be a positive integer or None")
        task_connection = self._connect()
        admission = self._ro_connect(self.admission_database)
        catalog = self._ro_connect(self.source_catalog)
        processed = 0
        try:
            completed_ids = {
                row[0]
                for row in task_connection.execute("SELECT document_id FROM documents")
            }
            rows = admission.execute(
                """SELECT document_id, content_sha256, language, relative_path,
                          semantic_spec_sha256, parser_runtime_identity_sha256,
                          canonical_sha256,
                          canonical_jsonl_zlib, semantic_ids_zlib,
                          semantic_record_count, semantic_token_count
                   FROM documents
                   WHERE status='admitted'
                   ORDER BY document_id"""
            )
            for admission_row in rows:
                document_id = str(admission_row[0])
                if (
                    self._eligible_ids is not None
                    and document_id not in self._eligible_ids
                ):
                    continue
                if document_id in completed_ids:
                    continue
                if max_documents is not None and processed >= max_documents:
                    break
                source_row = catalog.execute(
                    """SELECT canonical_language, content, relative_path, content_sha256
                       FROM documents WHERE document_id=?""",
                    (document_id,),
                ).fetchone()
                if source_row is None:
                    raise ValueError("admitted document is absent from source catalog")
                language, source, relative_path, source_sha256 = map(str, source_row)
                if (
                    language != admission_row[2]
                    or relative_path != admission_row[3]
                    or source_sha256 != admission_row[1]
                    or hashlib.sha256(source.encode("utf-8")).hexdigest()
                    != source_sha256
                ):
                    raise ValueError("source catalog document identity mismatch")
                if str(admission_row[5]) != PARSER_RUNTIME_IDENTITY_SHA256:
                    raise ValueError("admission parser runtime identity mismatch")
                canonical_jsonl = decompress_bounded(
                    admission_row[7],
                    maximum_output_bytes=MAX_CANONICAL_OUTPUT_BYTES,
                ).decode("utf-8")
                if (
                    hashlib.sha256(canonical_jsonl.encode("utf-8")).hexdigest()
                    != admission_row[6]
                ):
                    raise ValueError("admission canonical JSONL SHA-256 mismatch")
                expected_semantic_bytes = int(admission_row[10]) * 2
                semantic_ids = _decode_u16(
                    decompress_bounded(
                        admission_row[8],
                        maximum_output_bytes=expected_semantic_bytes,
                    )
                )
                if len(semantic_ids) != int(admission_row[10]):
                    raise ValueError("admission semantic token count mismatch")
                row = {
                    "document_id": document_id,
                    "language": language,
                    "relative_path": relative_path,
                    "source": source,
                    "source_sha256": source_sha256,
                    "binding_sha256": self._binding.binding_sha256,
                    "tokenizer_sha256": self._binding.tokenizer_sha256,
                    "semantic_spec_sha256": str(admission_row[4]),
                    "parser_runtime_identity_sha256": str(admission_row[5]),
                    "canonical_sha256": str(admission_row[6]),
                    "canonical_jsonl": canonical_jsonl,
                    "semantic_record_count": int(admission_row[9]),
                    "semantic_token_ids": semantic_ids,
                }
                examples, rejections = self._packer.pack_row(row)
                if self._forced_split is not None:
                    examples = [
                        {**item, "split": self._forced_split} for item in examples
                    ]
                elif self._eligible_ids is not None:
                    examples = [{**item, "split": "train"} for item in examples]
                task_connection.executemany(
                    """INSERT INTO tasks(
                           example_id, document_id, language, family, record_index, split,
                           input_ids_zlib, token_count, loss_start_index,
                           semantic_spec_sha256, parser_runtime_identity_sha256,
                           source_sha256
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [
                        (
                            item["example_id"],
                            item["document_id"],
                            item["language"],
                            item["family"],
                            item["record_index"],
                            item["split"],
                            zlib.compress(
                                _encode_u16(cast(list[int], item["input_ids"])), 6
                            ),
                            len(cast(list[int], item["input_ids"])),
                            item["loss_start_index"],
                            item["semantic_spec_sha256"],
                            item["parser_runtime_identity_sha256"],
                            item["source_sha256"],
                        )
                        for item in examples
                    ],
                )
                rejection_rows = []
                for item in rejections:
                    rejection_id = hashlib.sha256(
                        _canonical_json(item).encode("utf-8")
                    ).hexdigest()
                    rejection_rows.append(
                        (
                            rejection_id,
                            item["document_id"],
                            item["language"],
                            item["record_index"],
                            item["family"],
                            item["reason_code"],
                        )
                    )
                task_connection.executemany(
                    "INSERT INTO rejections VALUES(?,?,?,?,?,?)", rejection_rows
                )
                task_connection.execute(
                    "INSERT INTO documents VALUES(?,?,?)",
                    (document_id, len(examples), len(rejections)),
                )
                task_connection.commit()
                completed_ids.add(document_id)
                processed += 1
            processed_total = task_connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            if int(processed_total) == self._admitted_count:
                task_connection.execute(
                    "UPDATE metadata SET value='true' WHERE key='packing_complete'"
                )
                task_connection.commit()
        finally:
            catalog.close()
            admission.close()
            task_connection.close()
        return self.status(processed_this_run=processed)
