"""Bounded, split-exact curriculum selected only from audited eligible sources."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import sqlite3
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

from .bcnlt_adapter import MAX_CANONICAL_OUTPUT_BYTES
from .bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from .compact_semantic_wire import COMPACT_SEMANTIC_WIRE_SCHEMA
from .compact_task_packing import CompactCurrentCorpusTaskPacker
from .compression import decompress_bounded
from .current_task_packing import FAMILIES, CurrentCorpusTaskPacker
from .jsonl import split_jsonl_records
from .resumable_current_corpus import RESUMABLE_SCHEMA
from .resumable_task_packing import (
    FULL_SEMANTIC_WIRE_SCHEMA,
    TASK_DATABASE_SCHEMA,
    _canonical_json,
    _decode_u16,
    _encode_u16,
    _sha256_file,
    audit_task_database,
)
from .resumable_training_semantic_audit import ResumableTrainingSemanticAudit
from .tokenizer import NativeTokenizerBinding
from .training_eligibility import (
    ELIGIBILITY_MANIFEST_SCHEMA,
    ELIGIBLE_DOCUMENT_SCHEMA,
)

AUDITED_CURRICULUM_SCHEMA = "bosscrafts-audited-balanced-curriculum-v1"
SPLITS = ("train", "validation", "test")
_TARGET_BUCKET_LIMITS = (16, 32, 64, 128, 256, 512, 1024)


@dataclass(frozen=True, slots=True)
class _Candidate:
    example_id: str
    document_id: str
    language: str
    family: str
    record_index: int
    split: str
    input_ids_zlib: bytes
    token_count: int
    loss_start_index: int
    semantic_spec_sha256: str
    parser_runtime_identity_sha256: str
    source_sha256: str
    prompt_key: tuple[str, str, int]
    target_key: tuple[str, int]
    target_tokens: int


@dataclass(frozen=True, slots=True)
class AuditedBalancedCurriculumResult:
    database_path: Path
    database_identity_sha256: str
    policy_sha256: str
    example_count: int
    selected_document_count: int
    processed_document_count: int
    candidate_pool_count: int
    conflicting_prompt_group_count: int
    cross_split_prompt_group_count: int
    duplicate_rows_suppressed: int
    total_sequence_tokens: int
    cell_counts: dict[str, int]
    training_ready: bool


class _PromptClaimIndex:
    """Disk-backed global prompt/target index used before bounded selection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: sqlite3.Connection | None = None
        self._pending = 0

    def __enter__(self) -> Self:
        if self.path.exists():
            raise ValueError("prompt-claim scratch database already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                """CREATE TABLE prompt_claims(
                       family TEXT NOT NULL,prompt_sha256 TEXT NOT NULL,
                       loss_start_index INTEGER NOT NULL,split TEXT NOT NULL,
                       target_sha256 TEXT NOT NULL,target_tokens INTEGER NOT NULL,
                       PRIMARY KEY(family,prompt_sha256,loss_start_index,split,
                                   target_sha256,target_tokens)) WITHOUT ROWID"""
            )
        except BaseException:
            connection.close()
            self._remove_files()
            raise
        self._connection = connection
        return self

    def add(self, candidate: _Candidate) -> None:
        if self._connection is None:
            raise RuntimeError("prompt-claim index is not open")
        self._connection.execute(
            "INSERT OR IGNORE INTO prompt_claims VALUES(?,?,?,?,?,?)",
            (
                candidate.prompt_key[0],
                candidate.prompt_key[1],
                candidate.prompt_key[2],
                candidate.split,
                candidate.target_key[0],
                candidate.target_key[1],
            ),
        )
        self._pending += 1
        if self._pending >= 10_000:
            self._connection.commit()
            self._pending = 0

    def exclusions(
        self,
    ) -> tuple[set[tuple[str, str, int]], set[tuple[str, str, int]]]:
        if self._connection is None:
            raise RuntimeError("prompt-claim index is not open")
        self._connection.commit()
        self._pending = 0
        conflicting: set[tuple[str, str, int]] = set()
        cross_split: set[tuple[str, str, int]] = set()
        rows = self._connection.execute(
            """SELECT family,prompt_sha256,loss_start_index,
                      COUNT(DISTINCT target_sha256 || ':' || target_tokens),
                      COUNT(DISTINCT split)
                 FROM prompt_claims
                GROUP BY family,prompt_sha256,loss_start_index
               HAVING COUNT(DISTINCT target_sha256 || ':' || target_tokens) > 1
                   OR COUNT(DISTINCT split) > 1"""
        )
        for family, prompt_sha256, loss_start, target_count, split_count in rows:
            key = (str(family), str(prompt_sha256), int(loss_start))
            if int(target_count) > 1:
                conflicting.add(key)
            elif int(split_count) > 1:
                cross_split.add(key)
        return conflicting, cross_split

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._remove_files()

    def _remove_files(self) -> None:
        for path in (
            self.path,
            self.path.with_name(f"{self.path.name}-journal"),
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
        ):
            if path.exists():
                path.unlink()


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


def _bucket(target_tokens: int) -> int:
    for index, limit in enumerate(_TARGET_BUCKET_LIMITS):
        if target_tokens <= limit:
            return index
    return len(_TARGET_BUCKET_LIMITS)


def _push_candidate(
    heap: list[tuple[int, str, _Candidate]], candidate: _Candidate, limit: int
) -> None:
    entry = (-int(candidate.example_id, 16), candidate.example_id, candidate)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)


def _validate_map(name: str, value: Mapping[str, object]) -> None:
    if type(value) is not dict or set(value) != set(SPLITS):
        raise ValueError(f"{name} must define train, validation and test")


def _load_eligibility(
    admission_path: Path,
    eligibility_root: Path,
    expected_manifest_sha256: str,
    *,
    split: str,
    expected_catalog_sha256: str,
) -> tuple[set[str], dict[str, object], dict[str, str], str]:
    if not _is_sha256(expected_manifest_sha256):
        raise ValueError("eligibility manifest SHA-256 is invalid")
    admission = _ro_connect(admission_path)
    try:
        if str(admission.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise ValueError("source admission database integrity failed")
        metadata = dict(admission.execute("SELECT key,value FROM metadata"))
        if (
            metadata.get("schema_version") != RESUMABLE_SCHEMA
            or metadata.get("catalog_exhausted") != "true"
            or metadata.get("source_catalog_sha256") != expected_catalog_sha256
        ):
            raise ValueError("source admission lineage is incomplete")
        if split != "train" and metadata.get("frozen_source_split") != split:
            raise ValueError("source admission frozen split mismatch")
        admitted_ids = {
            str(row[0])
            for row in admission.execute(
                "SELECT document_id FROM documents WHERE status='admitted'"
            )
        }
        admission_identity = ResumableTrainingSemanticAudit._admission_identity(
            admission, metadata
        )
    finally:
        admission.close()
    try:
        raw_manifest = (eligibility_root / "manifest.json").read_text(encoding="utf-8")
        manifest = json.loads(raw_manifest)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("audited source eligibility manifest is invalid") from exc
    unsigned = dict(manifest) if type(manifest) is dict else {}
    claimed = unsigned.pop("manifest_sha256", None)
    if (
        type(manifest) is not dict
        or raw_manifest != f"{_canonical_json(manifest)}\n"
        or manifest.get("schema_version") != ELIGIBILITY_MANIFEST_SCHEMA
        or claimed
        != hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        or claimed != expected_manifest_sha256
        or manifest.get("eligibility_complete") is not True
        or manifest.get("training_ready") is not False
        or manifest.get("source_split") != split
        or manifest.get("source_catalog_sha256") != expected_catalog_sha256
        or manifest.get("source_admission_identity_sha256") != admission_identity
    ):
        raise ValueError("audited source eligibility lineage mismatch")
    try:
        eligible_text = (eligibility_root / str(manifest["eligible_file"])).read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise ValueError("audited source eligibility inventory cannot be read") from exc
    if hashlib.sha256(eligible_text.encode("utf-8")).hexdigest() != manifest.get(
        "eligible_sha256"
    ):
        raise ValueError("audited source eligibility inventory SHA-256 mismatch")
    eligible_ids: set[str] = set()
    for line in split_jsonl_records(eligible_text):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("audited source eligibility row is invalid") from exc
        document_id = row.get("document_id") if type(row) is dict else None
        if (
            type(row) is not dict
            or line != _canonical_json(row)
            or row.get("schema_version") != ELIGIBLE_DOCUMENT_SCHEMA
            or type(document_id) is not str
            or not document_id
            or document_id in eligible_ids
        ):
            raise ValueError("audited source eligibility row is invalid")
        eligible_ids.add(document_id)
    if (
        len(eligible_ids) != manifest.get("eligible_document_count")
        or not eligible_ids
        or not eligible_ids.issubset(admitted_ids)
    ):
        raise ValueError("audited source eligibility inventory mismatch")
    return eligible_ids, manifest, metadata, admission_identity


def build_audited_balanced_curriculum(
    binding: NativeTokenizerBinding,
    admission_databases: Mapping[str, str | Path],
    eligibility_roots: Mapping[str, str | Path],
    eligibility_manifest_sha256: Mapping[str, str],
    source_catalog: str | Path,
    destination_database: str | Path,
    *,
    languages: tuple[str, ...] = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256)),
    quotas: dict[str, int] | None = None,
    oversample_factor: int = 4,
    max_sequence: int = 2048,
    semantic_wire: str = "compact_v2",
) -> AuditedBalancedCurriculumResult:
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    _validate_map("admission_databases", admission_databases)
    _validate_map("eligibility_roots", eligibility_roots)
    _validate_map("eligibility_manifest_sha256", eligibility_manifest_sha256)
    if quotas is None:
        quotas = {"train": 512, "validation": 128, "test": 128}
    if (
        type(quotas) is not dict
        or set(quotas) != set(SPLITS)
        or any(
            type(value) is not int or not 8 <= value <= 100_000
            for value in quotas.values()
        )
    ):
        raise ValueError("curriculum quotas must define train, validation and test")
    if type(oversample_factor) is not int or not 2 <= oversample_factor <= 16:
        raise ValueError("oversample_factor must be between 2 and 16")
    if (
        type(languages) is not tuple
        or not languages
        or any(type(language) is not str for language in languages)
        or len(set(languages)) != len(languages)
        or set(languages) - set(CURRENT_RUNTIME_SPEC_SHA256)
    ):
        raise ValueError("curriculum languages are invalid")
    languages = tuple(sorted(languages))
    if type(max_sequence) is not int or max_sequence < 64:
        raise ValueError("max_sequence must be at least 64")
    if semantic_wire not in {"compact_v2", "full_v3"}:
        raise ValueError("semantic_wire must be compact_v2 or full_v3")
    catalog_path = Path(source_catalog).resolve()
    destination = Path(destination_database).resolve()
    if not catalog_path.is_file():
        raise ValueError("source catalog does not exist")
    if destination.exists():
        raise ValueError("destination curriculum already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    catalog_sha256 = _sha256_file(catalog_path)

    eligible_by_split: dict[str, set[str]] = {}
    manifests: dict[str, dict[str, object]] = {}
    metadata_by_split: dict[str, dict[str, str]] = {}
    admission_identities: dict[str, str] = {}
    for split in SPLITS:
        admission_path = Path(admission_databases[split]).resolve()
        eligibility_path = Path(eligibility_roots[split]).resolve()
        if not admission_path.is_file():
            raise ValueError(f"{split} source admission database does not exist")
        (
            eligible_by_split[split],
            manifests[split],
            metadata_by_split[split],
            admission_identities[split],
        ) = _load_eligibility(
            admission_path,
            eligibility_path,
            str(eligibility_manifest_sha256[split]),
            split=split,
            expected_catalog_sha256=catalog_sha256,
        )
    lineage_values = (
        "binding_sha256",
        "tokenizer_sha256",
        "vocabulary_size",
        "semantic_spec_sha256",
        "parser_runtime_identity_sha256",
        "review_manifest_sha256",
        "source_freeze_manifest_sha256",
        "semantic_audit_qualification_sha256",
    )
    for key in lineage_values:
        values = {metadata_by_split[split].get(key) for split in SPLITS}
        if len(values) != 1 or None in values:
            raise ValueError(f"source split lineage differs: {key}")
    training_metadata = metadata_by_split["train"]
    if (
        training_metadata["binding_sha256"] != binding.binding_sha256
        or training_metadata["tokenizer_sha256"] != binding.tokenizer_sha256
        or training_metadata["vocabulary_size"] != str(binding.vocab_size)
        or training_metadata["parser_runtime_identity_sha256"]
        != PARSER_RUNTIME_IDENTITY_SHA256
        or training_metadata["semantic_spec_sha256"]
        != _canonical_json(CURRENT_RUNTIME_SPEC_SHA256)
    ):
        raise ValueError("source split tokenizer or semantic lineage mismatch")

    packer = (
        CompactCurrentCorpusTaskPacker(binding, max_sequence=max_sequence)
        if semantic_wire == "compact_v2"
        else CurrentCorpusTaskPacker(binding, max_sequence=max_sequence)
    )
    pool_heaps: dict[
        tuple[str, str, str],
        tuple[
            list[list[tuple[int, str, _Candidate]]], list[tuple[int, str, _Candidate]]
        ],
    ] = {}
    processed_documents = 0
    prompt_scratch = destination.with_name(f".{destination.name}.prompt-claims.tmp")
    with _PromptClaimIndex(prompt_scratch) as prompt_claims:
        catalog = _ro_connect(catalog_path)
        try:
            for split in SPLITS:
                admission_path = Path(admission_databases[split]).resolve()
                admission = _ro_connect(admission_path)
                seen_eligible: set[str] = set()
                try:
                    for language in languages:
                        rows = admission.execute(
                            """SELECT document_id,content_sha256,relative_path,
                                  semantic_spec_sha256,parser_runtime_identity_sha256,
                                  canonical_sha256,canonical_jsonl_zlib,semantic_ids_zlib,
                                  semantic_record_count,semantic_token_count
                           FROM documents WHERE status='admitted' AND language=?
                           ORDER BY document_id""",
                            (language,),
                        )
                        for admission_row in rows:
                            document_id = str(admission_row[0])
                            if document_id not in eligible_by_split[split]:
                                continue
                            seen_eligible.add(document_id)
                            source_row = catalog.execute(
                                """SELECT canonical_language,content,relative_path,content_sha256
                               FROM documents WHERE document_id=?""",
                                (document_id,),
                            ).fetchone()
                            if source_row is None:
                                raise ValueError(
                                    "eligible document is absent from source catalog"
                                )
                            source_language, source, relative_path, source_sha = map(
                                str, source_row
                            )
                            if (
                                source_language != language
                                or relative_path != admission_row[2]
                                or source_sha != admission_row[1]
                                or hashlib.sha256(source.encode("utf-8")).hexdigest()
                                != source_sha
                                or admission_row[3]
                                != CURRENT_RUNTIME_SPEC_SHA256[language]
                                or admission_row[4] != PARSER_RUNTIME_IDENTITY_SHA256
                            ):
                                raise ValueError("eligible source lineage mismatch")
                            canonical_jsonl = decompress_bounded(
                                admission_row[6],
                                maximum_output_bytes=MAX_CANONICAL_OUTPUT_BYTES,
                            ).decode("utf-8")
                            if (
                                hashlib.sha256(
                                    canonical_jsonl.encode("utf-8")
                                ).hexdigest()
                                != admission_row[5]
                            ):
                                raise ValueError(
                                    "eligible canonical payload SHA-256 mismatch"
                                )
                            expected_semantic_bytes = int(admission_row[9]) * 2
                            semantic_ids = _decode_u16(
                                decompress_bounded(
                                    admission_row[7],
                                    maximum_output_bytes=expected_semantic_bytes,
                                )
                            )
                            if len(semantic_ids) != int(admission_row[9]):
                                raise ValueError(
                                    "eligible semantic token count mismatch"
                                )
                            examples, _ = packer.pack_row(
                                {
                                    "document_id": document_id,
                                    "language": language,
                                    "relative_path": relative_path,
                                    "source": source,
                                    "source_sha256": source_sha,
                                    "binding_sha256": binding.binding_sha256,
                                    "tokenizer_sha256": binding.tokenizer_sha256,
                                    "semantic_spec_sha256": str(admission_row[3]),
                                    "parser_runtime_identity_sha256": str(
                                        admission_row[4]
                                    ),
                                    "canonical_sha256": str(admission_row[5]),
                                    "canonical_jsonl": canonical_jsonl,
                                    "semantic_record_count": int(admission_row[8]),
                                    "semantic_token_ids": semantic_ids,
                                }
                            )
                            processed_documents += 1
                            for item in examples:
                                family = item.get("family")
                                raw_ids_value = item.get("input_ids")
                                loss_start_value = item.get("loss_start_index")
                                record_index_value = item.get("record_index")
                                example_id_value = item.get("example_id")
                                if family not in FAMILIES:
                                    raise ValueError("packed task family is invalid")
                                if type(raw_ids_value) is not list:
                                    raise ValueError(
                                        "packed task input IDs are invalid"
                                    )
                                input_ids = cast(list[int], raw_ids_value)
                                if (
                                    type(loss_start_value) is not int
                                    or not 0 <= loss_start_value < len(input_ids)
                                    or type(record_index_value) is not int
                                    or record_index_value < 0
                                    or not _is_sha256(example_id_value)
                                ):
                                    raise ValueError(
                                        "packed task identity or loss boundary is invalid"
                                    )
                                cell = (split, language, family)
                                raw_ids = _encode_u16(input_ids)
                                loss_start = loss_start_value
                                target_tokens = len(input_ids) - loss_start
                                candidate = _Candidate(
                                    example_id=cast(str, example_id_value),
                                    document_id=document_id,
                                    language=language,
                                    family=family,
                                    record_index=record_index_value,
                                    split=split,
                                    input_ids_zlib=zlib.compress(raw_ids, 6),
                                    token_count=len(input_ids),
                                    loss_start_index=loss_start,
                                    semantic_spec_sha256=str(
                                        item["semantic_spec_sha256"]
                                    ),
                                    parser_runtime_identity_sha256=str(
                                        item["parser_runtime_identity_sha256"]
                                    ),
                                    source_sha256=source_sha,
                                    prompt_key=(
                                        family,
                                        hashlib.sha256(
                                            raw_ids[: loss_start * 2]
                                        ).hexdigest(),
                                        loss_start,
                                    ),
                                    target_key=(
                                        hashlib.sha256(
                                            raw_ids[loss_start * 2 :]
                                        ).hexdigest(),
                                        target_tokens,
                                    ),
                                    target_tokens=target_tokens,
                                )
                                prompt_claims.add(candidate)
                                bucket_heaps, fallback_heap = pool_heaps.setdefault(
                                    cell, ([[] for _ in range(8)], [])
                                )
                                bucket_limit = math.ceil(
                                    quotas[split] * oversample_factor / 8
                                )
                                _push_candidate(
                                    bucket_heaps[_bucket(target_tokens)],
                                    candidate,
                                    bucket_limit,
                                )
                                _push_candidate(
                                    fallback_heap,
                                    candidate,
                                    quotas[split] * oversample_factor,
                                )
                    if seen_eligible != eligible_by_split[split]:
                        raise ValueError(
                            f"{split} eligibility inventory was not fully processed"
                        )
                finally:
                    admission.close()
        finally:
            catalog.close()
        conflicting_prompt_keys, cross_split_prompt_keys = prompt_claims.exclusions()

    expected_cells = {
        (split, language, family): quotas[split]
        for split in SPLITS
        for language in languages
        for family in FAMILIES
    }
    if set(pool_heaps) != set(expected_cells):
        missing = sorted(set(expected_cells) - set(pool_heaps))
        raise ValueError(f"audited candidate pool lacks cells: {missing}")
    candidate_by_id: dict[str, _Candidate] = {}
    for bucket_heaps, fallback_heap in pool_heaps.values():
        for heap in (*bucket_heaps, fallback_heap):
            for _, _, candidate in heap:
                candidate_by_id[candidate.example_id] = candidate

    prompt_groups: dict[tuple[str, str, int], list[_Candidate]] = defaultdict(list)
    for candidate in candidate_by_id.values():
        prompt_groups[candidate.prompt_key].append(candidate)
    clean_by_cell: dict[tuple[str, str, str], list[_Candidate]] = defaultdict(list)
    conflicting_groups = len(conflicting_prompt_keys)
    cross_split_groups = len(cross_split_prompt_keys)
    duplicate_rows = 0
    for candidates in prompt_groups.values():
        prompt_key = candidates[0].prompt_key
        if prompt_key in conflicting_prompt_keys:
            continue
        if prompt_key in cross_split_prompt_keys:
            continue
        chosen_candidate = min(candidates, key=lambda candidate: candidate.example_id)
        duplicate_rows += len(candidates) - 1
        clean_by_cell[
            (chosen_candidate.split, chosen_candidate.language, chosen_candidate.family)
        ].append(chosen_candidate)

    selected: list[_Candidate] = []
    cell_counts: Counter[tuple[str, str, str]] = Counter()
    for cell, target in sorted(expected_cells.items()):
        candidates = clean_by_cell.get(cell, [])
        by_bucket: dict[int, list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            by_bucket[_bucket(candidate.target_tokens)].append(candidate)
        selected_ids: set[str] = set()
        per_bucket = target // 8
        remainder = target % 8
        for bucket_index in range(8):
            bucket_target = per_bucket + (1 if bucket_index < remainder else 0)
            for candidate in sorted(
                by_bucket.get(bucket_index, []), key=lambda item: item.example_id
            )[:bucket_target]:
                selected_ids.add(candidate.example_id)
        if len(selected_ids) < target:
            for candidate in sorted(candidates, key=lambda item: item.example_id):
                selected_ids.add(candidate.example_id)
                if len(selected_ids) == target:
                    break
        if len(selected_ids) != target:
            raise ValueError(
                "audited balanced curriculum quota could not be filled: "
                f"{'|'.join(cell)} needs {target}, has {len(selected_ids)}"
            )
        chosen_candidates = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.example_id in selected_ids
            ),
            key=lambda item: item.example_id,
        )
        selected.extend(chosen_candidates)
        cell_counts[cell] = len(chosen_candidates)

    semantic_wire_schema = (
        COMPACT_SEMANTIC_WIRE_SCHEMA
        if semantic_wire == "compact_v2"
        else FULL_SEMANTIC_WIRE_SCHEMA
    )
    policy = {
        "schema_version": AUDITED_CURRICULUM_SCHEMA,
        "languages": languages,
        "task_families": FAMILIES,
        "quotas_per_language_family": dict(sorted(quotas.items())),
        "oversample_factor": oversample_factor,
        "target_token_buckets": [
            "1-16",
            "17-32",
            "33-64",
            "65-128",
            "129-256",
            "257-512",
            "513-1024",
            "1025+",
        ],
        "selection": (
            "lowest content-addressed example IDs within target-length buckets; "
            "deterministic fallback; exclude contradictory and cross-split prompts; "
            "deduplicate identical prompts"
        ),
        "semantic_wire_schema_version": semantic_wire_schema,
        "maximum_sequence_tokens": max_sequence,
        "source_admission_identity_sha256": dict(sorted(admission_identities.items())),
        "source_eligibility_manifest_sha256": {
            split: manifests[split]["manifest_sha256"] for split in SPLITS
        },
    }
    policy_sha = hashlib.sha256(_canonical_json(policy).encode("utf-8")).hexdigest()
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    output = sqlite3.connect(temporary)
    try:
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
        output.execute("CREATE INDEX tasks_cell ON tasks(split,language,family)")
        aggregate_admission_identity = hashlib.sha256(
            _canonical_json(admission_identities).encode("utf-8")
        ).hexdigest()
        aggregate_audit_identity = hashlib.sha256(
            _canonical_json(
                {
                    split: manifests[split]["source_audit_identity_sha256"]
                    for split in SPLITS
                }
            ).encode("utf-8")
        ).hexdigest()
        output_metadata = {
            "schema_version": TASK_DATABASE_SCHEMA,
            "packing_complete": "true",
            "binding_sha256": binding.binding_sha256,
            "tokenizer_sha256": binding.tokenizer_sha256,
            "vocabulary_size": str(binding.vocab_size),
            "maximum_sequence_tokens": str(max_sequence),
            "semantic_spec_sha256": training_metadata["semantic_spec_sha256"],
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "source_catalog_sha256": catalog_sha256,
            "source_admission_identity_sha256": aggregate_admission_identity,
            "source_admission_identities_json": _canonical_json(admission_identities),
            "source_audit_identity_sha256": aggregate_audit_identity,
            "audited_source_eligibility_manifests_json": _canonical_json(
                {split: manifests[split]["manifest_sha256"] for split in SPLITS}
            ),
            "training_eligibility_manifest_sha256": str(
                manifests["train"]["manifest_sha256"]
            ),
            "review_manifest_sha256": training_metadata["review_manifest_sha256"],
            "source_freeze_manifest_sha256": training_metadata[
                "source_freeze_manifest_sha256"
            ],
            "semantic_audit_qualification_sha256": training_metadata[
                "semantic_audit_qualification_sha256"
            ],
            "semantic_wire_schema_version": semantic_wire_schema,
            "task_split_policy": "exact_frozen_source_split",
            "curriculum_schema_version": AUDITED_CURRICULUM_SCHEMA,
            "curriculum_policy_sha256": policy_sha,
        }
        output.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)",
            sorted(output_metadata.items()),
        )
        output.executemany(
            "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    item.example_id,
                    item.document_id,
                    item.language,
                    item.family,
                    item.record_index,
                    item.split,
                    item.input_ids_zlib,
                    item.token_count,
                    item.loss_start_index,
                    item.semantic_spec_sha256,
                    item.parser_runtime_identity_sha256,
                    item.source_sha256,
                )
                for item in sorted(selected, key=lambda candidate: candidate.example_id)
            ],
        )
        output.commit()
    except BaseException:
        output.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        output.close()
    try:
        audit = audit_task_database(binding, temporary)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.replace(destination)
    return AuditedBalancedCurriculumResult(
        database_path=destination,
        database_identity_sha256=audit.database_identity_sha256,
        policy_sha256=policy_sha,
        example_count=audit.example_count,
        selected_document_count=len({candidate.document_id for candidate in selected}),
        processed_document_count=processed_documents,
        candidate_pool_count=len(candidate_by_id),
        conflicting_prompt_group_count=conflicting_groups,
        cross_split_prompt_group_count=cross_split_groups,
        duplicate_rows_suppressed=duplicate_rows,
        total_sequence_tokens=sum(candidate.token_count for candidate in selected),
        cell_counts={"|".join(cell): cell_counts[cell] for cell in sorted(cell_counts)},
        training_ready=False,
    )
