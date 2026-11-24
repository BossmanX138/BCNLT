from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import zlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any, Iterable

from bcnlt.adapters.carousel import BCNLTMultilingualAdapter
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine
from bcnlt.semantic_record import SemanticTokenRecord, records_from_assessments

from .admission import (
    AdmissionDecision,
    AdmissionPolicy,
    AdmissionTokenizer,
    _decision_id,
    _document_semantic_hash,
    _encoded_length,
    _gate_failures,
    _identifier_shape,
    _literal_type,
    _normalized_source_hash,
    _pair_hash,
    _precheck_candidate,
    _scope_depth,
    _semantic_payload_hash,
    assign_split,
)
from .curriculum import CurriculumRecord
from .database import SourceDocumentRecord


NEAR_DUPLICATE_SIGNATURE_SIZE = 8
NEAR_DUPLICATE_BUCKET_LIMIT = 128


@dataclass(frozen=True)
class DocumentSemanticSummary:
    event: dict[str, int]
    scope_depth: dict[str, int]
    control_flow: dict[str, int]
    identifier_shape: dict[str, int]
    literal_type: dict[str, int]
    uncertainty: dict[str, int]
    semantic_units: int
    unknown_events: int


@dataclass(frozen=True)
class StreamingValidationResult:
    accepted_record_count: int
    decision_count: int
    split_manifest: dict[str, str]
    distributions: dict[str, Any]
    failure_codes: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "bcnlt-streaming-corpus-validation-v1",
            "passed": self.passed,
            "failure_codes": list(self.failure_codes),
            "accepted_record_count": self.accepted_record_count,
            "decision_count": self.decision_count,
            "split_manifest": dict(sorted(self.split_manifest.items())),
            "distributions": self.distributions,
        }


class _DiskDigestSet:
    def __init__(self, connection: sqlite3.Connection, kind: str) -> None:
        self.connection = connection
        self.kind = kind

    def __contains__(self, digest: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM seen_digests WHERE kind = ? AND digest = ?",
                (self.kind, digest),
            ).fetchone()
            is not None
        )

    def add(self, digest: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO seen_digests(kind, digest) VALUES (?, ?)",
            (self.kind, digest),
        )


def validate_corpus_streaming(
    documents: Iterable[SourceDocumentRecord],
    candidates: Iterable[CurriculumRecord],
    *,
    candidate_count: int,
    tokenizer: AdmissionTokenizer,
    policy: AdmissionPolicy,
    accepted_handle: IO[str],
    decisions_handle: IO[str],
    state_database: str | Path,
    progress_interval: int = 10_000,
) -> StreamingValidationResult:
    document_list = tuple(documents)
    document_map = {document.document_id: document for document in document_list}
    split_manifest, holdout_documents, summaries = _document_context(
        document_list,
        policy.holdout_hashes,
    )
    document_limit = max(1, int(candidate_count * policy.maximum_document_fraction))
    template_limit = max(
        1,
        int(candidate_count * policy.maximum_template_family_fraction),
    )

    state_path = Path(state_database)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(
        """
        CREATE TABLE seen_digests (
            kind TEXT NOT NULL,
            digest TEXT NOT NULL,
            PRIMARY KEY (kind, digest)
        ) WITHOUT ROWID;
        CREATE TABLE accepted_shingles (
            accepted_index INTEGER PRIMARY KEY,
            payload BLOB NOT NULL
        );
        """
    )
    connection.execute("BEGIN")
    seen_record_ids = _DiskDigestSet(connection, "record")
    seen_pairs = _DiskDigestSet(connection, "pair")
    seen_semantic = _DiskDigestSet(connection, "semantic")
    near_buckets: dict[str, dict[bytes, deque[int]]] = defaultdict(dict)
    document_counts: Counter[str] = Counter()
    template_counts: Counter[tuple[str, str]] = Counter()
    accepted_document_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    records_by_language: Counter[str] = Counter()
    family_by_language: dict[str, Counter[str]] = defaultdict(Counter)
    encoded_tokens_by_language: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    duplicate_reasons: Counter[str] = Counter()
    accepted_count = 0
    decision_count = 0

    try:
        for ordinal, candidate in enumerate(candidates):
            source_ids = candidate.source_document_ids
            assigned_splits = {
                split_manifest.get(source_id, "unknown") for source_id in source_ids
            }
            split = next(iter(assigned_splits)) if len(assigned_splits) == 1 else "conflict"
            reason = _precheck_candidate(
                candidate,
                document_map,
                holdout_documents,
                split,
                assigned_splits,
                seen_record_ids,  # type: ignore[arg-type]
            )
            pair_hash = ""
            semantic_key = ""
            shingle_hashes: tuple[int, ...] = ()
            encoded_length = 0
            if reason is None:
                pair_hash = _pair_hash(candidate)
                semantic_hash = _semantic_payload_hash(candidate)
                semantic_key = (
                    f"{candidate.family}:{semantic_hash}" if semantic_hash else ""
                )
                shingle_hashes = _hashed_five_grams(
                    candidate.prompt + "\n" + candidate.target
                )
                if pair_hash in seen_pairs:
                    reason = "exact_pair_duplicate"
                elif semantic_key and semantic_key in seen_semantic:
                    reason = "canonical_semantic_duplicate"
                elif _is_near_duplicate(
                    candidate.family,
                    shingle_hashes,
                    near_buckets,
                    connection,
                    policy.near_duplicate_threshold,
                ):
                    reason = "near_duplicate_pair"
                elif any(
                    document_counts[source_id] >= document_limit
                    for source_id in source_ids
                ):
                    reason = "document_cap_exceeded"
                elif (
                    template_counts[(candidate.family, candidate.template_id)]
                    >= template_limit
                ):
                    reason = "template_family_cap_exceeded"
                else:
                    encoded_length = _encoded_length(candidate, tokenizer)
                    if encoded_length > policy.maximum_sequence_tokens:
                        reason = "sequence_too_long"

            accepted = reason is None
            reason_code = "accepted" if accepted else str(reason)
            decision = AdmissionDecision(
                decision_id=_decision_id(candidate.record_id, ordinal, reason_code, split),
                record_id=candidate.record_id,
                source_document_ids=source_ids,
                accepted=accepted,
                reason_code=reason_code,
                split=split,
                language=candidate.language,
            )
            decisions_handle.write(
                json.dumps(decision.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
            decision_count += 1
            seen_record_ids.add(candidate.record_id)
            if not accepted:
                rejection_reasons[reason_code] += 1
                if "duplicate" in reason_code:
                    duplicate_reasons[reason_code] += 1
            else:
                admitted = replace(candidate, split=split)
                accepted_handle.write(
                    json.dumps(admitted.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                )
                seen_pairs.add(pair_hash)
                if semantic_key:
                    seen_semantic.add(semantic_key)
                connection.execute(
                    "INSERT INTO accepted_shingles(accepted_index, payload) VALUES (?, ?)",
                    (accepted_count, _pack_hashes(shingle_hashes)),
                )
                _remember_near_duplicate(
                    candidate.family,
                    shingle_hashes,
                    accepted_count,
                    near_buckets,
                )
                accepted_count += 1
                accepted_document_ids.update(source_ids)
                family_counts[candidate.family] += 1
                split_counts[split] += 1
                records_by_language[candidate.language] += 1
                family_by_language[candidate.language][candidate.family] += 1
                encoded_tokens_by_language[candidate.language] += encoded_length
                for source_id in source_ids:
                    document_counts[source_id] += 1
                template_counts[(candidate.family, candidate.template_id)] += 1

            if decision_count % 1_000 == 0:
                connection.commit()
                connection.execute("BEGIN")
            if progress_interval and decision_count % progress_interval == 0:
                print(
                    json.dumps(
                        {
                            "candidates_processed": decision_count,
                            "candidates_total": candidate_count,
                            "accepted": accepted_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        connection.commit()
    finally:
        connection.close()

    encoded_tokens = sum(encoded_tokens_by_language.values())
    distributions = _streaming_distributions(
        accepted_document_ids=accepted_document_ids,
        documents=document_map,
        summaries=summaries,
        accepted_count=accepted_count,
        family_counts=family_counts,
        split_counts=split_counts,
        records_by_language=records_by_language,
        family_by_language=family_by_language,
        encoded_tokens=encoded_tokens,
        encoded_tokens_by_language=encoded_tokens_by_language,
        rejection_reasons=rejection_reasons,
        duplicate_reasons=duplicate_reasons,
        holdout_collisions=len(holdout_documents),
    )
    failures = _gate_failures(accepted_count, encoded_tokens, distributions, policy)
    return StreamingValidationResult(
        accepted_record_count=accepted_count,
        decision_count=decision_count,
        split_manifest=split_manifest,
        distributions=distributions,
        failure_codes=tuple(failures),
        passed=not failures,
    )


def _document_context(
    documents: tuple[SourceDocumentRecord, ...],
    holdout_hashes: frozenset[str],
) -> tuple[
    dict[str, str],
    set[str],
    dict[str, DocumentSemanticSummary],
]:
    parent = {document.document_id: document.document_id for document in documents}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    summaries: dict[str, DocumentSemanticSummary] = {}
    semantic_hashes: dict[str, str] = {}
    try:
        for index, document in enumerate(documents, start=1):
            parsed = adapter.parse(
                document.content,
                language=document.canonical_language,
                path=document.relative_path,
                dialect=document.dialect,
            )
            extraction = adapter.extract(parsed)
            records = records_from_assessments(extraction.assessments)
            semantic_hashes[document.document_id] = _document_semantic_hash(records)
            summaries[document.document_id] = _summarize_records(records)
            if index % 1_000 == 0:
                print(
                    json.dumps(
                        {
                            "validation_documents_processed": index,
                            "validation_documents_total": len(documents),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        if isinstance(adapter.engine, TreeSitterSemanticEngine):
            adapter.engine.close()

    fingerprint_owner: dict[str, str] = {}
    for document in documents:
        fingerprints = {
            "content:" + document.content_sha256,
            "normalized:" + _normalized_source_hash(document.content),
            "semantic:" + semantic_hashes[document.document_id],
        }
        for fingerprint in fingerprints:
            owner = fingerprint_owner.setdefault(fingerprint, document.document_id)
            union(document.document_id, owner)

    groups: dict[str, list[SourceDocumentRecord]] = defaultdict(list)
    for document in documents:
        groups[find(document.document_id)].append(document)
    manifest: dict[str, str] = {}
    holdout_documents: set[str] = set()
    for members in groups.values():
        group_holdout = any(
            member.content_sha256 in holdout_hashes for member in members
        )
        identity = min(
            f"{member.repository_identity}:{_normalized_source_hash(member.content)}"
            for member in members
        )
        split = "evaluation_holdout" if group_holdout else assign_split(identity)
        for member in members:
            manifest[member.document_id] = split
            if group_holdout:
                holdout_documents.add(member.document_id)
    return manifest, holdout_documents, summaries


def _summarize_records(
    records: tuple[SemanticTokenRecord, ...],
) -> DocumentSemanticSummary:
    event: Counter[str] = Counter()
    scope_depth: Counter[str] = Counter()
    control_flow: Counter[str] = Counter()
    identifier_shape: Counter[str] = Counter()
    literal_type: Counter[str] = Counter()
    uncertainty: Counter[str] = Counter()
    for record in records:
        event[record.event] += 1
        scope_depth[str(_scope_depth(record.scope))] += 1
        uncertainty[record.uncertainty_band.value] += 1
        if record.event in {"BRANCH", "LOOP", "RAISE", "RETURN"}:
            control_flow[record.event.casefold()] += 1
        if record.event in {"DEFINE", "REFERENCE", "CALL", "MEMBER_ACCESS"} and record.surface:
            identifier_shape[_identifier_shape(record.surface)] += 1
        if record.event == "LITERAL":
            literal_type[_literal_type(record)] += 1
    return DocumentSemanticSummary(
        event=dict(event),
        scope_depth=dict(scope_depth),
        control_flow=dict(control_flow),
        identifier_shape=dict(identifier_shape),
        literal_type=dict(literal_type),
        uncertainty=dict(uncertainty),
        semantic_units=len(records),
        unknown_events=sum(record.event == "UNKNOWN_SEMANTIC" for record in records),
    )


def _streaming_distributions(
    *,
    accepted_document_ids: set[str],
    documents: dict[str, SourceDocumentRecord],
    summaries: dict[str, DocumentSemanticSummary],
    accepted_count: int,
    family_counts: Counter[str],
    split_counts: Counter[str],
    records_by_language: Counter[str],
    family_by_language: dict[str, Counter[str]],
    encoded_tokens: int,
    encoded_tokens_by_language: Counter[str],
    rejection_reasons: Counter[str],
    duplicate_reasons: Counter[str],
    holdout_collisions: int,
) -> dict[str, Any]:
    event: Counter[str] = Counter()
    scope_depth: Counter[str] = Counter()
    control_flow: Counter[str] = Counter()
    identifier_shape: Counter[str] = Counter()
    literal_type: Counter[str] = Counter()
    uncertainty: Counter[str] = Counter()
    semantic_by_language: dict[str, Counter[str]] = defaultdict(Counter)
    source: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    for source_id in accepted_document_ids:
        document = documents[source_id]
        summary = summaries[source_id]
        event.update(summary.event)
        scope_depth.update(summary.scope_depth)
        control_flow.update(summary.control_flow)
        identifier_shape.update(summary.identifier_shape)
        literal_type.update(summary.literal_type)
        uncertainty.update(summary.uncertainty)
        semantic_by_language[document.canonical_language]["total"] += summary.semantic_units
        semantic_by_language[document.canonical_language]["unknown"] += summary.unknown_events
        source[document.source_name] += 1
        license_counts[document.license_spdx] += 1

    language: dict[str, dict[str, object]] = {}
    for language_id in sorted(set(records_by_language) | set(semantic_by_language)):
        semantic_total = semantic_by_language[language_id]["total"]
        unknown = semantic_by_language[language_id]["unknown"]
        language[language_id] = {
            "records": records_by_language[language_id],
            "encoded_tokens": encoded_tokens_by_language[language_id],
            "family": dict(sorted(family_by_language[language_id].items())),
            "semantic_units": semantic_total,
            "common_core_coverage": (
                (semantic_total - unknown) / semantic_total if semantic_total else 0.0
            ),
            "unknown_event_fraction": unknown / semantic_total if semantic_total else 1.0,
        }
    return {
        "family": dict(sorted(family_counts.items())),
        "event": dict(sorted(event.items())),
        "scope_depth": dict(sorted(scope_depth.items())),
        "control_flow": dict(sorted(control_flow.items())),
        "identifier_shape": dict(sorted(identifier_shape.items())),
        "literal_type": dict(sorted(literal_type.items())),
        "uncertainty": dict(sorted(uncertainty.items())),
        "source": dict(sorted(source.items())),
        "license": dict(sorted(license_counts.items())),
        "split": dict(sorted(split_counts.items())),
        "rejection_reason": dict(sorted(rejection_reasons.items())),
        "duplicate": dict(sorted(duplicate_reasons.items())),
        "encoded_tokens": encoded_tokens,
        "language": language,
        "holdout_collisions": holdout_collisions,
        "accepted_record_count": accepted_count,
    }


def _hashed_five_grams(value: str) -> tuple[int, ...]:
    import re

    words = tuple(re.findall(r"[A-Za-z0-9_]+|\S", value.casefold()))
    if not words:
        return ()
    width = min(5, len(words))
    return tuple(
        sorted(
            {
                int.from_bytes(
                    hashlib.blake2b(
                        "\x1f".join(words[index : index + width]).encode("utf-8"),
                        digest_size=8,
                    ).digest(),
                    "big",
                )
                for index in range(len(words) - width + 1)
            }
        )
    )


def _signature_values(hashes: tuple[int, ...]) -> tuple[bytes, ...]:
    return tuple(
        value.to_bytes(8, "big")
        for value in hashes[:NEAR_DUPLICATE_SIGNATURE_SIZE]
    )


def _remember_near_duplicate(
    family: str,
    hashes: tuple[int, ...],
    accepted_index: int,
    buckets: dict[str, dict[bytes, deque[int]]],
) -> None:
    family_buckets = buckets[family]
    for value in _signature_values(hashes):
        bucket = family_buckets.get(value)
        if bucket is None:
            bucket = deque(maxlen=NEAR_DUPLICATE_BUCKET_LIMIT)
            family_buckets[value] = bucket
        bucket.append(accepted_index)


def _is_near_duplicate(
    family: str,
    hashes: tuple[int, ...],
    buckets: dict[str, dict[bytes, deque[int]]],
    connection: sqlite3.Connection,
    threshold: float,
) -> bool:
    family_buckets = buckets.get(family, {})
    candidate_indexes = {
        index
        for value in _signature_values(hashes)
        for index in family_buckets.get(value, ())
    }
    if not candidate_indexes:
        return False
    current = set(hashes)
    indexes = sorted(candidate_indexes)
    for start in range(0, len(indexes), 500):
        chunk = indexes[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"SELECT payload FROM accepted_shingles WHERE accepted_index IN ({placeholders})",
            chunk,
        )
        for (payload,) in rows:
            previous = set(_unpack_hashes(bytes(payload)))
            if not current and not previous:
                return True
            if len(current & previous) / len(current | previous) >= threshold:
                return True
    return False


def _pack_hashes(values: tuple[int, ...]) -> bytes:
    if not values:
        return b""
    packed = struct.pack(f">{len(values)}Q", *values)
    return zlib.compress(packed, level=1)


def _unpack_hashes(payload: bytes) -> tuple[int, ...]:
    if not payload:
        return ()
    raw = zlib.decompress(payload)
    return struct.unpack(f">{len(raw) // 8}Q", raw)
