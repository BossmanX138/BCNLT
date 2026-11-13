from __future__ import annotations

import ast
import hashlib
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Protocol, Sequence

from bcnlt.adapters.python import PythonAdapter
from bcnlt.canonical import canonical_record_bytes, record_from_canonical_bytes
from bcnlt.semantic_record import SemanticTokenRecord, records_from_assessments

from .curriculum import CurriculumRecord
from .database import SourceDocumentRecord


_REQUIRED_PROVENANCE = frozenset(
    {
        "source_provider",
        "source_name",
        "source_revision",
        "canonical_language",
        "repository_identity",
        "relative_path",
        "content_sha256",
        "license_spdx",
        "ownership",
        "attribution",
        "upstream_url",
    }
)


class AdmissionTokenizer(Protocol):
    def encode_text(self, text: str) -> list[int]: ...

    def encode_raw_code(self, source: str) -> list[int]: ...

    def encode_records(self, records: Iterable[SemanticTokenRecord]) -> list[int]: ...


@dataclass(frozen=True)
class CorpusValidationInput:
    documents: tuple[SourceDocumentRecord, ...]
    candidates: tuple[CurriculumRecord, ...]


@dataclass(frozen=True)
class AdmissionPolicy:
    minimum_records: int = 100_000
    minimum_encoded_tokens: int = 50_000_000
    maximum_document_fraction: float = 0.005
    maximum_template_family_fraction: float = 0.10
    near_duplicate_threshold: float = 0.90
    maximum_sequence_tokens: int = 4096
    holdout_hashes: frozenset[str] = frozenset()
    minimum_event_types: int = 8
    minimum_scope_depths: int = 3
    minimum_control_flow_types: int = 3
    minimum_identifier_shapes: int = 3
    minimum_literal_types: int = 3
    minimum_uncertainty_bands: int = 2

    def __post_init__(self) -> None:
        for name in ("minimum_records", "minimum_encoded_tokens", "maximum_sequence_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "maximum_document_fraction",
            "maximum_template_family_fraction",
            "near_duplicate_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class AdmissionDecision:
    decision_id: str
    record_id: str
    source_document_ids: tuple[str, ...]
    accepted: bool
    reason_code: str
    split: str

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "record_id": self.record_id,
            "source_document_ids": list(self.source_document_ids),
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "split": self.split,
        }


@dataclass(frozen=True)
class CorpusValidationReport:
    accepted_records: tuple[CurriculumRecord, ...]
    decisions: tuple[AdmissionDecision, ...]
    split_manifest: dict[str, str]
    distributions: dict[str, Any]
    failure_codes: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failure_codes": list(self.failure_codes),
            "accepted_record_count": len(self.accepted_records),
            "split_manifest": dict(sorted(self.split_manifest.items())),
            "distributions": self.distributions,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def assign_split(group_identity: str) -> str:
    bucket = int(hashlib.sha256(group_identity.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def validate_corpus(
    database: CorpusValidationInput,
    tokenizer: AdmissionTokenizer,
    policy: AdmissionPolicy | None = None,
) -> CorpusValidationReport:
    active_policy = policy or AdmissionPolicy()
    documents = {document.document_id: document for document in database.documents}
    records_by_document = _canonical_records(database.documents)
    split_manifest, holdout_documents = _build_split_manifest(
        database.documents, records_by_document, active_policy.holdout_hashes
    )
    candidates = list(database.candidates)
    document_limit = max(1, math.floor(len(candidates) * active_policy.maximum_document_fraction))
    template_limit = max(
        1,
        math.floor(len(candidates) * active_policy.maximum_template_family_fraction),
    )

    accepted: list[CurriculumRecord] = []
    decisions: list[AdmissionDecision] = []
    seen_record_ids: set[str] = set()
    seen_pairs: set[str] = set()
    seen_semantic: set[tuple[str, str]] = set()
    near_buckets: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    document_counts: Counter[str] = Counter()
    template_counts: Counter[tuple[str, str]] = Counter()

    for ordinal, candidate in enumerate(candidates):
        source_ids = candidate.source_document_ids
        assigned_splits = {split_manifest.get(source_id, "unknown") for source_id in source_ids}
        split = next(iter(assigned_splits)) if len(assigned_splits) == 1 else "conflict"
        reason = _precheck_candidate(
            candidate,
            documents,
            holdout_documents,
            split,
            assigned_splits,
            seen_record_ids,
        )
        if reason is None:
            pair_hash = _pair_hash(candidate)
            semantic_hash = _semantic_payload_hash(candidate)
            shingles = _five_grams(candidate.prompt + "\n" + candidate.target)
            if pair_hash in seen_pairs:
                reason = "exact_pair_duplicate"
            elif semantic_hash and (candidate.family, semantic_hash) in seen_semantic:
                reason = "canonical_semantic_duplicate"
            elif _is_near_duplicate(
                candidate,
                shingles,
                accepted,
                near_buckets[candidate.family],
                active_policy.near_duplicate_threshold,
            ):
                reason = "near_duplicate_pair"
            elif any(document_counts[source_id] >= document_limit for source_id in source_ids):
                reason = "document_cap_exceeded"
            elif template_counts[(candidate.family, candidate.template_id)] >= template_limit:
                reason = "template_family_cap_exceeded"
            elif _encoded_length(candidate, tokenizer) > active_policy.maximum_sequence_tokens:
                reason = "sequence_too_long"

        is_accepted = reason is None
        reason_code = "accepted" if is_accepted else reason
        decisions.append(
            AdmissionDecision(
                decision_id=_decision_id(candidate.record_id, ordinal, reason_code, split),
                record_id=candidate.record_id,
                source_document_ids=source_ids,
                accepted=is_accepted,
                reason_code=reason_code,
                split=split,
            )
        )
        seen_record_ids.add(candidate.record_id)
        if not is_accepted:
            continue
        admitted = replace(candidate, split=split)
        accepted.append(admitted)
        seen_pairs.add(_pair_hash(candidate))
        semantic_hash = _semantic_payload_hash(candidate)
        if semantic_hash:
            seen_semantic.add((candidate.family, semantic_hash))
        accepted_index = len(accepted) - 1
        for signature_value in _minhash_signature(shingles):
            near_buckets[candidate.family][signature_value].append(accepted_index)
        for source_id in source_ids:
            document_counts[source_id] += 1
        template_counts[(candidate.family, candidate.template_id)] += 1

    encoded_tokens = sum(_encoded_length(item, tokenizer) for item in accepted)
    distributions = _build_distributions(
        accepted,
        decisions,
        documents,
        records_by_document,
        encoded_tokens,
        len(holdout_documents),
    )
    failures = _gate_failures(len(accepted), encoded_tokens, distributions, active_policy)
    return CorpusValidationReport(
        accepted_records=tuple(accepted),
        decisions=tuple(decisions),
        split_manifest=split_manifest,
        distributions=distributions,
        failure_codes=tuple(failures),
        passed=not failures,
    )


def _precheck_candidate(
    candidate: CurriculumRecord,
    documents: dict[str, SourceDocumentRecord],
    holdout_documents: set[str],
    split: str,
    assigned_splits: set[str],
    seen_record_ids: set[str],
) -> str | None:
    if not candidate.source_document_ids or any(
        source_id not in documents for source_id in candidate.source_document_ids
    ):
        return "unknown_source_document"
    provenance = dict(candidate.provenance)
    if not _REQUIRED_PROVENANCE <= provenance.keys() or any(
        not provenance[key] for key in _REQUIRED_PROVENANCE
    ):
        return "missing_provenance"
    if any(source_id in holdout_documents for source_id in candidate.source_document_ids):
        return "evaluation_holdout_collision"
    if len(assigned_splits) != 1 or split in {"unknown", "conflict"}:
        return "split_isolation_conflict"
    if candidate.split not in {"unassigned", split}:
        return "split_mismatch"
    if candidate.record_id in seen_record_ids:
        return "exact_record_duplicate"
    return None


def _canonical_records(
    documents: Sequence[SourceDocumentRecord],
) -> dict[str, tuple[SemanticTokenRecord, ...]]:
    adapter = PythonAdapter()
    result: dict[str, tuple[SemanticTokenRecord, ...]] = {}
    for document in documents:
        parsed = adapter.parse(document.content, path=document.relative_path)
        extraction = adapter.extract(parsed)
        result[document.document_id] = records_from_assessments(extraction.assessments)
    return result


def _build_split_manifest(
    documents: Sequence[SourceDocumentRecord],
    records_by_document: dict[str, tuple[SemanticTokenRecord, ...]],
    holdout_hashes: frozenset[str],
) -> tuple[dict[str, str], set[str]]:
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

    fingerprint_owner: dict[str, str] = {}
    for document in documents:
        fingerprints = {
            "content:" + document.content_sha256,
            "normalized:" + _normalized_source_hash(document.content),
            "semantic:" + _document_semantic_hash(records_by_document[document.document_id]),
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
        group_holdout = any(member.content_sha256 in holdout_hashes for member in members)
        identity = min(
            f"{member.repository_identity}:{_normalized_source_hash(member.content)}"
            for member in members
        )
        split = "evaluation_holdout" if group_holdout else assign_split(identity)
        for member in members:
            manifest[member.document_id] = split
            if group_holdout:
                holdout_documents.add(member.document_id)
    return manifest, holdout_documents


def _normalized_source_hash(source: str) -> str:
    tree = ast.parse(source)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _document_semantic_hash(records: Sequence[SemanticTokenRecord]) -> str:
    reduced = []
    for record in records:
        payload = json.loads(canonical_record_bytes(record))
        for field in ("surface", "span", "recent_history"):
            payload.pop(field, None)
        reduced.append(payload)
    return hashlib.sha256(
        json.dumps(reduced, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _semantic_payload_hash(candidate: CurriculumRecord) -> str | None:
    records: list[dict[str, object]] = []
    for line in (candidate.prompt + "\n" + candidate.target).splitlines():
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict) and "event" in raw and "span" in raw:
            records.append(raw)
    if not records:
        return None
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _pair_hash(candidate: CurriculumRecord) -> str:
    normalized = "\n".join(
        (
            candidate.family,
            candidate.prompt_mode,
            candidate.target_mode,
            _normalize_text(candidate.prompt),
            _normalize_text(candidate.target),
        )
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("\r\n", "\n").replace("\r", "\n").split())


def _five_grams(value: str) -> frozenset[tuple[str, ...]]:
    words = tuple(re.findall(r"[A-Za-z0-9_]+|\S", value.casefold()))
    if not words:
        return frozenset()
    width = min(5, len(words))
    return frozenset(tuple(words[index : index + width]) for index in range(len(words) - width + 1))


def _jaccard(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _minhash_signature(
    shingles: frozenset[tuple[str, ...]], *, size: int = 64
) -> tuple[int, ...]:
    values = {
        int.from_bytes(
            hashlib.blake2b("\x1f".join(shingle).encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for shingle in shingles
    }
    return tuple(heapq.nsmallest(size, values))


def _is_near_duplicate(
    candidate: CurriculumRecord,
    shingles: frozenset[tuple[str, ...]],
    accepted: Sequence[CurriculumRecord],
    buckets: dict[int, list[int]],
    threshold: float,
) -> bool:
    signature = _minhash_signature(shingles)
    candidate_indexes = {
        index for value in signature for index in buckets.get(value, ())
    }
    for index in sorted(candidate_indexes):
        previous = accepted[index]
        previous_shingles = _five_grams(previous.prompt + "\n" + previous.target)
        if _jaccard(shingles, previous_shingles) >= threshold:
            return True
    return False


def _decision_id(record_id: str, ordinal: int, reason: str, split: str) -> str:
    return hashlib.sha256(f"{record_id}:{ordinal}:{reason}:{split}".encode()).hexdigest()


def _semantic_records_from_text(value: str) -> tuple[SemanticTokenRecord, ...]:
    records = []
    for line in value.splitlines():
        if not line.startswith("{"):
            continue
        try:
            records.append(record_from_canonical_bytes((line + "\n").encode("utf-8")))
        except ValueError:
            continue
    return tuple(records)


def _encoded_length(record: CurriculumRecord, tokenizer: AdmissionTokenizer) -> int:
    def encode(mode: str, value: str) -> list[int]:
        if mode == "raw_code":
            return tokenizer.encode_raw_code(value)
        if mode == "semantic_code":
            semantic = _semantic_records_from_text(value)
            if semantic:
                return tokenizer.encode_records(semantic)
        return tokenizer.encode_text(value)

    return len(encode(record.prompt_mode, record.prompt)) + len(
        encode(record.target_mode, record.target)
    )


def _build_distributions(
    accepted: Sequence[CurriculumRecord],
    decisions: Sequence[AdmissionDecision],
    documents: dict[str, SourceDocumentRecord],
    records_by_document: dict[str, tuple[SemanticTokenRecord, ...]],
    encoded_tokens: int,
    holdout_collisions: int,
) -> dict[str, Any]:
    accepted_document_ids = {
        source_id for record in accepted for source_id in record.source_document_ids
    }
    semantic_records = [
        record
        for source_id in sorted(accepted_document_ids)
        for record in records_by_document.get(source_id, ())
    ]
    identifier_shapes: Counter[str] = Counter()
    literal_types: Counter[str] = Counter()
    control_flow: Counter[str] = Counter()
    for source_id in sorted(accepted_document_ids):
        tree = ast.parse(documents[source_id].content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifier_shapes[_identifier_shape(node.id)] += 1
            if isinstance(node, ast.Constant):
                literal_types[type(node.value).__name__] += 1
            control = _control_flow_name(node)
            if control:
                control_flow[control] += 1
    return {
        "family": _counter(record.family for record in accepted),
        "event": _counter(record.event for record in semantic_records),
        "scope_depth": _counter(str(_scope_depth(record.scope)) for record in semantic_records),
        "control_flow": dict(sorted(control_flow.items())),
        "identifier_shape": dict(sorted(identifier_shapes.items())),
        "literal_type": dict(sorted(literal_types.items())),
        "uncertainty": _counter(record.uncertainty_band.value for record in semantic_records),
        "source": _counter(
            documents[source_id].source_name for source_id in accepted_document_ids
        ),
        "license": _counter(
            documents[source_id].license_spdx for source_id in accepted_document_ids
        ),
        "split": _counter(record.split for record in accepted),
        "rejection_reason": _counter(
            decision.reason_code for decision in decisions if not decision.accepted
        ),
        "duplicate": _counter(
            decision.reason_code
            for decision in decisions
            if "duplicate" in decision.reason_code
        ),
        "encoded_tokens": encoded_tokens,
        "holdout_collisions": holdout_collisions,
    }


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _identifier_shape(value: str) -> str:
    if value.startswith("__") and value.endswith("__"):
        return "dunder"
    if value.isupper():
        return "upper"
    if "_" in value:
        return "snake_case"
    if value[:1].isupper():
        return "pascal_case"
    if any(character.isupper() for character in value[1:]):
        return "camel_case"
    if len(value) == 1:
        return "single_character"
    return "lower"


def _control_flow_name(node: ast.AST) -> str | None:
    for node_type, name in (
        (ast.If, "branch"),
        (ast.For, "for_loop"),
        (ast.AsyncFor, "async_for_loop"),
        (ast.While, "while_loop"),
        (ast.Try, "try"),
        (ast.Match, "match"),
        (ast.comprehension, "comprehension"),
    ):
        if isinstance(node, node_type):
            return name
    return None


def _scope_depth(scope: str) -> int:
    return max(0, len([part for part in re.split(r"::|[/.]", scope) if part]) - 1)


def _gate_failures(
    accepted_count: int,
    encoded_tokens: int,
    distributions: dict[str, Any],
    policy: AdmissionPolicy,
) -> list[str]:
    failures = []
    if accepted_count < policy.minimum_records:
        failures.append("minimum_record_count_not_met")
    if encoded_tokens < policy.minimum_encoded_tokens:
        failures.append("minimum_encoded_tokens_not_met")
    if distributions["holdout_collisions"]:
        failures.append("holdout_collision_detected")
    diversity = (
        ("event", policy.minimum_event_types, "insufficient_event_diversity"),
        ("scope_depth", policy.minimum_scope_depths, "insufficient_scope_depth_diversity"),
        ("control_flow", policy.minimum_control_flow_types, "insufficient_control_flow_diversity"),
        ("identifier_shape", policy.minimum_identifier_shapes, "insufficient_identifier_shape_diversity"),
        ("literal_type", policy.minimum_literal_types, "insufficient_literal_type_diversity"),
        ("uncertainty", policy.minimum_uncertainty_bands, "insufficient_uncertainty_diversity"),
    )
    for key, minimum, reason in diversity:
        if len(distributions[key]) < minimum:
            failures.append(reason)
    return failures
