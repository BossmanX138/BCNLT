from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator

from bcnlt.canonical import record_from_canonical_bytes
from bcnlt.corpus.curriculum import CurriculumRecord

from .config import AdaptiveVocabularyConfig
from .secrets import SensitiveTextFinding, classify_sensitive_text


_FIELD_ORDER = (
    "surface",
    "scope",
    "symbol_id",
    "defines",
    "span",
    "ast_relation",
    "ir",
    "recent_history",
    "metadata",
)
_JSON_FIELDS = {"span", "ir", "recent_history", "metadata"}


@dataclass(frozen=True, slots=True)
class PayloadObservation:
    language: str
    repository: str
    document_id: str
    event: str
    field: str
    base_ids: tuple[int, ...]
    decoded_text: str
    window_texts: tuple[tuple[int, int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SupportCount:
    identity: str
    occurrences: int

    def to_dict(self) -> dict[str, object]:
        return {"identity": self.identity, "occurrences": self.occurrences}


@dataclass(frozen=True, slots=True)
class CandidateRate:
    language: str
    payload_class: str
    capped_occurrences: int
    eligible_base_tokens: int
    rate_per_million: float

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "payload_class": self.payload_class,
            "capped_occurrences": self.capped_occurrences,
            "eligible_base_tokens": self.eligible_base_tokens,
            "rate_per_million": self.rate_per_million,
        }


@dataclass(frozen=True, slots=True)
class PayloadClassScore:
    payload_class: str
    third_language_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "payload_class": self.payload_class,
            "third_language_rate": self.third_language_rate,
        }


@dataclass(frozen=True, slots=True)
class MacroCandidate:
    expansion: tuple[int, ...]
    length: int
    score: float
    languages: tuple[str, ...]
    repositories: tuple[str, ...]
    documents: tuple[str, ...]
    raw_occurrences: int
    capped_occurrences: int
    repository_counts: tuple[SupportCount, ...]
    document_counts: tuple[SupportCount, ...]
    rates: tuple[CandidateRate, ...]
    payload_class_scores: tuple[PayloadClassScore, ...]
    eligible: bool
    reason_codes: tuple[str, ...]
    sensitive_evidence: tuple[SensitiveTextFinding, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "expansion": list(self.expansion),
            "length": self.length,
            "score": self.score,
            "languages": list(self.languages),
            "repositories": list(self.repositories),
            "documents": list(self.documents),
            "raw_occurrences": self.raw_occurrences,
            "capped_occurrences": self.capped_occurrences,
            "repository_counts": [item.to_dict() for item in self.repository_counts],
            "document_counts": [item.to_dict() for item in self.document_counts],
            "rates": [item.to_dict() for item in self.rates],
            "payload_class_scores": [
                item.to_dict() for item in self.payload_class_scores
            ],
            "eligible": self.eligible,
            "reason_codes": list(self.reason_codes),
            "sensitive_evidence": [
                {
                    "reason_code": item.reason_code,
                    "category": item.category,
                    "evidence_sha256": item.evidence_sha256,
                    "character_count": item.character_count,
                }
                for item in self.sensitive_evidence
            ],
        }


@dataclass(frozen=True, slots=True)
class CandidateDiscoveryReport:
    candidates: tuple[MacroCandidate, ...]
    observation_count: int
    eligible_base_tokens: tuple[tuple[str, str, int], ...]
    lineage_sha256: str = ""
    candidate_report_sha256: str = ""

    @property
    def eligible_candidates(self) -> tuple[MacroCandidate, ...]:
        return tuple(item for item in self.candidates if item.eligible)

    @property
    def rejected_candidates(self) -> tuple[MacroCandidate, ...]:
        return tuple(item for item in self.candidates if not item.eligible)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "bcnlt-adaptive-candidate-discovery-v1",
            "observation_count": self.observation_count,
            "candidate_count": len(self.candidates),
            "eligible_candidate_count": len(self.eligible_candidates),
            "rejected_candidate_count": len(self.rejected_candidates),
            "eligible_base_tokens": [
                {
                    "language": language,
                    "payload_class": payload_class,
                    "count": count,
                }
                for language, payload_class, count in self.eligible_base_tokens
            ],
        }


def _semantic_payloads(value: str) -> Iterator[tuple[str, dict[str, object]]]:
    for line in value.splitlines():
        if not line.startswith("{"):
            continue
        canonical = (line + "\n").encode("utf-8")
        record = record_from_canonical_bytes(canonical)
        raw = json.loads(canonical)
        yield record.event, raw


def iter_payload_observations(
    records: Iterable[CurriculumRecord], tokenizer
) -> Iterator[PayloadObservation]:
    base_vocab_size = int(tokenizer.manifest.base_vocab_size)
    for curriculum in sorted(records, key=lambda item: item.record_id):
        if curriculum.split != "train" or not curriculum.source_document_ids:
            continue
        provenance = dict(curriculum.provenance)
        language = provenance.get("canonical_language")
        repository = provenance.get("repository_identity")
        if not language or not repository:
            continue
        for mode, value in (
            (curriculum.prompt_mode, curriculum.prompt),
            (curriculum.target_mode, curriculum.target),
        ):
            if mode != "semantic_code":
                continue
            for event, raw in _semantic_payloads(value):
                for field in _FIELD_ORDER:
                    field_value = raw.get(field)
                    if field_value is None:
                        continue
                    text = (
                        json.dumps(
                            field_value,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if field in _JSON_FIELDS
                        else field_value
                    )
                    if not isinstance(text, str):
                        raise ValueError(f"semantic field {field} is not text")
                    base_ids = tuple(tokenizer.base.encode(text, add_special_tokens=False))
                    if any(
                        not isinstance(token_id, int)
                        or token_id < 0
                        or token_id >= base_vocab_size
                        for token_id in base_ids
                    ):
                        raise ValueError("semantic payload contains a structural or invalid token id")
                    windows = tuple(
                        (
                            start,
                            end,
                            tokenizer.base.decode(base_ids[start:end]),
                        )
                        for start in range(len(base_ids))
                        for end in range(start + 2, min(len(base_ids), start + 16) + 1)
                    )
                    yield PayloadObservation(
                        language=str(language),
                        repository=str(repository),
                        document_id=curriculum.source_document_ids[0],
                        event=event,
                        field=field,
                        base_ids=base_ids,
                        decoded_text=text,
                        window_texts=windows,
                    )


@dataclass(frozen=True, slots=True)
class _Occurrence:
    observation: PayloadObservation
    position: int
    decoded_text: str | None


def discover_candidates(
    observations: Iterable[PayloadObservation],
    config: AdaptiveVocabularyConfig,
) -> CandidateDiscoveryReport:
    ordered = tuple(
        sorted(
            observations,
            key=lambda item: (
                item.language,
                item.repository,
                item.document_id,
                item.event,
                item.field,
                item.base_ids,
            ),
        )
    )
    denominators: Counter[tuple[str, str]] = Counter()
    occurrences: dict[tuple[int, ...], list[_Occurrence]] = defaultdict(list)
    for observation in ordered:
        denominators[(observation.language, observation.field)] += len(
            observation.base_ids
        )
        window_text = {
            (start, end): text for start, end, text in observation.window_texts
        }
        for length in range(config.candidate_min_tokens, config.candidate_max_tokens + 1):
            if length > len(observation.base_ids):
                break
            for start in range(len(observation.base_ids) - length + 1):
                expansion = observation.base_ids[start : start + length]
                decoded = window_text.get((start, start + length))
                if decoded is None and length == len(observation.base_ids) and start == 0:
                    decoded = observation.decoded_text
                occurrences[expansion].append(_Occurrence(observation, start, decoded))

    candidates = tuple(
        sorted(
            (
                _candidate(expansion, values, denominators, config)
                for expansion, values in occurrences.items()
            ),
            key=lambda item: (
                -item.score,
                -len(item.languages),
                -len(item.repositories),
                -len(item.documents),
                item.expansion,
            ),
        )
    )
    return CandidateDiscoveryReport(
        candidates=candidates,
        observation_count=len(ordered),
        eligible_base_tokens=tuple(
            (language, payload_class, count)
            for (language, payload_class), count in sorted(denominators.items())
        ),
    )


def _candidate(
    expansion: tuple[int, ...],
    raw: list[_Occurrence],
    denominators: Counter[tuple[str, str]],
    config: AdaptiveVocabularyConfig,
) -> MacroCandidate:
    by_document: dict[str, list[_Occurrence]] = defaultdict(list)
    for occurrence in raw:
        by_document[occurrence.observation.document_id].append(occurrence)
    capped = [
        occurrence
        for document_id in sorted(by_document)
        for occurrence in by_document[document_id][: config.maximum_occurrences_per_document]
    ]
    language_counts = Counter(item.observation.language for item in capped)
    repository_counts = Counter(item.observation.repository for item in capped)
    document_counts = Counter(item.observation.document_id for item in capped)
    field_counts = Counter(
        (item.observation.language, item.observation.field) for item in capped
    )
    payload_classes = sorted({item.observation.field for item in capped})
    rates = []
    class_scores = []
    for payload_class in payload_classes:
        class_rates = []
        for language in config.canonical_languages:
            numerator = field_counts[(language, payload_class)]
            denominator = denominators[(language, payload_class)]
            rate = 1_000_000 * numerator / denominator if denominator else 0.0
            class_rates.append(rate)
            rates.append(
                CandidateRate(
                    language=language,
                    payload_class=payload_class,
                    capped_occurrences=numerator,
                    eligible_base_tokens=denominator,
                    rate_per_million=rate,
                )
            )
        third = sorted(class_rates, reverse=True)[2] if len(class_rates) >= 3 else 0.0
        class_scores.append(PayloadClassScore(payload_class, third))
    score = (len(expansion) - 1) * max(
        (item.third_language_rate for item in class_scores), default=0.0
    )
    reasons: set[str] = set()
    if len(language_counts) < config.minimum_supporting_languages:
        reasons.add("insufficient_language_support")
    if len(repository_counts) < config.minimum_supporting_repositories:
        reasons.add("insufficient_repository_support")
    if len(document_counts) < config.minimum_supporting_documents:
        reasons.add("insufficient_document_support")
    capped_count = len(capped)
    if capped_count and max(repository_counts.values()) / capped_count > config.maximum_repository_fraction:
        reasons.add("repository_concentration_above_threshold")
    if capped_count and max(document_counts.values()) / capped_count > config.maximum_document_fraction:
        reasons.add("document_concentration_above_threshold")
    if score < config.minimum_candidate_score:
        reasons.add("candidate_score_below_threshold")

    sensitive: dict[tuple[str, str], SensitiveTextFinding] = {}
    decoded_values = sorted({item.decoded_text for item in capped if item.decoded_text})
    for decoded in decoded_values:
        for repository in (*sorted(repository_counts), None):
            finding = classify_sensitive_text(
                decoded,
                config,
                repository=repository,
                document_count=len(document_counts),
                repository_count=len(repository_counts),
            )
            if finding is not None:
                sensitive[(finding.reason_code, finding.evidence_sha256)] = finding
                reasons.add(finding.reason_code)
                break

    return MacroCandidate(
        expansion=expansion,
        length=len(expansion),
        score=score,
        languages=tuple(sorted(language_counts)),
        repositories=tuple(sorted(repository_counts)),
        documents=tuple(sorted(document_counts)),
        raw_occurrences=len(raw),
        capped_occurrences=capped_count,
        repository_counts=tuple(
            SupportCount(identity, count)
            for identity, count in sorted(repository_counts.items())
        ),
        document_counts=tuple(
            SupportCount(identity, count)
            for identity, count in sorted(document_counts.items())
        ),
        rates=tuple(rates),
        payload_class_scores=tuple(class_scores),
        eligible=not reasons,
        reason_codes=tuple(sorted(reasons)),
        sensitive_evidence=tuple(sensitive[key] for key in sorted(sensitive)),
    )
