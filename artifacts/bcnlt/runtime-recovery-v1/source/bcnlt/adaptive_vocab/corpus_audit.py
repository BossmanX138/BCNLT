from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from bcnlt.canonical import record_from_canonical_bytes
from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.corpus.database import SourceDocumentRecord
from bcnlt.semantic_record import SemanticTokenRecord

from .config import AdaptiveVocabularyConfig


AUDIT_SCHEMA_VERSION = "bcnlt-adaptive-corpus-audit-v1"


@dataclass(frozen=True, slots=True)
class LanguageAudit:
    canonical_language: str
    document_count: int
    repository_count: int
    payload_base_tokens: int
    source_document_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "document_count": self.document_count,
            "repository_count": self.repository_count,
            "payload_base_tokens": self.payload_base_tokens,
            "source_document_ids": list(self.source_document_ids),
        }


@dataclass(frozen=True, slots=True)
class CorpusAuditReport:
    schema_version: str
    passed: bool
    failure_codes: tuple[str, ...]
    languages: tuple[LanguageAudit, ...]
    training_source_document_ids: tuple[str, ...]

    @property
    def by_language(self) -> dict[str, LanguageAudit]:
        return {item.canonical_language: item for item in self.languages}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "failure_codes": list(self.failure_codes),
            "per_language": {
                item.canonical_language: item.to_dict() for item in self.languages
            },
            "training_source_document_ids": list(self.training_source_document_ids),
        }


def _semantic_records(value: str) -> tuple[SemanticTokenRecord, ...] | None:
    records: list[SemanticTokenRecord] = []
    for line in value.splitlines():
        if not line.startswith("{"):
            continue
        try:
            records.append(record_from_canonical_bytes((line + "\n").encode("utf-8")))
        except ValueError:
            return None
    return tuple(records)


def _semantic_payload_base_tokens(record: CurriculumRecord, tokenizer) -> int | None:
    total = 0
    semantic_sides = 0
    base_vocab_size = int(tokenizer.manifest.base_vocab_size)
    for mode, value in (
        (record.prompt_mode, record.prompt),
        (record.target_mode, record.target),
    ):
        if mode != "semantic_code":
            continue
        semantic_sides += 1
        semantic_records = _semantic_records(value)
        if not semantic_records:
            return None
        encoded = tokenizer.encode_records(semantic_records)
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in encoded):
            return None
        total += sum(token_id < base_vocab_size for token_id in encoded)
    return total if semantic_sides else None


def audit_corpus(
    documents: Sequence[SourceDocumentRecord],
    records: Sequence[CurriculumRecord],
    tokenizer,
    config: AdaptiveVocabularyConfig,
) -> CorpusAuditReport:
    document_by_id: dict[str, SourceDocumentRecord] = {}
    failures: set[str] = set()
    for document in documents:
        if document.document_id in document_by_id:
            failures.add("duplicate_source_document_id")
        document_by_id[document.document_id] = document

    document_ids = {language: set() for language in config.canonical_languages}
    repositories = {language: set() for language in config.canonical_languages}
    payload_tokens = {language: 0 for language in config.canonical_languages}

    for record in sorted(records, key=lambda item: item.record_id):
        if record.split != "train":
            continue
        if not record.source_document_ids or any(
            source_id not in document_by_id for source_id in record.source_document_ids
        ):
            failures.add("unknown_source_document")
            continue
        provenance_language = dict(record.provenance).get("canonical_language")
        referenced = tuple(document_by_id[source_id] for source_id in record.source_document_ids)
        referenced_languages = {document.canonical_language for document in referenced}
        if (
            provenance_language not in config.canonical_languages
            or referenced_languages != {provenance_language}
        ):
            failures.add("canonical_language_mismatch")
            continue
        if "semantic_code" not in (record.prompt_mode, record.target_mode):
            continue
        count = _semantic_payload_base_tokens(record, tokenizer)
        if count is None:
            failures.add("invalid_semantic_payload")
            continue
        language = str(provenance_language)
        payload_tokens[language] += count
        for document in referenced:
            document_ids[language].add(document.document_id)
            repositories[language].add(document.repository_identity)

    language_reports = tuple(
        LanguageAudit(
            canonical_language=language,
            document_count=len(document_ids[language]),
            repository_count=len(repositories[language]),
            payload_base_tokens=payload_tokens[language],
            source_document_ids=tuple(sorted(document_ids[language])),
        )
        for language in config.canonical_languages
    )
    if any(
        item.document_count < config.minimum_documents_per_language
        or item.repository_count < config.minimum_repositories_per_language
        or item.payload_base_tokens < config.minimum_payload_tokens_per_language
        for item in language_reports
    ):
        failures.add("insufficient_shared_candidate_diversity")
    training_ids = tuple(
        sorted(source_id for values in document_ids.values() for source_id in values)
    )
    failure_codes = tuple(sorted(failures))
    return CorpusAuditReport(
        schema_version=AUDIT_SCHEMA_VERSION,
        passed=not failure_codes,
        failure_codes=failure_codes,
        languages=language_reports,
        training_source_document_ids=training_ids,
    )
