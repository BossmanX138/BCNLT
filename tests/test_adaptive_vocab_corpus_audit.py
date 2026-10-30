from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path

from bcnlt.adapters.python import PythonAdapter
from bcnlt.adaptive_vocab.config import AdaptiveVocabularyConfig
from bcnlt.adaptive_vocab.corpus_audit import audit_corpus
from bcnlt.canonical import canonical_record_bytes
from bcnlt.corpus.curriculum import CurriculumRecord, make_curriculum_record
from bcnlt.corpus.database import SourceDocumentRecord
from bcnlt.languages import CANONICAL_LANGUAGE_IDS
from bcnlt.semantic_record import records_from_assessments
from scripts.audit_bcnlt_adaptive_vocab import main as audit_main


SOURCE = "value = 42\n"


class _Manifest:
    base_vocab_size = 100


class _AuditTokenizer:
    manifest = _Manifest()

    def encode_records(self, records):
        assert tuple(records)
        return [100, 101, 1, 2, 102]


def _config() -> AdaptiveVocabularyConfig:
    return replace(
        AdaptiveVocabularyConfig.load("configs/bcnlt_adaptive_vocab_v5.json"),
        minimum_documents_per_language=1,
        minimum_repositories_per_language=1,
        minimum_payload_tokens_per_language=1,
    )


def _document(language: str, ordinal: int, repository: str | None = None) -> SourceDocumentRecord:
    digest = hashlib.sha256(f"{language}:{ordinal}:{SOURCE}".encode()).hexdigest()
    return SourceDocumentRecord(
        document_id=f"{language}-doc-{ordinal}",
        source_provider="test",
        source_name=f"source/{language}",
        source_revision="a" * 40,
        canonical_language=language,
        repository_identity=repository or f"repo/{language}/{ordinal}",
        relative_path=f"src/example-{ordinal}.txt",
        content_sha256=digest,
        content=SOURCE,
        license_spdx="MIT",
        ownership="third_party",
        attribution="test",
        upstream_url="https://example.invalid",
    )


def _semantic_stream() -> str:
    adapter = PythonAdapter()
    extraction = adapter.extract(adapter.parse(SOURCE, path="example.py"))
    records = records_from_assessments(extraction.assessments)
    return b"".join(canonical_record_bytes(record) for record in records).decode()


def _record(document: SourceDocumentRecord, *, split: str = "train") -> CurriculumRecord:
    return make_curriculum_record(
        document,
        family="code_to_semantic",
        template_id="audit-v1",
        prompt_mode="raw_code",
        target_mode="semantic_code",
        prompt=document.content,
        target=_semantic_stream(),
        split=split,
    )


def test_python_only_input_hard_stops_shared_candidate_discovery():
    document = _document("python", 1)

    report = audit_corpus((document,), (_record(document),), _AuditTokenizer(), _config())

    assert report.passed is False
    assert report.failure_codes == ("insufficient_shared_candidate_diversity",)
    python = report.by_language["python"]
    assert python.document_count == 1
    assert python.repository_count == 1
    assert python.payload_base_tokens == 2
    assert report.training_source_document_ids == (document.document_id,)


def test_all_fourteen_qualifying_languages_pass_in_canonical_order():
    documents = tuple(
        _document(language, ordinal)
        for ordinal, language in enumerate(CANONICAL_LANGUAGE_IDS)
    )
    records = tuple(_record(document) for document in documents)

    report = audit_corpus(documents, records, _AuditTokenizer(), _config())

    assert report.passed is True
    assert report.failure_codes == ()
    assert tuple(report.by_language) == CANONICAL_LANGUAGE_IDS
    assert all(item.document_count == 1 for item in report.by_language.values())
    assert all(item.payload_base_tokens == 2 for item in report.by_language.values())


def test_repository_counts_are_unique_and_nontraining_records_do_not_contribute():
    first = _document("python", 1, repository="shared/repository")
    second = _document("python", 2, repository="shared/repository")
    validation = _document("javascript", 3)

    report = audit_corpus(
        (first, second, validation),
        (_record(first), _record(second), _record(validation, split="validation")),
        _AuditTokenizer(),
        _config(),
    )

    assert report.by_language["python"].document_count == 2
    assert report.by_language["python"].repository_count == 1
    assert report.by_language["javascript"].document_count == 0
    assert validation.document_id not in report.training_source_document_ids


def test_training_record_without_a_semantic_side_is_ineligible_not_invalid():
    document = _document("python", 1)
    semantic = _record(document)
    nonsemantic = replace(
        semantic,
        record_id="e" * 64,
        family="mixed_instruction",
        prompt_mode="mixed",
        target_mode="text",
        prompt="Explain the code.",
        target="It assigns a value.",
    )

    report = audit_corpus(
        (document,),
        (semantic, nonsemantic),
        _AuditTokenizer(),
        _config(),
    )

    assert "invalid_semantic_payload" not in report.failure_codes
    assert report.by_language["python"].payload_base_tokens == 2


def test_language_mismatch_fails_closed_without_counting_the_record():
    document = _document("python", 1)
    record = _record(document)
    provenance = dict(record.provenance)
    provenance["canonical_language"] = "javascript"
    mismatched = replace(record, provenance=tuple(sorted(provenance.items())))

    report = audit_corpus((document,), (mismatched,), _AuditTokenizer(), _config())

    assert "canonical_language_mismatch" in report.failure_codes
    assert report.by_language["python"].document_count == 0


def test_missing_source_document_fails_closed():
    document = _document("python", 1)
    missing = replace(
        _record(document),
        record_id="f" * 64,
        source_document_ids=("missing-document",),
    )

    report = audit_corpus((document,), (missing,), _AuditTokenizer(), _config())

    assert "unknown_source_document" in report.failure_codes
    assert report.training_source_document_ids == ()


def test_cli_refuses_legacy_database_without_writing_partial_report(
    tmp_path: Path, capsys
):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                source_provider TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                repository_identity TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                content TEXT NOT NULL,
                license_spdx TEXT NOT NULL,
                ownership TEXT NOT NULL,
                attribution TEXT NOT NULL,
                upstream_url TEXT NOT NULL
            );
            INSERT INTO documents VALUES (
                'doc', 'test', 'legacy', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                'repo', 'x.py', 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                'x = 1', 'MIT', 'third_party', 'repo', 'https://example.invalid'
            );
            """
        )
    output = tmp_path / "corpus-audit.json"

    result = audit_main(
        [
            "--database", str(database),
            "--accepted-records", str(tmp_path / "accepted.jsonl"),
            "--tokenizer-artifact", str(tmp_path / "tokenizer"),
            "--base-tokenizer", str(tmp_path / "base"),
            "--config", "configs/bcnlt_adaptive_vocab_v5.json",
            "--source-config", str(tmp_path / "sources.json"),
            "--corpus-manifest", str(tmp_path / "corpus.json"),
            "--lineage", str(tmp_path / "lineage.json"),
            "--output", str(output),
        ]
    )

    assert result == 2
    assert "legacy corpus database requires a new v2 output path" in capsys.readouterr().err
    assert not output.exists()
