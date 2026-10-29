from __future__ import annotations

from dataclasses import replace

import pytest

from bcnlt.adapters.python import PythonAdapter
from bcnlt.adaptive_vocab.candidates import (
    PayloadObservation,
    discover_candidates,
    iter_payload_observations,
)
from bcnlt.adaptive_vocab.config import AdaptiveVocabularyConfig
from bcnlt.canonical import canonical_record_bytes
from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.semantic_record import records_from_assessments


def _config(**changes) -> AdaptiveVocabularyConfig:
    base = AdaptiveVocabularyConfig.load("configs/bcnlt_adaptive_vocab_v5.json")
    defaults = {
        "candidate_min_tokens": 2,
        "candidate_max_tokens": 16,
        "minimum_supporting_languages": 3,
        "minimum_supporting_repositories": 3,
        "minimum_supporting_documents": 5,
        "maximum_repository_fraction": 0.50,
        "maximum_document_fraction": 0.10,
        "minimum_candidate_score": 25,
    }
    defaults.update(changes)
    return replace(base, **defaults)


def _observation(
    base_ids: tuple[int, ...],
    *,
    language: str = "python",
    repository: str = "repo/0",
    document_id: str = "doc-0",
    field: str = "surface",
    decoded_text: str = "ordinary payload",
) -> PayloadObservation:
    return PayloadObservation(
        language=language,
        repository=repository,
        document_id=document_id,
        event="REFERENCE",
        field=field,
        base_ids=base_ids,
        decoded_text=decoded_text,
    )


class _BaseTokenizer:
    def __init__(self):
        self.text_to_ids: dict[str, tuple[int, ...]] = {}
        self.ids_to_text: dict[tuple[int, ...], str] = {}
        self.next_id = 1

    def encode(self, text: str, add_special_tokens: bool = False):
        if text not in self.text_to_ids:
            ids = tuple(range(self.next_id, self.next_id + max(1, len(text))))
            self.next_id += len(ids)
            self.text_to_ids[text] = ids
            self.ids_to_text[ids] = text
        return list(self.text_to_ids[text])

    def decode(self, ids):
        key = tuple(ids)
        if key in self.ids_to_text:
            return self.ids_to_text[key]
        return f"window:{','.join(map(str, key))}"


class _ObservationTokenizer:
    class Manifest:
        base_vocab_size = 100_000

    manifest = Manifest()

    def __init__(self):
        self.base = _BaseTokenizer()


def test_observation_extraction_uses_training_semantic_fields_and_base_ids_only():
    source = "value = 42\n"
    adapter = PythonAdapter()
    extracted = adapter.extract(adapter.parse(source, path="example.py"))
    semantic = b"".join(
        canonical_record_bytes(record)
        for record in records_from_assessments(extracted.assessments)
    ).decode()
    provenance = (
        ("canonical_language", "python"),
        ("repository_identity", "owner/repository"),
    )
    training = CurriculumRecord(
        record_id="a" * 64,
        family="audit",
        template_id="audit-v1",
        source_document_ids=("doc-1",),
        prompt_mode="semantic_code",
        target_mode="raw_code",
        prompt=semantic,
        target=source,
        provenance=provenance,
        split="train",
    )
    validation = replace(training, record_id="b" * 64, split="validation")
    tokenizer = _ObservationTokenizer()

    observations = tuple(iter_payload_observations((training, validation), tokenizer))

    assert observations
    assert {item.document_id for item in observations} == {"doc-1"}
    assert {item.language for item in observations} == {"python"}
    assert {item.repository for item in observations} == {"owner/repository"}
    assert len({item.field for item in observations}) > 1
    assert all(
        token_id < tokenizer.manifest.base_vocab_size
        for item in observations
        for token_id in item.base_ids
    )


def test_windows_cover_two_through_sixteen_tokens_without_crossing_fields():
    first = _observation(tuple(range(1, 17)), field="surface", document_id="doc-a")
    second = _observation((30, 31), field="scope", document_id="doc-b")
    report = discover_candidates(
        (first, second),
        _config(
            minimum_supporting_languages=1,
            minimum_supporting_repositories=1,
            minimum_supporting_documents=1,
            maximum_repository_fraction=1.0,
            maximum_document_fraction=1.0,
            minimum_candidate_score=0,
        ),
    )
    expansions = {candidate.expansion for candidate in report.candidates}

    assert (1, 2) in expansions
    assert tuple(range(1, 17)) in expansions
    assert not any(len(expansion) > 16 for expansion in expansions)
    assert (16, 30) not in expansions


def test_exact_base_id_sequences_preserve_line_endings_and_unicode_variants():
    observations = (
        _observation((1, 2), document_id="crlf", decoded_text="a\r\nb"),
        _observation((1, 3), document_id="lf", decoded_text="a\nb"),
        _observation((4, 5), document_id="nfc", decoded_text="é"),
        _observation((4, 6), document_id="nfd", decoded_text="e\u0301"),
    )
    report = discover_candidates(
        observations,
        _config(
            minimum_supporting_languages=1,
            minimum_supporting_repositories=1,
            minimum_supporting_documents=1,
            maximum_repository_fraction=1.0,
            maximum_document_fraction=1.0,
            minimum_candidate_score=0,
        ),
    )

    assert {(1, 2), (1, 3), (4, 5), (4, 6)} <= {
        candidate.expansion for candidate in report.candidates
    }


def test_score_uses_third_language_rate_and_document_cap():
    observations = []
    languages = ("python", "javascript", "typescript")
    for language_index, language in enumerate(languages):
        for document_index in range(10):
            observations.append(
                _observation(
                    (7, 8, 9),
                    language=language,
                    repository=f"repo/{language_index}/{document_index % 3}",
                    document_id=f"{language}-{document_index}",
                )
            )
    repeated = (7, 8, 9) * 20
    observations.append(
        _observation(
            repeated,
            language="python",
            repository="repo/0/extra",
            document_id="python-repeat",
        )
    )

    candidate = next(
        item
        for item in discover_candidates(
            observations, _config(maximum_document_fraction=0.25)
        ).candidates
        if item.expansion == (7, 8, 9)
    )
    third_language_rate = sorted(
        (
            score.rate_per_million
            for score in candidate.rates
            if score.payload_class == "surface"
        ),
        reverse=True,
    )[2]

    assert candidate.score == pytest.approx((candidate.length - 1) * third_language_rate)
    assert candidate.score >= 25
    assert candidate.eligible is True
    assert candidate.raw_occurrences == 50
    assert candidate.capped_occurrences == 38
    assert max(item.occurrences for item in candidate.document_counts) == 8


def test_support_and_concentration_failures_have_stable_reason_codes():
    sparse = (
        _observation((1, 2), language="python", repository="one", document_id="a"),
        _observation((1, 2), language="javascript", repository="one", document_id="b"),
    )

    candidate = next(
        item for item in discover_candidates(sparse, _config()).candidates
        if item.expansion == (1, 2)
    )

    assert candidate.eligible is False
    assert {
        "insufficient_language_support",
        "insufficient_repository_support",
        "insufficient_document_support",
        "repository_concentration_above_threshold",
        "document_concentration_above_threshold",
    } <= set(candidate.reason_codes)


def test_candidates_sort_by_score_support_then_lexical_base_ids():
    observations = []
    for language_index, language in enumerate(("python", "javascript", "typescript")):
        for document_index in range(10):
            observations.extend(
                (
                    _observation(
                        (1, 2),
                        language=language,
                        repository=f"r/{language_index}/{document_index}",
                        document_id=f"a/{language}/{document_index}",
                    ),
                    _observation(
                        (1, 3),
                        language=language,
                        repository=f"s/{language_index}/{document_index}",
                        document_id=f"b/{language}/{document_index}",
                    ),
                )
            )

    eligible = [
        item for item in discover_candidates(observations, _config()).candidates
        if item.eligible and item.expansion in {(1, 2), (1, 3)}
    ]

    assert [item.expansion for item in eligible] == [(1, 2), (1, 3)]


def test_discovery_requires_exact_accepted_records_and_tokenizer_evidence(tmp_path):
    from scripts.discover_bcnlt_macro_candidates import verify_audit_inputs

    records = tmp_path / "accepted.jsonl"
    manifest = tmp_path / "bcnlt_tokenizer_manifest.json"
    records.write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    import hashlib

    audit = {
        "accepted_records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        "tokenizer_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    verify_audit_inputs(audit, records, manifest)
    records.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="accepted records"):
        verify_audit_inputs(audit, records, manifest)
