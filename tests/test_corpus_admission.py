from __future__ import annotations

import hashlib
from dataclasses import replace

from bcnlt.adapters.python import PythonAdapter
from bcnlt.corpus.admission import (
    AdmissionPolicy,
    CorpusValidationInput,
    assign_split,
    validate_corpus,
)
from bcnlt.corpus.curriculum import CurriculumRecord, generate_document_curriculum
from bcnlt.corpus.database import SourceDocumentRecord, stable_document_id
from bcnlt.semantic_record import records_from_units
from scripts.validate_bcnlt_corpus import admission_policy_for_profile


class CountingTokenizer:
    def encode_text(self, value: str) -> list[int]:
        return list(range(max(1, len(value.split()))))

    encode_raw_code = encode_text

    def encode_records(self, records) -> list[int]:
        return list(range(max(1, len(tuple(records)) * 4)))


def test_smoke_admission_profile_relaxes_only_scale_and_cap_gates():
    full = admission_policy_for_profile("full")
    smoke = admission_policy_for_profile("smoke")

    assert full == AdmissionPolicy()
    assert smoke.minimum_records == 100
    assert smoke.minimum_encoded_tokens == 10_000
    assert smoke.maximum_document_fraction == 1.0
    assert smoke.maximum_template_family_fraction == 1.0
    assert smoke.maximum_sequence_tokens == full.maximum_sequence_tokens
    assert smoke.near_duplicate_threshold == full.near_duplicate_threshold
    assert smoke.minimum_event_types == full.minimum_event_types


def _document(
    content: str,
    *,
    repository: str = "example/repository",
    path: str = "module.py",
) -> SourceDocumentRecord:
    digest = hashlib.sha256(content.encode()).hexdigest()
    return SourceDocumentRecord(
        document_id=stable_document_id(
            provider="test",
            revision="a" * 40,
            repository_identity=repository,
            relative_path=path,
            content_sha256=digest,
        ),
        source_provider="test",
        source_name=repository,
        source_revision="a" * 40,
        canonical_language="python",
        repository_identity=repository,
        relative_path=path,
        content_sha256=digest,
        content=content,
        license_spdx="MIT",
        ownership="third_party",
        attribution=repository,
        upstream_url=f"https://example.invalid/{repository}",
    )


def _curriculum(document: SourceDocumentRecord) -> list[CurriculumRecord]:
    adapter = PythonAdapter()
    parsed = adapter.parse(document.content, path=document.relative_path)
    units = tuple(adapter.to_units(parsed))
    records = records_from_units(units, parsed.index)
    return list(generate_document_curriculum(document, units, records))


def _permissive(**changes) -> AdmissionPolicy:
    values = dict(
        minimum_records=1,
        minimum_encoded_tokens=1,
        maximum_document_fraction=1.0,
        maximum_template_family_fraction=1.0,
        minimum_event_types=0,
        minimum_scope_depths=0,
        minimum_control_flow_types=0,
        minimum_identifier_shapes=0,
        minimum_literal_types=0,
        minimum_uncertainty_bands=0,
    )
    values.update(changes)
    return AdmissionPolicy(**values)


def test_split_assignment_clusters_copies_and_semantic_duplicates():
    first = _document("value=1\n", path="one.py")
    renamed = _document("value=1\n", path="renamed.py")
    semantic_copy = _document("value = 1\n", repository="other/repo", path="copy.py")
    corpus = CorpusValidationInput(
        documents=(first, renamed, semantic_copy),
        candidates=tuple(_curriculum(first) + _curriculum(renamed) + _curriculum(semantic_copy)),
    )

    report = validate_corpus(corpus, CountingTokenizer(), _permissive())

    assert len(set(report.split_manifest.values())) == 1
    assert set(report.split_manifest) == {
        first.document_id,
        renamed.document_id,
        semantic_copy.document_id,
    }
    assert assign_split("stable-group") == assign_split("stable-group")
    assert all(
        decision.split == report.split_manifest[decision.source_document_ids[0]]
        for decision in report.decisions
    )


def test_holdout_hash_quarantines_every_copy_before_expansion():
    first = _document("value = 1\n", path="one.py")
    copy = _document("value = 1\n", repository="other/repo", path="copy.py")
    corpus = CorpusValidationInput(
        documents=(first, copy),
        candidates=tuple(_curriculum(first) + _curriculum(copy)),
    )

    report = validate_corpus(
        corpus,
        CountingTokenizer(),
        _permissive(holdout_hashes=frozenset({first.content_sha256})),
    )

    assert not report.accepted_records
    assert {decision.reason_code for decision in report.decisions} == {
        "evaluation_holdout_collision"
    }
    assert report.distributions["holdout_collisions"] == 2


def test_exact_semantic_and_near_duplicates_have_stable_reason_codes():
    document = _document("def add(a, b):\n    return a + b\n")
    original = _curriculum(document)
    semantic = next(item for item in original if item.family == "code_to_semantic")
    paraphrase = replace(
        semantic,
        record_id="f" * 64,
        prompt=semantic.prompt.replace("Convert", "Please convert"),
    )
    corpus = CorpusValidationInput(
        documents=(document,),
        candidates=(semantic, semantic, paraphrase),
    )

    report = validate_corpus(corpus, CountingTokenizer(), _permissive())

    reasons = [decision.reason_code for decision in report.decisions]
    assert reasons.count("accepted") == 1
    assert "exact_record_duplicate" in reasons
    assert "canonical_semantic_duplicate" in reasons
    assert len({decision.decision_id for decision in report.decisions}) == 3


def test_document_and_template_caps_are_applied_deterministically():
    first = _document("a = 1\n", path="a.py")
    second = _document("b = 2\n", path="b.py")
    candidates = tuple(_curriculum(first) + _curriculum(second))
    corpus = CorpusValidationInput(documents=(first, second), candidates=candidates)

    document_capped = validate_corpus(
        corpus,
        CountingTokenizer(),
        _permissive(maximum_document_fraction=0.10),
    )
    template_capped = validate_corpus(
        corpus,
        CountingTokenizer(),
        _permissive(maximum_template_family_fraction=0.10),
    )

    assert "document_cap_exceeded" in {
        decision.reason_code for decision in document_capped.decisions
    }
    assert "template_family_cap_exceeded" in {
        decision.reason_code for decision in template_capped.decisions
    }


def test_hard_thresholds_and_missing_provenance_fail_admission():
    document = _document("value = 1\n")
    candidate = _curriculum(document)[0]
    missing_provenance = replace(candidate, record_id="e" * 64, provenance=())
    corpus = CorpusValidationInput(
        documents=(document,), candidates=(candidate, missing_provenance)
    )

    report = validate_corpus(
        corpus,
        CountingTokenizer(),
        _permissive(minimum_records=3, minimum_encoded_tokens=10_000),
    )

    assert "missing_provenance" in {
        decision.reason_code for decision in report.decisions
    }
    assert set(report.failure_codes) >= {
        "minimum_record_count_not_met",
        "minimum_encoded_tokens_not_met",
    }
    assert not report.passed


def test_diversity_gate_reports_all_required_dimensions():
    document = _document("value = 1\n")
    corpus = CorpusValidationInput(
        documents=(document,), candidates=tuple(_curriculum(document))
    )

    report = validate_corpus(
        corpus,
        CountingTokenizer(),
        _permissive(
            minimum_event_types=99,
            minimum_scope_depths=99,
            minimum_control_flow_types=99,
            minimum_identifier_shapes=99,
            minimum_literal_types=99,
            minimum_uncertainty_bands=99,
        ),
    )

    assert set(report.failure_codes) >= {
        "insufficient_event_diversity",
        "insufficient_scope_depth_diversity",
        "insufficient_control_flow_diversity",
        "insufficient_identifier_shape_diversity",
        "insufficient_literal_type_diversity",
        "insufficient_uncertainty_diversity",
    }
    assert set(report.distributions) >= {
        "family",
        "event",
        "scope_depth",
        "control_flow",
        "identifier_shape",
        "literal_type",
        "uncertainty",
        "source",
        "license",
        "split",
        "rejection_reason",
        "encoded_tokens",
    }
