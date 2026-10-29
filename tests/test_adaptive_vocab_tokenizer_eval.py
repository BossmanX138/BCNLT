from __future__ import annotations

from dataclasses import replace

import pytest

from bcnlt.canonical import canonical_record_bytes
from bcnlt.adaptive_vocab.config import AdaptiveVocabularyConfig
from bcnlt.adaptive_vocab.tokenizer_eval import (
    AdaptiveTokenizerEvaluation,
    LatencyMetrics,
    LanguageMetrics,
    SplitMetrics,
    apply_platform_artifact_digests,
    gate_failures,
    select_inventory,
    evaluate_tokenizer,
)
from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.semantic_record import ScoreBand, SemanticTokenRecord


def _config():
    return AdaptiveVocabularyConfig.load("configs/bcnlt_adaptive_vocab_v5.json")


def _split(compression: float, *, language_expansion: bool = False) -> SplitMetrics:
    v1_tokens = 1000
    adaptive_tokens = round(v1_tokens * (1 - compression))
    languages = (
        LanguageMetrics(
            "python",
            v1_tokens=100,
            adaptive_tokens=101 if language_expansion else 90,
            semantic_bytes=400,
        ),
    )
    return SplitMetrics(
        name="heldout",
        record_count=10,
        v1_semantic_tokens=v1_tokens,
        adaptive_semantic_tokens=adaptive_tokens,
        semantic_bytes=4000,
        compression=1 - adaptive_tokens / v1_tokens,
        v1_bytes_per_token=4.0,
        adaptive_bytes_per_token=4000 / adaptive_tokens,
        macro_tokens=100,
        macro_coverage=100 / adaptive_tokens,
        languages=languages,
    )


def _latency(slowdown: float = 0.0) -> LatencyMetrics:
    return LatencyMetrics(
        v1_encode_p50_ms=1.0,
        v1_encode_p95_ms=10.0,
        v1_decode_p50_ms=1.0,
        v1_decode_p95_ms=10.0,
        adaptive_encode_p50_ms=1.0,
        adaptive_encode_p95_ms=10.0 * (1 + slowdown),
        adaptive_decode_p50_ms=1.0,
        adaptive_decode_p95_ms=10.0,
        maximum_p95_slowdown=slowdown,
    )


def _evaluation(
    budget: int,
    compression: float,
    *,
    slowdown: float = 0.0,
    roundtrip_failures: int = 0,
    mode_identity_failures: int = 0,
    language_expansion: bool = False,
) -> AdaptiveTokenizerEvaluation:
    heldout = _split(compression, language_expansion=language_expansion)
    failures = gate_failures(
        heldout,
        _latency(slowdown),
        roundtrip_failures=roundtrip_failures,
        mode_identity_failures=mode_identity_failures,
        config=_config(),
    )
    return AdaptiveTokenizerEvaluation(
        schema_version="bcnlt-adaptive-tokenizer-eval-v1",
        manifest_sha256=f"{budget:064x}",
        budget=budget,
        train=replace(heldout, name="train"),
        heldout=heldout,
        roundtrip_failures=roundtrip_failures,
        mode_identity_failures=mode_identity_failures,
        latency=_latency(slowdown),
        peak_memory_bytes=123,
        platform_artifact_digests=(),
        gate_failures=failures,
        passed=not failures,
    )


def test_aggregate_compression_math_and_boundary_gates_are_exact():
    passing = _evaluation(256, 0.10, slowdown=0.10)
    low_compression = _evaluation(256, 0.099)
    high_latency = _evaluation(256, 0.20, slowdown=0.101)

    assert passing.heldout.compression == 1 - 900 / 1000
    assert passing.gate_failures == ()
    assert "compression_below_threshold" in low_compression.gate_failures
    assert "tokenizer_latency_above_threshold" in high_latency.gate_failures


def test_roundtrip_mode_identity_and_per_language_expansion_fail_stably():
    evaluation = _evaluation(
        256,
        0.20,
        roundtrip_failures=1,
        mode_identity_failures=1,
        language_expansion=True,
    )

    assert evaluation.gate_failures == (
        "candidate_roundtrip_failure",
        "per_language_expansion",
        "text_or_raw_code_regression",
    )


def test_smallest_inventory_within_one_point_of_best_compression_wins():
    evaluations = (
        _evaluation(256, 0.145),
        _evaluation(512, 0.151),
        _evaluation(1024, 0.152),
    )

    selected = select_inventory(evaluations, _config())

    assert selected.training_eligible is True
    assert selected.selected_budget == 256


def test_smaller_inventory_farther_than_one_point_loses_and_no_pass_stops():
    selected = select_inventory(
        (_evaluation(256, 0.13), _evaluation(512, 0.151)), _config()
    )
    failed = select_inventory(
        (_evaluation(256, 0.09), _evaluation(512, 0.099)), _config()
    )

    assert selected.selected_budget == 512
    assert failed.training_eligible is False
    assert failed.failure_codes == ("no_inventory_passed",)


def test_cross_platform_artifacts_compare_bytes_not_host_clocks():
    evaluation = _evaluation(256, 0.20)
    equal = apply_platform_artifact_digests(
        evaluation, {"windows": "a" * 64, "wsl": "a" * 64}
    )
    different = apply_platform_artifact_digests(
        evaluation, {"windows": "a" * 64, "wsl": "b" * 64}
    )

    assert equal.passed is True
    assert equal.platform_artifact_digests == (("windows", "a" * 64), ("wsl", "a" * 64))
    assert "cross_platform_artifact_mismatch" in different.gate_failures


def test_evaluation_uses_aggregate_counts_and_nearest_rank_p95():
    semantic_record = SemanticTokenRecord(
        event="REFERENCE",
        surface="value",
        scope="",
        span=(1, 0, 1, 5),
        symbol_id=None,
        defines=None,
        ast_relation=None,
        ir=None,
        recent_history=(),
        metadata=(),
        confidence_band=ScoreBand.HIGH,
        uncertainty_band=ScoreBand.LOW,
    )
    semantic = canonical_record_bytes(semantic_record).decode()
    curriculum = CurriculumRecord(
        record_id="a" * 64,
        family="test",
        template_id="test-v1",
        source_document_ids=("doc",),
        prompt_mode="raw_code",
        target_mode="semantic_code",
        prompt="value = 1\n",
        target=semantic,
        provenance=(("canonical_language", "python"),),
        split="train",
    )

    class Manifest:
        manifest_sha256 = "a" * 64
        v1_vocab_size = 100
        macros = (object(),)

    class FakeTokenizer:
        def __init__(self, semantic_length: int):
            self.manifest = Manifest()
            self.semantic_length = semantic_length
            self.values = {}

        def _encode(self, value):
            ids = [1, *value.encode(), 2]
            self.values[tuple(ids)] = value
            return ids

        encode_text = _encode
        encode_raw_code = _encode

        def encode_records(self, records):
            value = b"".join(canonical_record_bytes(item) for item in records).decode()
            ids = list(range(self.semantic_length))
            self.values[tuple(ids)] = value
            return ids

        def decode(self, ids):
            return self.values[tuple(ids)]

    counter = iter(index / 1000 for index in range(1000))
    evaluation = evaluate_tokenizer(
        FakeTokenizer(10),
        FakeTokenizer(8),
        (curriculum,),
        (replace(curriculum, record_id="b" * 64, split="validation"),),
        _config(),
        clock=lambda: next(counter),
        memory_probe=lambda: 321,
    )

    assert evaluation.heldout.compression == 1 - 8 / 10
    assert evaluation.heldout.languages[0].adaptive_tokens == 8
    assert evaluation.latency.v1_encode_p95_ms == pytest.approx(1.0)
    assert evaluation.peak_memory_bytes == 321
