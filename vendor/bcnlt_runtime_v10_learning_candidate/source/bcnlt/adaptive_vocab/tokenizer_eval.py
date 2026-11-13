from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

from bcnlt.canonical import canonical_record_bytes, record_from_canonical_bytes
from bcnlt.corpus.curriculum import CurriculumRecord

from .config import AdaptiveVocabularyConfig


@dataclass(frozen=True, slots=True)
class LanguageMetrics:
    language: str
    v1_tokens: int
    adaptive_tokens: int
    semantic_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "v1_tokens": self.v1_tokens,
            "adaptive_tokens": self.adaptive_tokens,
            "semantic_bytes": self.semantic_bytes,
            "compression": 1 - self.adaptive_tokens / self.v1_tokens
            if self.v1_tokens
            else 0.0,
        }


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    name: str
    record_count: int
    v1_semantic_tokens: int
    adaptive_semantic_tokens: int
    semantic_bytes: int
    compression: float
    v1_bytes_per_token: float
    adaptive_bytes_per_token: float
    macro_tokens: int
    macro_coverage: float
    languages: tuple[LanguageMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "record_count": self.record_count,
            "v1_semantic_tokens": self.v1_semantic_tokens,
            "adaptive_semantic_tokens": self.adaptive_semantic_tokens,
            "semantic_bytes": self.semantic_bytes,
            "compression": self.compression,
            "v1_bytes_per_token": self.v1_bytes_per_token,
            "adaptive_bytes_per_token": self.adaptive_bytes_per_token,
            "macro_tokens": self.macro_tokens,
            "macro_coverage": self.macro_coverage,
            "per_language": {
                item.language: item.to_dict() for item in self.languages
            },
        }


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    v1_encode_p50_ms: float
    v1_encode_p95_ms: float
    v1_decode_p50_ms: float
    v1_decode_p95_ms: float
    adaptive_encode_p50_ms: float
    adaptive_encode_p95_ms: float
    adaptive_decode_p50_ms: float
    adaptive_decode_p95_ms: float
    maximum_p95_slowdown: float

    def to_dict(self) -> dict[str, float]:
        return {
            field: float(getattr(self, field)) for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class AdaptiveTokenizerEvaluation:
    schema_version: str
    manifest_sha256: str
    budget: int
    train: SplitMetrics
    heldout: SplitMetrics
    roundtrip_failures: int
    mode_identity_failures: int
    latency: LatencyMetrics
    peak_memory_bytes: int
    platform_artifact_digests: tuple[tuple[str, str], ...]
    gate_failures: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
            "budget": self.budget,
            "train": self.train.to_dict(),
            "heldout": self.heldout.to_dict(),
            "roundtrip_failures": self.roundtrip_failures,
            "mode_identity_failures": self.mode_identity_failures,
            "latency": self.latency.to_dict(),
            "peak_memory_bytes": self.peak_memory_bytes,
            "platform_artifact_digests": dict(self.platform_artifact_digests),
            "gate_failures": list(self.gate_failures),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class InventorySelection:
    selected_budget: int | None
    selected_manifest_sha256: str | None
    best_compression: float | None
    training_eligible: bool
    failure_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "bcnlt-adaptive-inventory-selection-v1",
            "selected_budget": self.selected_budget,
            "selected_manifest_sha256": self.selected_manifest_sha256,
            "best_compression": self.best_compression,
            "training_eligible": self.training_eligible,
            "failure_codes": list(self.failure_codes),
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _slowdown(adaptive: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0 if adaptive <= 0 else float("inf")
    return adaptive / baseline - 1


def gate_failures(
    heldout: SplitMetrics,
    latency: LatencyMetrics,
    *,
    roundtrip_failures: int,
    mode_identity_failures: int,
    config: AdaptiveVocabularyConfig,
) -> tuple[str, ...]:
    failures = set()
    if roundtrip_failures:
        failures.add("candidate_roundtrip_failure")
    if mode_identity_failures:
        failures.add("text_or_raw_code_regression")
    if heldout.compression + 1e-12 < config.minimum_heldout_compression:
        failures.add("compression_below_threshold")
    if any(item.adaptive_tokens > item.v1_tokens for item in heldout.languages):
        failures.add("per_language_expansion")
    if (
        latency.maximum_p95_slowdown
        > config.maximum_p95_latency_slowdown + 1e-12
    ):
        failures.add("tokenizer_latency_above_threshold")
    return tuple(sorted(failures))


def apply_platform_artifact_digests(
    evaluation: AdaptiveTokenizerEvaluation, digests: Mapping[str, str]
) -> AdaptiveTokenizerEvaluation:
    ordered = tuple(sorted((str(key), str(value)) for key, value in digests.items()))
    failures = set(evaluation.gate_failures)
    failures.discard("cross_platform_artifact_mismatch")
    if len(ordered) < 2 or len({digest for _, digest in ordered}) != 1:
        failures.add("cross_platform_artifact_mismatch")
    final = tuple(sorted(failures))
    return replace(
        evaluation,
        platform_artifact_digests=ordered,
        gate_failures=final,
        passed=not final,
    )


def select_inventory(
    evaluations: Sequence[AdaptiveTokenizerEvaluation],
    config: AdaptiveVocabularyConfig,
) -> InventorySelection:
    passing = [item for item in evaluations if item.passed]
    if not passing:
        return InventorySelection(None, None, None, False, ("no_inventory_passed",))
    best = max(item.heldout.compression for item in passing)
    near_best = [item for item in passing if best - item.heldout.compression <= 0.01]
    selected = min(near_best, key=lambda item: item.budget)
    return InventorySelection(
        selected.budget,
        selected.manifest_sha256,
        best,
        True,
        (),
    )


def _semantic_records(value: str):
    result = []
    for line in value.splitlines():
        if line.startswith("{"):
            result.append(record_from_canonical_bytes((line + "\n").encode("utf-8")))
    return tuple(result)


def _mode_call(tokenizer, mode: str, value: str):
    if mode == "raw_code":
        return tokenizer.encode_raw_code(value), value
    if mode == "semantic_code":
        records = _semantic_records(value)
        canonical = b"".join(canonical_record_bytes(item) for item in records).decode()
        return tokenizer.encode_records(records), canonical
    return tokenizer.encode_text(value), value


def evaluate_tokenizer(
    v1,
    adaptive,
    train_records: Sequence[CurriculumRecord],
    heldout_records: Sequence[CurriculumRecord],
    config: AdaptiveVocabularyConfig,
    *,
    clock: Callable[[], float] = time.perf_counter,
    memory_probe: Callable[[], int] = lambda: 0,
) -> AdaptiveTokenizerEvaluation:
    timings = {key: [] for key in ("v1_encode", "v1_decode", "adaptive_encode", "adaptive_decode")}
    roundtrip_failures = 0
    identity_failures = 0
    peak_memory = int(memory_probe())

    def evaluate_split(name: str, records: Sequence[CurriculumRecord]) -> SplitMetrics:
        nonlocal roundtrip_failures, identity_failures, peak_memory
        v1_total = adaptive_total = byte_total = macro_total = 0
        language_values: dict[str, list[int]] = {}
        for record in sorted(records, key=lambda item: item.record_id):
            language = dict(record.provenance).get("canonical_language", "unknown")
            for mode, value in ((record.prompt_mode, record.prompt), (record.target_mode, record.target)):
                start = clock()
                v1_ids, expected = _mode_call(v1, mode, value)
                timings["v1_encode"].append((clock() - start) * 1000)
                start = clock()
                adaptive_ids, adaptive_expected = _mode_call(adaptive, mode, value)
                timings["adaptive_encode"].append((clock() - start) * 1000)
                start = clock()
                v1_decoded = v1.decode(v1_ids)
                timings["v1_decode"].append((clock() - start) * 1000)
                start = clock()
                adaptive_decoded = adaptive.decode(adaptive_ids)
                timings["adaptive_decode"].append((clock() - start) * 1000)
                if v1_decoded != expected or adaptive_decoded != adaptive_expected:
                    roundtrip_failures += 1
                if mode in {"text", "raw_code"} and v1_ids != adaptive_ids:
                    identity_failures += 1
                if mode == "semantic_code":
                    encoded_bytes = len(expected.encode("utf-8"))
                    v1_total += len(v1_ids)
                    adaptive_total += len(adaptive_ids)
                    byte_total += encoded_bytes
                    macro_total += sum(
                        token_id >= adaptive.manifest.v1_vocab_size
                        for token_id in adaptive_ids
                    )
                    bucket = language_values.setdefault(str(language), [0, 0, 0])
                    bucket[0] += len(v1_ids)
                    bucket[1] += len(adaptive_ids)
                    bucket[2] += encoded_bytes
                peak_memory = max(peak_memory, int(memory_probe()))
        languages = tuple(
            LanguageMetrics(language, values[0], values[1], values[2])
            for language, values in sorted(language_values.items())
        )
        return SplitMetrics(
            name=name,
            record_count=len(records),
            v1_semantic_tokens=v1_total,
            adaptive_semantic_tokens=adaptive_total,
            semantic_bytes=byte_total,
            compression=1 - adaptive_total / v1_total if v1_total else 0.0,
            v1_bytes_per_token=byte_total / v1_total if v1_total else 0.0,
            adaptive_bytes_per_token=byte_total / adaptive_total if adaptive_total else 0.0,
            macro_tokens=macro_total,
            macro_coverage=macro_total / adaptive_total if adaptive_total else 0.0,
            languages=languages,
        )

    # Warm both paths without including the warm-up in measurements.
    ordered_all = tuple(sorted((*train_records, *heldout_records), key=lambda item: item.record_id))
    if ordered_all:
        first = ordered_all[0]
        _mode_call(v1, first.prompt_mode, first.prompt)
        _mode_call(adaptive, first.prompt_mode, first.prompt)

    train = evaluate_split("train", train_records)
    heldout = evaluate_split("heldout", heldout_records)
    latency = LatencyMetrics(
        v1_encode_p50_ms=_percentile(timings["v1_encode"], 0.50),
        v1_encode_p95_ms=_percentile(timings["v1_encode"], 0.95),
        v1_decode_p50_ms=_percentile(timings["v1_decode"], 0.50),
        v1_decode_p95_ms=_percentile(timings["v1_decode"], 0.95),
        adaptive_encode_p50_ms=_percentile(timings["adaptive_encode"], 0.50),
        adaptive_encode_p95_ms=_percentile(timings["adaptive_encode"], 0.95),
        adaptive_decode_p50_ms=_percentile(timings["adaptive_decode"], 0.50),
        adaptive_decode_p95_ms=_percentile(timings["adaptive_decode"], 0.95),
        maximum_p95_slowdown=max(
            _slowdown(
                _percentile(timings["adaptive_encode"], 0.95),
                _percentile(timings["v1_encode"], 0.95),
            ),
            _slowdown(
                _percentile(timings["adaptive_decode"], 0.95),
                _percentile(timings["v1_decode"], 0.95),
            ),
        ),
    )
    failures = gate_failures(
        heldout,
        latency,
        roundtrip_failures=roundtrip_failures,
        mode_identity_failures=identity_failures,
        config=config,
    )
    return AdaptiveTokenizerEvaluation(
        schema_version="bcnlt-adaptive-tokenizer-eval-v1",
        manifest_sha256=adaptive.manifest.manifest_sha256,
        budget=len(adaptive.manifest.macros),
        train=train,
        heldout=heldout,
        roundtrip_failures=roundtrip_failures,
        mode_identity_failures=identity_failures,
        latency=latency,
        peak_memory_bytes=peak_memory,
        platform_artifact_digests=(),
        gate_failures=failures,
        passed=not failures,
    )
