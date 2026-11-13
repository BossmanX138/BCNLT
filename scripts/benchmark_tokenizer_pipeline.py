"""Bounded performance probe for the frozen tokenizer and semantic sidecar."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_TOKENIZERS_SITE = PROJECT_ROOT / "vendor" / "tokenizers_runtime_0_23"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PINNED_TOKENIZERS_SITE))

from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline  # noqa: E402
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402


def _time(callable_, repeats: int) -> list[float]:
    elapsed: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        elapsed.append(time.perf_counter() - started)
    return elapsed


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_times = _time(NativeTokenizerBinding.load, 3)
    binding = NativeTokenizerBinding.load()
    samples = [
        f"def function_{index}(value):\n    return value + {index}\n" for index in range(1000)
    ]
    started = time.perf_counter()
    individual = [binding.encode(sample) for sample in samples]
    individual_seconds = time.perf_counter() - started
    started = time.perf_counter()
    batch = binding.encode_batch(samples)
    batch_seconds = time.perf_counter() - started
    if batch != individual:
        raise RuntimeError("batch and individual tokenizer output differ")

    pipeline_init = _time(lambda: BCNLTSourcePipeline(binding), 3)
    pipeline = BCNLTSourcePipeline(binding)
    source = "def add(left, right):\n    return left + right\n"
    sidecar_times = _time(
        lambda: pipeline.prepare_semantic(source, language="python", path="audit.py"),
        8,
    )
    reference = pipeline.prepare_semantic(source, language="python", path="audit.py")
    persistent = BCNLTSourcePipeline(binding, persistent_sidecars=True)
    try:
        persistent_result = persistent.prepare_semantic(
            source, language="python", path="audit.py"
        )
        if persistent_result != reference:
            raise RuntimeError("persistent and one-shot semantic outputs differ")
        persistent_times = _time(
            lambda: persistent.prepare_semantic(
                source, language="python", path="audit.py"
            ),
            32,
        )
    finally:
        persistent.close()

    threaded = BCNLTSourcePipeline(binding, persistent_sidecars=True)
    threaded_count = 64
    threaded_workers = 4
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=threaded_workers) as executor:
            threaded_results = list(
                executor.map(
                    lambda _: threaded.prepare_semantic(
                        source, language="python", path="audit.py"
                    ),
                    range(threaded_count),
                )
            )
        if any(result != reference for result in threaded_results):
            raise RuntimeError("threaded persistent semantic outputs differ")
    finally:
        threaded.close()
    threaded_seconds = time.perf_counter() - started
    result = {
        "schema_version": "bosscrafts-tokenizer-pipeline-benchmark-v2",
        "tokenizer_load": _summary(load_times),
        "encode_1000_individual_seconds": individual_seconds,
        "encode_1000_batch_seconds": batch_seconds,
        "batch_speedup": individual_seconds / batch_seconds,
        "pipeline_initialization": _summary(pipeline_init),
        "semantic_sidecar_one_source": _summary(sidecar_times),
        "semantic_sidecar_sources_per_second": 1.0 / statistics.mean(sidecar_times),
        "persistent_semantic_sidecar_one_source": _summary(persistent_times),
        "persistent_semantic_sidecar_sources_per_second": 1.0
        / statistics.mean(persistent_times),
        "persistent_sidecar_speedup": statistics.mean(sidecar_times)
        / statistics.mean(persistent_times),
        "threaded_persistent": {
            "workers": threaded_workers,
            "sources": threaded_count,
            "seconds": threaded_seconds,
            "sources_per_second": threaded_count / threaded_seconds,
        },
        "semantic_equivalence": True,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
