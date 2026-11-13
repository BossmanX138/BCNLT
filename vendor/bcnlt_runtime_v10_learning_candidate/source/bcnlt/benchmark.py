from __future__ import annotations

import concurrent.futures
import io
import re
import time
import tokenize
from dataclasses import dataclass
from pathlib import Path

from .adapters.python import PythonAdapter


@dataclass(frozen=True)
class BenchmarkReport:
    path: str
    standard_token_count: int
    standard_tokenizer_name: str
    lexical_token_count: int
    event_count: int
    uncertain_event_count: int
    uyr_activation_count: int
    wildcard_count: int
    elapsed_ms: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "path": self.path,
            "standard_token_count": self.standard_token_count,
            "standard_tokenizer_name": self.standard_tokenizer_name,
            "lexical_token_count": self.lexical_token_count,
            "event_count": self.event_count,
            "uncertain_event_count": self.uncertain_event_count,
            "uyr_activation_count": self.uyr_activation_count,
            "wildcard_count": self.wildcard_count,
            "event_to_standard_ratio": self.event_to_standard_ratio,
            "elapsed_ms": self.elapsed_ms,
        }

    @property
    def event_to_standard_ratio(self) -> float:
        if self.standard_token_count == 0:
            return 0.0
        return self.event_count / self.standard_token_count


@dataclass(frozen=True)
class ProjectBenchmarkReport:
    root: str
    file_count: int
    total_standard_token_count: int
    total_lexical_token_count: int
    total_event_count: int
    total_uncertain_events: int
    total_uyr_activations: int
    total_wildcards: int
    elapsed_ms: float
    workers: int
    files: tuple[BenchmarkReport, ...]

    @property
    def event_to_standard_ratio(self) -> float:
        if self.total_standard_token_count == 0:
            return 0.0
        return self.total_event_count / self.total_standard_token_count

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "file_count": self.file_count,
            "total_standard_token_count": self.total_standard_token_count,
            "total_lexical_token_count": self.total_lexical_token_count,
            "total_event_count": self.total_event_count,
            "total_uncertain_events": self.total_uncertain_events,
            "total_uyr_activations": self.total_uyr_activations,
            "total_wildcards": self.total_wildcards,
            "event_to_standard_ratio": self.event_to_standard_ratio,
            "elapsed_ms": self.elapsed_ms,
            "workers": self.workers,
            "files": [file.to_dict() for file in self.files],
        }


@dataclass(frozen=True)
class StandardTokenCount:
    count: int
    tokenizer_name: str


def lexical_token_count(source: str) -> int:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return sum(1 for token in tokens if token.type not in {tokenize.ENCODING, tokenize.ENDMARKER})


def standard_token_count(source: str) -> StandardTokenCount:
    """Count ordinary tokenizer units, preferring tiktoken when installed."""

    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        return StandardTokenCount(len(encoding.encode(source)), "tiktoken:cl100k_base")
    except Exception:
        pieces = re.findall(r"\w+|[^\w\s]|\s+", source, flags=re.UNICODE)
        return StandardTokenCount(len([piece for piece in pieces if piece]), "regex-bpe-style-fallback")


def benchmark_source(
    source: str,
    path: str = "<memory>",
    confidence_threshold: float = 0.72,
) -> BenchmarkReport:
    start = time.perf_counter()
    adapter = PythonAdapter()
    document = adapter.parse(source, path=path)
    extraction = adapter.extract(document)
    units = list(extraction.units)
    scores = [assessment.final_score for assessment in extraction.assessments]
    standard = standard_token_count(source)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return BenchmarkReport(
        path=path,
        standard_token_count=standard.count,
        standard_tokenizer_name=standard.tokenizer_name,
        lexical_token_count=lexical_token_count(source),
        event_count=len(units),
        uncertain_event_count=sum(1 for score in scores if score.uncertainty > 0.25),
        uyr_activation_count=sum(1 for score in scores if score.confidence < confidence_threshold),
        wildcard_count=sum(1 for unit in units if unit.event == "UNKNOWN_SEMANTIC"),
        elapsed_ms=elapsed_ms,
    )


def benchmark_project(
    root: str | Path,
    workers: int = 1,
    confidence_threshold: float = 0.72,
) -> ProjectBenchmarkReport:
    root_path = Path(root)
    paths = sorted(path for path in root_path.rglob("*.py") if path.is_file())
    start = time.perf_counter()
    if workers > 1 and len(paths) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            files = tuple(
                pool.map(
                    _benchmark_path_for_pool,
                    [(str(path), confidence_threshold) for path in paths],
                )
            )
    else:
        files = tuple(_benchmark_path(path, confidence_threshold) for path in paths)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return ProjectBenchmarkReport(
        root=str(root_path),
        file_count=len(files),
        total_standard_token_count=sum(file.standard_token_count for file in files),
        total_lexical_token_count=sum(file.lexical_token_count for file in files),
        total_event_count=sum(file.event_count for file in files),
        total_uncertain_events=sum(file.uncertain_event_count for file in files),
        total_uyr_activations=sum(file.uyr_activation_count for file in files),
        total_wildcards=sum(file.wildcard_count for file in files),
        elapsed_ms=elapsed_ms,
        workers=workers,
        files=files,
    )


def trace_comparison(source: str, path: str = "<memory>", max_units: int = 24) -> str:
    adapter = PythonAdapter()
    document = adapter.parse(source, path=path)
    units = list(adapter.to_units(document))
    standard = standard_token_count(source)
    lexical_preview = _lexical_preview(source, max_units=max_units)
    semantic_preview = "\n".join(
        f"{unit.event} {unit.surface} ({unit.scope})" for unit in units[:max_units]
    )
    return (
        f"Standard tokens ({standard.tokenizer_name}): {standard.count}\n"
        f"Python lexical preview: {lexical_preview}\n\n"
        "BCNLT semantic events:\n"
        f"{semantic_preview}\n"
    )


def _benchmark_path_for_pool(args: tuple[str, float]) -> BenchmarkReport:
    path, confidence_threshold = args
    return _benchmark_path(Path(path), confidence_threshold)


def _benchmark_path(path: Path, confidence_threshold: float) -> BenchmarkReport:
    source = path.read_text(encoding="utf-8")
    return benchmark_source(source, path=str(path), confidence_threshold=confidence_threshold)


def _lexical_preview(source: str, max_units: int) -> str:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    pieces = [
        token.string
        for token in tokens
        if token.type not in {tokenize.ENCODING, tokenize.ENDMARKER}
        and token.string.strip()
    ]
    return " | ".join(pieces[:max_units])
