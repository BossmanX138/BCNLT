from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .adapters.carousel import BCNLTMultilingualAdapter
from .adapters.tree_sitter import TreeSitterSemanticEngine
from .cache import adapter_cache_identity
from .core import ContextualUnit


@dataclass(frozen=True)
class MeaningProbe:
    question: str
    surface: str
    symbol_id: str
    target_line: int
    target_event: str
    scope: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    probe: MeaningProbe
    standard_hit: bool
    bcnlt_hit: bool
    standard_lines: tuple[int, int] | None
    bcnlt_lines: tuple[int, int] | None
    standard_context_units: int
    bcnlt_context_units: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["probe"] = self.probe.to_dict()
        return data


@dataclass(frozen=True)
class MeaningEvalReport:
    path: str
    probe_count: int
    standard_hit_count: int
    bcnlt_hit_count: int
    standard_accuracy: float
    bcnlt_accuracy: float
    results: tuple[RetrievalResult, ...]
    language: str = "python"
    dialect: str | None = None
    parser_bundle_sha256: str = "legacy-python"
    semantic_spec_sha256: str = "legacy-python"
    evaluation_kind: str = "extractor_self_consistency_retrieval"
    target_origin: str = "same_extractor_output"
    independent_semantic_oracle: bool = False
    promotion_eligible: bool = False

    @property
    def standard_retrieval_hit_rate(self) -> float:
        return self.standard_accuracy

    @property
    def bcnlt_retrieval_hit_rate(self) -> float:
        return self.bcnlt_accuracy

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "probe_count": self.probe_count,
            "standard_hit_count": self.standard_hit_count,
            "bcnlt_hit_count": self.bcnlt_hit_count,
            "standard_accuracy": self.standard_accuracy,
            "bcnlt_accuracy": self.bcnlt_accuracy,
            "standard_retrieval_hit_rate": self.standard_retrieval_hit_rate,
            "bcnlt_retrieval_hit_rate": self.bcnlt_retrieval_hit_rate,
            "evaluation_kind": self.evaluation_kind,
            "target_origin": self.target_origin,
            "independent_semantic_oracle": self.independent_semantic_oracle,
            "promotion_eligible": self.promotion_eligible,
            "legacy_metric_names": ["standard_accuracy", "bcnlt_accuracy"],
            "results": [result.to_dict() for result in self.results],
            "language": self.language,
            "dialect": self.dialect,
            "parser_bundle_sha256": self.parser_bundle_sha256,
            "semantic_spec_sha256": self.semantic_spec_sha256,
        }


def evaluate_meaning_source(
    source: str,
    path: str = "<memory>",
    max_probes: int = 50,
    standard_context_lines: int = 1,
    bcnlt_context_units: int = 2,
    *,
    language: str = "python",
    dialect: str | None = None,
) -> MeaningEvalReport:
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    identity = adapter_cache_identity(adapter, language, dialect)
    document = adapter.parse(source, language=language, path=path, dialect=dialect)
    units = list(adapter.extract(document).units)
    probes = _generate_definition_probes(units, max_probes=max_probes)
    results = tuple(
        _evaluate_probe(
            source,
            units,
            probe,
            standard_context_lines=standard_context_lines,
            bcnlt_context_units=bcnlt_context_units,
        )
        for probe in probes
    )
    standard_hits = sum(1 for result in results if result.standard_hit)
    bcnlt_hits = sum(1 for result in results if result.bcnlt_hit)
    probe_count = len(results)
    return MeaningEvalReport(
        path=path,
        probe_count=probe_count,
        standard_hit_count=standard_hits,
        bcnlt_hit_count=bcnlt_hits,
        standard_accuracy=(standard_hits / probe_count) if probe_count else 0.0,
        bcnlt_accuracy=(bcnlt_hits / probe_count) if probe_count else 0.0,
        results=results,
        language=language,
        dialect=dialect,
        parser_bundle_sha256=identity.parser_bundle_sha256,
        semantic_spec_sha256=identity.semantic_spec_sha256,
    )


def evaluate_meaning_file(
    path: str | Path,
    max_probes: int = 50,
    *,
    language: str = "python",
    dialect: str | None = None,
) -> MeaningEvalReport:
    file_path = Path(path)
    source = file_path.read_text(encoding="utf-8-sig")
    return evaluate_meaning_source(
        source, path=str(file_path), max_probes=max_probes,
        language=language, dialect=dialect,
    )


def _generate_definition_probes(
    units: list[ContextualUnit],
    max_probes: int,
) -> list[MeaningProbe]:
    probes: list[MeaningProbe] = []
    for unit in units:
        if unit.event != "DEFINE" or not unit.symbol_id:
            continue
        if unit.ast_relation == "arguments.args":
            continue
        question = f"Where is `{unit.surface}` defined in scope `{unit.scope}`?"
        probes.append(
            MeaningProbe(
                question=question,
                surface=unit.surface,
                symbol_id=unit.symbol_id,
                target_line=unit.span[0],
                target_event=unit.event,
                scope=unit.scope,
            )
        )
        if len(probes) >= max_probes:
            break
    return probes


def _evaluate_probe(
    source: str,
    units: list[ContextualUnit],
    probe: MeaningProbe,
    standard_context_lines: int,
    bcnlt_context_units: int,
) -> RetrievalResult:
    standard_lines = _standard_text_retrieve(source, probe, standard_context_lines)
    bcnlt_lines = _bcnlt_semantic_retrieve(units, probe, bcnlt_context_units)
    return RetrievalResult(
        probe=probe,
        standard_hit=_line_range_contains(standard_lines, probe.target_line),
        bcnlt_hit=_line_range_contains(bcnlt_lines, probe.target_line),
        standard_lines=standard_lines,
        bcnlt_lines=bcnlt_lines,
        standard_context_units=(standard_lines[1] - standard_lines[0] + 1) if standard_lines else 0,
        bcnlt_context_units=bcnlt_context_units * 2 + 1,
    )


def _standard_text_retrieve(
    source: str,
    probe: MeaningProbe,
    context_lines: int,
) -> tuple[int, int] | None:
    lines = source.splitlines()
    for index, line in enumerate(lines, start=1):
        if probe.surface in line:
            return (
                max(1, index - context_lines),
                min(len(lines), index + context_lines),
            )
    return None


def _bcnlt_semantic_retrieve(
    units: list[ContextualUnit],
    probe: MeaningProbe,
    context_units: int,
) -> tuple[int, int] | None:
    for index, unit in enumerate(units):
        if (
            unit.event == probe.target_event
            and unit.symbol_id == probe.symbol_id
            and unit.span[0] == probe.target_line
        ):
            start = max(0, index - context_units)
            end = min(len(units), index + context_units + 1)
            selected = units[start:end]
            return (
                min(item.span[0] for item in selected),
                max(item.span[2] for item in selected),
            )
    return None


def _line_range_contains(line_range: tuple[int, int] | None, line: int) -> bool:
    if line_range is None:
        return False
    return line_range[0] <= line <= line_range[1]
