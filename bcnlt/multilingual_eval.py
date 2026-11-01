from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from bcnlt.adapters.carousel import BCNLTMultilingualAdapter
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine
from bcnlt.languages import CANONICAL_LANGUAGE_IDS
from bcnlt.semantic_record import CanonicalIR, FrozenMap, SemanticTokenRecord, records_from_assessments


STRUCTURAL_CATEGORIES = (
    "event_classification",
    "definition_reference_linking",
    "scope_pairing",
    "control_flow",
    "code_to_semantic",
    "semantic_to_code",
)
CONTROL_FLOW_EVENTS = frozenset({"BRANCH", "LOOP", "RAISE", "RETURN"})


@dataclass(frozen=True)
class SemanticProjection:
    language: str
    events: tuple[str, ...]
    values: tuple[str | None, ...]
    scope_topology: tuple[tuple[str, ...], ...]
    links: tuple[tuple[str | None, str | None, str], ...]
    ast_relations: tuple[str | None, ...]
    ir: tuple[object, ...]
    uncertainty: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "events": list(self.events),
            "values": list(self.values),
            "scope_topology": [list(item) for item in self.scope_topology],
            "links": [list(item) for item in self.links],
            "ast_relations": list(self.ast_relations),
            "ir": list(self.ir),
            "uncertainty": [list(item) for item in self.uncertainty],
        }


@dataclass(frozen=True)
class MultilingualEvaluationCase:
    case_id: str
    language: str
    category: str
    expected: SemanticProjection
    observed: SemanticProjection | None
    runtime_error: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("multilingual evaluation cases require an ID")
        if self.category not in STRUCTURAL_CATEGORIES:
            raise ValueError(f"unsupported multilingual evaluation category: {self.category}")
        if self.expected.language != self.language:
            raise ValueError("golden projection language does not match its case")
        if self.observed is not None and self.observed.language != self.language:
            raise ValueError("observed projection language does not match its case")

    @property
    def passed(self) -> bool:
        return (
            self.runtime_error is None
            and self.observed is not None
            and score_semantic_projection(self.expected, self.observed, self.category)
        )


@dataclass(frozen=True)
class LanguageEvaluation:
    language: str
    category_scores: tuple[tuple[str, float], ...]
    structural_accuracy: float
    knowledge_state_accuracy: float
    uyr_resolution_accuracy: float
    false_certainty_rate: float
    raw_code_accuracy: float
    ordinary_instruction_accuracy: float
    runtime_failures: tuple[str, ...] = ()
    case_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        categories = dict(self.category_scores)
        if tuple(sorted(categories)) != tuple(sorted(STRUCTURAL_CATEGORIES)):
            raise ValueError("language evaluation must contain all six structural categories")
        for name, value in (
            *self.category_scores,
            ("structural_accuracy", self.structural_accuracy),
            ("knowledge_state_accuracy", self.knowledge_state_accuracy),
            ("uyr_resolution_accuracy", self.uyr_resolution_accuracy),
            ("false_certainty_rate", self.false_certainty_rate),
            ("raw_code_accuracy", self.raw_code_accuracy),
            ("ordinary_instruction_accuracy", self.ordinary_instruction_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        expected_macro = fmean(categories[name] for name in STRUCTURAL_CATEGORIES)
        if abs(expected_macro - self.structural_accuracy) > 1e-12:
            raise ValueError("structural accuracy must be the equal six-category macro average")

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "category_scores": dict(self.category_scores),
            "structural_accuracy": self.structural_accuracy,
            "knowledge_state_accuracy": self.knowledge_state_accuracy,
            "uyr_resolution_accuracy": self.uyr_resolution_accuracy,
            "false_certainty_rate": self.false_certainty_rate,
            "raw_code_accuracy": self.raw_code_accuracy,
            "ordinary_instruction_accuracy": self.ordinary_instruction_accuracy,
            "runtime_failures": list(self.runtime_failures),
            "case_counts": dict(self.case_counts),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LanguageEvaluation":
        return cls(
            language=str(value["language"]),
            category_scores=tuple(sorted(
                (str(key), float(score))
                for key, score in dict(value["category_scores"]).items()
            )),
            structural_accuracy=float(value["structural_accuracy"]),
            knowledge_state_accuracy=float(value["knowledge_state_accuracy"]),
            uyr_resolution_accuracy=float(value["uyr_resolution_accuracy"]),
            false_certainty_rate=float(value["false_certainty_rate"]),
            raw_code_accuracy=float(value["raw_code_accuracy"]),
            ordinary_instruction_accuracy=float(value["ordinary_instruction_accuracy"]),
            runtime_failures=tuple(str(item) for item in value.get("runtime_failures", ())),
            case_counts=tuple(sorted(
                (str(key), int(count)) for key, count in dict(value.get("case_counts", {})).items()
            )),
        )


@dataclass(frozen=True)
class MultilingualPromotionDecision:
    promoted: bool
    failure_codes: tuple[str, ...]
    structural_macro_average: float
    language_scores: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "promoted": self.promoted,
            "failure_codes": list(self.failure_codes),
            "structural_macro_average": self.structural_macro_average,
            "language_scores": dict(self.language_scores),
        }


def project_semantic_records(
    language: str, records: Sequence[SemanticTokenRecord]
) -> SemanticProjection:
    link_ids: dict[str, str] = {}
    for record in records:
        for value in (record.symbol_id, record.defines):
            if value is not None and value not in link_ids:
                link_ids[value] = f"symbol-{len(link_ids)}"
    return SemanticProjection(
        language=language,
        events=tuple(record.event for record in records),
        values=tuple(_semantic_value(record) for record in records),
        scope_topology=tuple(_scope_parts(record.scope) for record in records),
        links=tuple(_link_projection(record, link_ids) for record in records),
        ast_relations=tuple(record.ast_relation for record in records),
        ir=tuple(_ir_projection(record.ir) for record in records),
        uncertainty=tuple(
            (record.uncertainty_band.value, _metadata_value(record, "knowledge_state") or "")
            for record in records
        ),
    )


def project_source(
    source: str,
    *,
    language: str,
    path: str,
    dialect: str | None = None,
    adapter: BCNLTMultilingualAdapter | None = None,
) -> SemanticProjection:
    active = adapter or BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = active.parse(source, language=language, path=path, dialect=dialect)
    extraction = active.extract(document)
    return project_semantic_records(
        language, records_from_assessments(extraction.assessments)
    )


def score_semantic_projection(
    expected: SemanticProjection,
    observed: SemanticProjection,
    category: str,
) -> bool:
    if category not in STRUCTURAL_CATEGORIES:
        raise ValueError(f"unsupported multilingual evaluation category: {category}")
    if expected.language != observed.language:
        return False
    if category == "event_classification":
        return expected.events == observed.events
    if category == "definition_reference_linking":
        return _links_match(expected.links, observed.links)
    if category == "scope_pairing":
        return expected.scope_topology == observed.scope_topology
    if category == "control_flow":
        return _control_flow_projection(expected) == _control_flow_projection(observed)
    return (
        _complete_projection(expected) == _complete_projection(observed)
        and _links_match(expected.links, observed.links)
    )


def evaluate_language(
    language: str,
    cases: Iterable[MultilingualEvaluationCase],
    *,
    knowledge_state_accuracy: float,
    uyr_resolution_accuracy: float,
    false_certainty_rate: float,
    raw_code_accuracy: float,
    ordinary_instruction_accuracy: float,
    runtime_failures: Iterable[str] = (),
) -> LanguageEvaluation:
    selected = tuple(case for case in cases if case.language == language)
    scores = []
    counts = []
    case_runtime_failures = []
    for category in STRUCTURAL_CATEGORIES:
        category_cases = tuple(case for case in selected if case.category == category)
        if not category_cases:
            score = 0.0
        else:
            score = sum(case.passed for case in category_cases) / len(category_cases)
        scores.append((category, score))
        counts.append((category, len(category_cases)))
        case_runtime_failures.extend(
            f"{case.case_id}: {case.runtime_error}"
            for case in category_cases
            if case.runtime_error is not None
        )
    macro = fmean(score for _, score in scores)
    return LanguageEvaluation(
        language=language,
        category_scores=tuple(scores),
        structural_accuracy=macro,
        knowledge_state_accuracy=knowledge_state_accuracy,
        uyr_resolution_accuracy=uyr_resolution_accuracy,
        false_certainty_rate=false_certainty_rate,
        raw_code_accuracy=raw_code_accuracy,
        ordinary_instruction_accuracy=ordinary_instruction_accuracy,
        runtime_failures=tuple((*runtime_failures, *case_runtime_failures)),
        case_counts=tuple(counts),
    )


def evaluate_multilingual_promotion(
    baseline: Sequence[LanguageEvaluation],
    adapter: Sequence[LanguageEvaluation],
    *,
    minimum_category_accuracy: float = 0.80,
    minimum_language_macro: float = 0.90,
    minimum_knowledge_state_accuracy: float = 0.90,
    minimum_uyr_resolution_accuracy: float = 0.85,
    maximum_false_certainty_rate: float = 0.05,
    maximum_control_regression: float = 0.02,
    required_languages: Sequence[str] | None = None,
) -> MultilingualPromotionDecision:
    baseline_by_language = _unique_languages(baseline)
    adapter_by_language = _unique_languages(adapter)
    if set(baseline_by_language) != set(adapter_by_language):
        raise ValueError("baseline and adapter language sets differ")
    failures: list[str] = []
    if required_languages is not None:
        for language in required_languages:
            if language not in adapter_by_language:
                failures.append(f"{language}:missing_language")
    for language in sorted(adapter_by_language):
        result = adapter_by_language[language]
        for category, score in result.category_scores:
            if score < minimum_category_accuracy:
                failures.append(f"{language}:{category}_below_threshold")
        if result.structural_accuracy < minimum_language_macro:
            failures.append(f"{language}:structural_macro_below_threshold")
        if result.knowledge_state_accuracy < minimum_knowledge_state_accuracy:
            failures.append(f"{language}:knowledge_state_accuracy_below_threshold")
        if result.uyr_resolution_accuracy < minimum_uyr_resolution_accuracy:
            failures.append(f"{language}:uyr_resolution_accuracy_below_threshold")
        if result.false_certainty_rate > maximum_false_certainty_rate:
            failures.append(f"{language}:false_certainty_above_threshold")
        if result.runtime_failures:
            failures.append(f"{language}:runtime_failure")
    for metric in ("raw_code_accuracy", "ordinary_instruction_accuracy"):
        baseline_macro = fmean(getattr(item, metric) for item in baseline_by_language.values())
        adapter_macro = fmean(getattr(item, metric) for item in adapter_by_language.values())
        if baseline_macro - adapter_macro > maximum_control_regression + 1e-12:
            failures.append(f"{metric.removesuffix('_accuracy')}_regression")
    scores = tuple(
        (language, adapter_by_language[language].structural_accuracy)
        for language in sorted(adapter_by_language)
    )
    macro = fmean(score for _, score in scores) if scores else 0.0
    return MultilingualPromotionDecision(
        promoted=not failures,
        failure_codes=tuple(dict.fromkeys(failures)),
        structural_macro_average=macro,
        language_scores=scores,
    )


def load_language_evaluations(path: str | Path) -> tuple[LanguageEvaluation, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw.get("languages", raw)
    if not isinstance(rows, list):
        raise ValueError("multilingual evaluation report must contain a languages list")
    return tuple(LanguageEvaluation.from_dict(item) for item in rows)


def load_multilingual_eval_config(path: str | Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != "bcnlt-multilingual-eval-v1":
        raise ValueError("unsupported multilingual evaluation schema")
    if tuple(raw.get("categories", ())) != STRUCTURAL_CATEGORIES:
        raise ValueError("multilingual evaluation categories do not match the frozen order")
    weights = raw.get("category_weights")
    if not isinstance(weights, dict) or set(weights) != set(STRUCTURAL_CATEGORIES):
        raise ValueError("multilingual evaluation requires six category weights")
    if any(abs(float(value) - 1.0 / 6.0) > 1e-12 for value in weights.values()):
        raise ValueError("every multilingual evaluation category must have weight one sixth")
    if tuple(raw.get("languages", ())) != CANONICAL_LANGUAGE_IDS:
        raise ValueError("multilingual evaluation languages do not match the carousel")
    return raw


def _unique_languages(values: Sequence[LanguageEvaluation]) -> dict[str, LanguageEvaluation]:
    result = {item.language: item for item in values}
    if len(result) != len(values) or not result:
        raise ValueError("language evaluations must be nonempty and unique")
    return result


def _semantic_value(record: SemanticTokenRecord) -> str | None:
    if record.event not in {
        "LITERAL", "DEFINE", "REFERENCE", "CALL", "MEMBER_ACCESS", "OPERATOR"
    }:
        return None
    if record.event == "OPERATOR":
        return re.sub(r"\s+", "", record.surface)
    return record.surface


def _scope_parts(scope: str) -> tuple[str, ...]:
    return tuple(part for part in scope.replace("/", "::").split("::") if part)


def _metadata_value(record: SemanticTokenRecord, key: str) -> str | None:
    for name, value in record.metadata:
        if name == key:
            return str(value)
    return None


def _link_projection(
    record: SemanticTokenRecord, link_ids: Mapping[str, str]
) -> tuple[str | None, str | None, str]:
    state = _metadata_value(record, "knowledge_state")
    if state in {"provisional", "unresolved"}:
        resolution = "uncertain"
    elif record.symbol_id is None and record.defines is None:
        resolution = "null"
    else:
        resolution = "resolved"
    return (
        link_ids.get(record.symbol_id) if record.symbol_id is not None else None,
        link_ids.get(record.defines) if record.defines is not None else None,
        resolution,
    )


def _links_match(
    expected: Sequence[tuple[str | None, str | None, str]],
    observed: Sequence[tuple[str | None, str | None, str]],
) -> bool:
    if len(expected) != len(observed):
        return False
    for golden, actual in zip(expected, observed, strict=True):
        if golden[2] in {"uncertain", "null"}:
            if actual[2] != golden[2]:
                return False
            if golden[2] == "null" and actual[:2] != (None, None):
                return False
        elif actual != golden:
            return False
    return True


def _control_flow_projection(projection: SemanticProjection) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            projection.events[index],
            projection.values[index],
            projection.scope_topology[index],
            projection.ast_relations[index],
            projection.ir[index],
            projection.uncertainty[index],
        )
        for index in range(len(projection.events))
        if projection.events[index] in CONTROL_FLOW_EVENTS
    )


def _complete_projection(projection: SemanticProjection) -> tuple[object, ...]:
    return (
        projection.language,
        projection.events,
        projection.values,
        projection.scope_topology,
        projection.ast_relations,
        projection.ir,
        projection.uncertainty,
    )


def _ir_projection(ir: CanonicalIR | None) -> object:
    if ir is None:
        return None
    return (
        ir.kind,
        _frozen_json(ir.value),
        tuple(_ir_projection(child) for child in ir.children),
    )


def _frozen_json(value: object) -> object:
    if isinstance(value, FrozenMap):
        return tuple((key, _frozen_json(item)) for key, item in value.items)
    if isinstance(value, tuple):
        return tuple(_frozen_json(item) for item in value)
    return value
