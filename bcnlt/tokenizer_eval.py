from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


_CATEGORIES = frozenset(
    {
        "tokenizer",
        "structural",
        "semantic_to_code",
        "raw_code",
        "ordinary_instruction",
    }
)


@dataclass(frozen=True)
class GenerationSettings:
    seed: int
    temperature: float
    max_new_tokens: int

    def to_dict(self) -> dict[str, int | float]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    category: str
    prompt: str
    expected: str

    def __post_init__(self) -> None:
        if self.category not in _CATEGORIES:
            raise ValueError(f"unsupported evaluation category: {self.category}")
        if not self.case_id or not self.prompt:
            raise ValueError("evaluation cases require an ID and prompt")


@dataclass(frozen=True)
class EvaluationSuite:
    schema_version: str
    settings: GenerationSettings
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    prompt: str
    expected: str
    output: str
    passed: bool
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class EvaluationRun:
    identity: str
    settings: GenerationSettings
    results: tuple[CaseResult, ...]
    metrics: dict[str, float]
    runtime_failures: tuple[str, ...]
    heldout_loss: float

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "settings": self.settings.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "metrics": dict(sorted(self.metrics.items())),
            "runtime_failures": list(self.runtime_failures),
            "heldout_loss": self.heldout_loss,
        }


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    failure_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"promoted": self.promoted, "failure_codes": list(self.failure_codes)}


def load_evaluation_suite(path_value: str | Path) -> EvaluationSuite:
    path = Path(path_value)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "bcnlt-tokenizer-eval-v1":
        raise ValueError("unsupported tokenizer evaluation schema")
    settings = GenerationSettings(**raw["settings"])
    cases = tuple(EvaluationCase(**item) for item in raw["cases"])
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("evaluation cases must have unique IDs")
    if {case.category for case in cases} != _CATEGORIES:
        raise ValueError("evaluation suite must cover every frozen category")
    return EvaluationSuite(raw["schema_version"], settings, cases)


def evaluate_model(
    identity: str,
    cases: Sequence[EvaluationCase],
    settings: GenerationSettings,
    generate: Callable[[str, GenerationSettings], str],
    *,
    heldout_loss: float,
) -> EvaluationRun:
    results = []
    failures = []
    for case in cases:
        output = ""
        error = None
        try:
            output = generate(case.prompt, settings)
            passed = _score(case, output)
        except Exception as exc:
            passed = False
            error = f"{type(exc).__name__}: {exc}"
            failures.append(f"{case.case_id}: {error}")
        results.append(
            CaseResult(
                case.case_id,
                case.category,
                case.prompt,
                case.expected,
                output,
                passed,
                error,
            )
        )
    metrics = {}
    for category in sorted(_CATEGORIES):
        selected = [result for result in results if result.category == category]
        metrics[f"{category}_accuracy"] = (
            sum(result.passed for result in selected) / len(selected) if selected else 0.0
        )
    return EvaluationRun(
        identity,
        settings,
        tuple(results),
        metrics,
        tuple(failures),
        float(heldout_loss),
    )


def evaluate_promotion(
    baseline: EvaluationRun,
    adapter: EvaluationRun,
    *,
    maximum_control_regression: float = 0.02,
    minimum_structural_accuracy: float = 0.90,
) -> PromotionDecision:
    if baseline.settings != adapter.settings:
        raise ValueError("base and adapter generation settings differ")
    baseline_prompts = tuple((item.case_id, item.prompt) for item in baseline.results)
    adapter_prompts = tuple((item.case_id, item.prompt) for item in adapter.results)
    if baseline_prompts != adapter_prompts:
        raise ValueError("base and adapter prompts differ")
    failures = []
    if not adapter.heldout_loss < baseline.heldout_loss:
        failures.append("heldout_loss_not_improved")
    if adapter.metrics["structural_accuracy"] < minimum_structural_accuracy:
        failures.append("structural_accuracy_below_threshold")
    if adapter.runtime_failures:
        failures.append("runtime_failure")
    for category in ("raw_code", "ordinary_instruction"):
        key = f"{category}_accuracy"
        if baseline.metrics[key] - adapter.metrics[key] > maximum_control_regression:
            failures.append(f"{category}_regression")
    return PromotionDecision(not failures, tuple(failures))


def decode_generated_stream(tokenizer, token_ids: Sequence[int]) -> str:
    """Decode one generated BCNLT stream and ignore tokens after its boundary."""
    token_to_id = getattr(getattr(tokenizer, "manifest", None), "token_to_id", {})
    stream_end = token_to_id.get("<|bcnlt:stream:end|>")
    if not isinstance(stream_end, int):
        raise ValueError("tokenizer manifest has no BCNLT stream end token")
    try:
        boundary = list(token_ids).index(stream_end)
    except ValueError as exc:
        raise ValueError("generated completion has no BCNLT stream end token") from exc
    return tokenizer.decode(list(token_ids)[: boundary + 1])


def _score(case: EvaluationCase, output: str) -> bool:
    if case.category == "semantic_to_code":
        try:
            return ast.dump(ast.parse(output), include_attributes=False) == ast.dump(
                ast.parse(case.expected), include_attributes=False
            )
        except SyntaxError:
            return False
    if case.category in {"raw_code", "tokenizer"}:
        return _line_endings(output) == _line_endings(case.expected)
    return " ".join(output.split()) == " ".join(case.expected.split())


def _line_endings(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")
