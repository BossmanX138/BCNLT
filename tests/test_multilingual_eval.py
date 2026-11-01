from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bcnlt.multilingual_eval import (
    STRUCTURAL_CATEGORIES,
    LanguageEvaluation,
    MultilingualEvaluationCase,
    SemanticProjection,
    evaluate_language,
    evaluate_multilingual_promotion,
    load_multilingual_eval_config,
    project_source,
    score_semantic_projection,
)
from bcnlt.tokenizer_eval import semantic_projection_equivalent


PYTHON_SOURCE = "def add(a, b):\n    return a + b\n"


def _projection(source: str = PYTHON_SOURCE) -> SemanticProjection:
    return project_source(source, language="python", path="fixture.py")


def _language(
    language: str,
    *,
    category_scores: dict[str, float] | None = None,
    knowledge: float = 1.0,
    uyr: float = 1.0,
    false_certainty: float = 0.0,
    raw: float = 1.0,
    instruction: float = 1.0,
    runtime_failures: tuple[str, ...] = (),
    case_count: int = 1,
) -> LanguageEvaluation:
    scores = category_scores or {name: 1.0 for name in STRUCTURAL_CATEGORIES}
    macro = sum(scores[name] for name in STRUCTURAL_CATEGORIES) / 6.0
    return LanguageEvaluation(
        language=language,
        category_scores=tuple((name, scores[name]) for name in STRUCTURAL_CATEGORIES),
        structural_accuracy=macro,
        knowledge_state_accuracy=knowledge,
        uyr_resolution_accuracy=uyr,
        false_certainty_rate=false_certainty,
        raw_code_accuracy=raw,
        ordinary_instruction_accuracy=instruction,
        runtime_failures=runtime_failures,
        case_counts=tuple((name, case_count) for name in STRUCTURAL_CATEGORIES),
    )


def test_each_category_has_equal_weight_and_exact_thresholds():
    expected = _projection()
    cases = []
    for category in STRUCTURAL_CATEGORIES:
        for index in range(5):
            cases.append(
                MultilingualEvaluationCase(
                    f"{category}-{index}",
                    "python",
                    category,
                    expected,
                        None if index == 0 else expected,
                )
            )
    result = evaluate_language(
        "python",
        cases,
        knowledge_state_accuracy=1.0,
        uyr_resolution_accuracy=1.0,
        false_certainty_rate=0.0,
        raw_code_accuracy=1.0,
        ordinary_instruction_accuracy=1.0,
    )

    assert dict(result.category_scores) == {name: 0.8 for name in STRUCTURAL_CATEGORIES}
    assert result.structural_accuracy == pytest.approx(0.8)

    scores = {name: 1.0 for name in STRUCTURAL_CATEGORIES}
    scores["event_classification"] = 0.79
    rejected = evaluate_multilingual_promotion(
        [_language("python")], [_language("python", category_scores=scores)]
    )
    assert "python:event_classification_below_threshold" in rejected.failure_codes
    assert "python:structural_macro_below_threshold" not in rejected.failure_codes


def test_uncertain_and_null_links_only_pass_when_the_golden_link_matches():
    projection = _projection()
    uncertain = replace(projection, links=((None, None, "uncertain"),))
    null = replace(projection, links=((None, None, "null"),))
    resolved = replace(projection, links=(("module.value", "module.value", "resolved"),))

    assert score_semantic_projection(uncertain, uncertain, "definition_reference_linking")
    assert score_semantic_projection(null, null, "definition_reference_linking")
    assert not score_semantic_projection(uncertain, resolved, "definition_reference_linking")
    assert not score_semantic_projection(resolved, uncertain, "definition_reference_linking")
    assert not score_semantic_projection(null, uncertain, "definition_reference_linking")


def test_semantic_to_code_ignores_formatting_comments_and_spans_but_not_meaning():
    formatted = "# comment\ndef add( a , b ):\n\treturn a+b  # same meaning\n"
    changed = "def add(a, b):\n    return a - b\n"

    assert semantic_projection_equivalent(
        PYTHON_SOURCE, formatted, language="python", expected_path="a.py", observed_path="b.py"
    )
    assert not semantic_projection_equivalent(
        PYTHON_SOURCE, changed, language="python", expected_path="a.py", observed_path="b.py"
    )
    assert not semantic_projection_equivalent(
        PYTHON_SOURCE, "def broken(:\n", language="python"
    )


def test_one_failed_language_runtime_or_boundary_gate_blocks_the_adapter():
    baseline = [_language("python"), _language("javascript")]
    weak_scores = {name: 1.0 for name in STRUCTURAL_CATEGORIES}
    weak_scores["scope_pairing"] = 0.79
    adapter = [
        _language("python"),
        _language(
            "javascript",
            category_scores=weak_scores,
            knowledge=0.89,
            uyr=0.84,
            false_certainty=0.06,
            runtime_failures=("generation failed",),
        ),
    ]

    decision = evaluate_multilingual_promotion(baseline, adapter)

    assert not decision.promoted
    assert set(decision.failure_codes) >= {
        "javascript:scope_pairing_below_threshold",
        "javascript:knowledge_state_accuracy_below_threshold",
        "javascript:uyr_resolution_accuracy_below_threshold",
        "javascript:false_certainty_above_threshold",
        "javascript:runtime_failure",
    }


def test_language_macro_and_control_regression_are_unweighted():
    baseline = [
        _language("python", case_count=10_000),
        _language("javascript", case_count=1),
    ]
    js_scores = {name: 0.9 for name in STRUCTURAL_CATEGORIES}
    adapter = [
        _language("python", raw=0.97, instruction=0.97, case_count=10_000),
        _language(
            "javascript", category_scores=js_scores, raw=0.97, instruction=0.97, case_count=1
        ),
    ]

    decision = evaluate_multilingual_promotion(baseline, adapter)

    assert decision.structural_macro_average == pytest.approx(0.95)
    assert "raw_code_regression" in decision.failure_codes
    assert "ordinary_instruction_regression" in decision.failure_codes

    exact_limit = evaluate_multilingual_promotion(
        baseline,
        [
            _language("python", raw=0.98, instruction=0.98),
            _language("javascript", raw=0.98, instruction=0.98),
        ],
    )
    assert exact_limit.promoted


def test_frozen_multilingual_configuration_has_exact_math():
    config = load_multilingual_eval_config("configs/bcnlt_multilingual_eval_v1.json")

    assert config["schema_version"] == "bcnlt-multilingual-eval-v1"
    assert tuple(config["categories"]) == STRUCTURAL_CATEGORIES
    assert len(config["languages"]) == 14
    assert sum(config["category_weights"].values()) == pytest.approx(1.0)
    assert config["thresholds"] == {
        "minimum_category_accuracy": 0.8,
        "minimum_language_macro": 0.9,
        "minimum_knowledge_state_accuracy": 0.9,
        "minimum_uyr_resolution_accuracy": 0.85,
        "maximum_false_certainty_rate": 0.05,
        "maximum_control_regression": 0.02,
    }
    assert Path("configs/bcnlt_multilingual_eval_v1.json").is_file()
