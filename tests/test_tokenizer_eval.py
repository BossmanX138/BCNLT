from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcnlt.tokenizer_eval import (
    EvaluationCase,
    EvaluationRun,
    GenerationSettings,
    decode_generated_stream,
    encode_evaluation_prompt,
    evaluate_model,
    evaluate_promotion,
    load_evaluation_suite,
)


SETTINGS = GenerationSettings(seed=17, temperature=0.0, max_new_tokens=128)
CASES = (
    EvaluationCase("struct-1", "structural", "symbol?", "pkg::value"),
    EvaluationCase(
        "semantic-code-1",
        "semantic_to_code",
        "reconstruct",
        "def add(a, b):\n    return a + b\n",
    ),
    EvaluationCase("raw-1", "raw_code", "return exact", "value = 1\n"),
    EvaluationCase("instruction-1", "ordinary_instruction", "say ready", "ready"),
    EvaluationCase("tokenizer-1", "tokenizer", "round trip", "hello λ"),
)


def _perfect(prompt: str, settings: GenerationSettings) -> str:
    assert settings == SETTINGS
    return {case.prompt: case.expected for case in CASES}[prompt]


def test_evaluator_scores_frozen_categories_deterministically():
    first = evaluate_model("base", CASES, SETTINGS, _perfect, heldout_loss=2.0)
    second = evaluate_model("base", CASES, SETTINGS, _perfect, heldout_loss=2.0)

    assert first == second
    assert first.metrics == {
        "ast_equivalence_accuracy": 1.0,
        "exact_match_accuracy": 1.0,
        "ordinary_instruction_accuracy": 1.0,
        "parseability_accuracy": 1.0,
        "raw_code_accuracy": 1.0,
        "semantic_to_code_accuracy": 1.0,
        "structural_accuracy": 1.0,
        "tokenizer_accuracy": 1.0,
    }
    assert not first.runtime_failures
    assert all(result.prompt == case.prompt for result, case in zip(first.results, CASES))


def test_evaluation_run_round_trips_through_cache_payload():
    report = evaluate_model("base", CASES, SETTINGS, _perfect, heldout_loss=2.0)

    assert EvaluationRun.from_dict(report.to_dict()) == report


def test_semantic_code_requires_parseable_ast_equivalence_and_failures_are_recorded():
    def generate(prompt: str, settings: GenerationSettings) -> str:
        if prompt == "reconstruct":
            return "def add(a, b):\n  return a - b\n"
        if prompt == "symbol?":
            raise RuntimeError("boom")
        return _perfect(prompt, settings)

    report = evaluate_model("adapter", CASES, SETTINGS, generate, heldout_loss=1.0)

    assert report.metrics["semantic_to_code_accuracy"] == 0.0
    assert report.metrics["ast_equivalence_accuracy"] == 0.0
    assert report.metrics["structural_accuracy"] == 0.0
    assert report.runtime_failures == ("struct-1: RuntimeError: boom",)


def test_promotion_requires_identical_prompts_settings_and_all_gates():
    baseline = evaluate_model("untrained", CASES, SETTINGS, _perfect, heldout_loss=2.0)
    adapter = evaluate_model("adapter", CASES, SETTINGS, _perfect, heldout_loss=1.5)

    decision = evaluate_promotion(baseline, adapter)
    assert decision.promoted
    assert not decision.failure_codes

    regressed = evaluate_model(
        "adapter",
        CASES,
        SETTINGS,
        lambda prompt, settings: "wrong" if prompt == "return exact" else _perfect(prompt, settings),
        heldout_loss=2.1,
    )
    rejected = evaluate_promotion(baseline, regressed)
    assert set(rejected.failure_codes) >= {
        "heldout_loss_not_improved",
        "raw_code_regression",
    }

    changed = evaluate_model(
        "adapter",
        CASES,
        GenerationSettings(seed=18, temperature=0.0, max_new_tokens=128),
        lambda prompt, settings: {case.prompt: case.expected for case in CASES}[prompt],
        heldout_loss=1.0,
    )
    with pytest.raises(ValueError, match="settings"):
        evaluate_promotion(baseline, changed)


@pytest.mark.parametrize(
    "category",
    ("tokenizer", "semantic_to_code", "raw_code", "ordinary_instruction"),
)
def test_promotion_rejects_any_failed_required_category(category: str):
    baseline = evaluate_model("untrained", CASES, SETTINGS, _perfect, heldout_loss=2.0)
    failed_prompt = next(case.prompt for case in CASES if case.category == category)
    adapter = evaluate_model(
        "adapter",
        CASES,
        SETTINGS,
        lambda prompt, settings: (
            "wrong" if prompt == failed_prompt else _perfect(prompt, settings)
        ),
        heldout_loss=1.0,
    )

    decision = evaluate_promotion(baseline, adapter)

    assert not decision.promoted
    assert f"{category}_accuracy_below_threshold" in decision.failure_codes


def test_frozen_v1_configuration_loads_exact_cases():
    path = Path("configs/bcnlt_tokenizer_eval_v1.json")
    suite = load_evaluation_suite(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert suite.schema_version == "bcnlt-tokenizer-eval-v1"
    assert suite.settings.temperature == 0.0
    assert suite.cases
    assert {case.category for case in suite.cases} == {
        "tokenizer",
        "structural",
        "semantic_to_code",
        "raw_code",
        "ordinary_instruction",
    }
    assert len({case.case_id for case in suite.cases}) == len(raw["cases"])


def test_generated_hybrid_stream_is_trimmed_at_its_structural_end_token():
    class Tokenizer:
        manifest = type(
            "Manifest",
            (),
            {"token_to_id": {"<|bcnlt:stream:end|>": 99}},
        )()

        def decode(self, ids):
            assert ids == [10, 11, 99]
            return "ready"

    assert decode_generated_stream(Tokenizer(), [10, 11, 99, 2, 2]) == "ready"
    with pytest.raises(ValueError, match="stream end"):
        decode_generated_stream(Tokenizer(), [10, 11, 2])


def test_evaluation_prompt_matches_packed_language_and_body_stream_framing():
    class Tokenizer:
        def __init__(self):
            self.calls = []

        def encode_text(self, value):
            self.calls.append(value)
            return [len(self.calls)]

    tokenizer = Tokenizer()

    assert encode_evaluation_prompt(tokenizer, "Reply ready.", language="python") == [1, 2]
    assert tokenizer.calls == ["Language: python\n", "Reply ready."]
