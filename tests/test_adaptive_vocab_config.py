from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcnlt.adaptive_vocab.config import AdaptiveVocabularyConfig
from bcnlt.languages import CANONICAL_LANGUAGE_IDS


CONFIG_PATH = Path("configs/bcnlt_adaptive_vocab_v5.json")


def test_production_adaptive_configuration_is_exact_and_canonical():
    config = AdaptiveVocabularyConfig.load(CONFIG_PATH)

    assert config.schema_version == "bcnlt-adaptive-vocab-v5"
    assert config.canonical_languages == CANONICAL_LANGUAGE_IDS
    assert config.minimum_documents_per_language == 100
    assert config.minimum_repositories_per_language == 3
    assert config.minimum_payload_tokens_per_language == 100_000
    assert (config.candidate_min_tokens, config.candidate_max_tokens) == (2, 16)
    assert config.maximum_occurrences_per_document == 8
    assert (
        config.minimum_supporting_languages,
        config.minimum_supporting_repositories,
        config.minimum_supporting_documents,
    ) == (3, 3, 5)
    assert config.maximum_repository_fraction == 0.50
    assert config.maximum_document_fraction == 0.10
    assert config.minimum_candidate_score == 25
    assert config.vocabulary_budgets == (256, 512, 1024)
    assert config.minimum_heldout_compression == 0.10
    assert config.maximum_p95_latency_slowdown == 0.10
    assert config.minimum_macro_training_occurrences == 8
    assert config.maximum_training_examples == 10_000
    assert config.maximum_optimizer_steps == 512
    assert config.validation_interval_steps == 64
    assert {pattern.name for pattern in config.secret_patterns} == {
        "private_key",
        "aws_access_key",
        "github_token",
        "openai_style_key",
        "jwt",
        "uri_userinfo",
        "hex_24",
        "base64_24",
    }
    assert CONFIG_PATH.read_bytes() == config.canonical_bytes()


def _mutated_config(tmp_path: Path, mutate) -> Path:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_configuration_rejects_malformed_secret_regex(tmp_path: Path):
    path = _mutated_config(
        tmp_path,
        lambda payload: payload["secret_patterns"][0].__setitem__("expression", "["),
    )

    with pytest.raises(ValueError, match="secret pattern"):
        AdaptiveVocabularyConfig.load(path)


def test_configuration_rejects_duplicate_vocabulary_budgets(tmp_path: Path):
    path = _mutated_config(
        tmp_path,
        lambda payload: payload.__setitem__("vocabulary_budgets", [256, 256, 1024]),
    )

    with pytest.raises(ValueError, match="vocabulary_budgets"):
        AdaptiveVocabularyConfig.load(path)


def test_configuration_rejects_noncanonical_language(tmp_path: Path):
    path = _mutated_config(
        tmp_path,
        lambda payload: payload.__setitem__(
            "canonical_languages", [*CANONICAL_LANGUAGE_IDS[:-1], "shell"]
        ),
    )

    with pytest.raises(ValueError, match="canonical_languages"):
        AdaptiveVocabularyConfig.load(path)


@pytest.mark.parametrize(
    ("field", "relaxed_value"),
    [
        ("minimum_documents_per_language", 99),
        ("minimum_payload_tokens_per_language", 99_999),
        ("minimum_candidate_score", 24),
        ("minimum_heldout_compression", 0.09),
        ("maximum_p95_latency_slowdown", 0.11),
        ("minimum_macro_training_occurrences", 7),
        ("maximum_training_examples", 10_001),
        ("maximum_optimizer_steps", 513),
    ],
)
def test_configuration_rejects_relaxed_production_thresholds(
    tmp_path: Path, field: str, relaxed_value: object
):
    path = _mutated_config(
        tmp_path,
        lambda payload: payload.__setitem__(field, relaxed_value),
    )

    with pytest.raises(ValueError, match=field):
        AdaptiveVocabularyConfig.load(path)
