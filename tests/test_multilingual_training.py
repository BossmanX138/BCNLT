from __future__ import annotations

import math
import json
from collections import Counter
from pathlib import Path

import pytest

from bcnlt.languages import CANONICAL_LANGUAGE_IDS
from bcnlt.multilingual_training import (
    LORA_TARGET_MODULES,
    TRAINING_FAMILIES,
    LanguageBalancedSampler,
    MultilingualTrainingConfig,
    SmokeMetrics,
    audit_packed_multilingual_dataset,
    evaluate_smoke_gate,
    optimizer_step_means,
)


SMOKE_CONFIG = Path("configs/runeforge_mk2_bcnlt_multilingual_smoke_v1.json")
FULL_CONFIG = Path("configs/runeforge_mk2_bcnlt_multilingual_full_v1.json")


def _records() -> list[dict[str, str]]:
    return [
        {
            "record_id": f"{language}:{family}:{copy}",
            "language": language,
            "family": family,
        }
        for language in CANONICAL_LANGUAGE_IDS
        for family in TRAINING_FAMILIES
        for copy in range(2)
    ]


def _passing_metrics() -> SmokeMetrics:
    initial = {language: 1.0 for language in CANONICAL_LANGUAGE_IDS}
    final = {language: 0.9 for language in CANONICAL_LANGUAGE_IDS}
    return SmokeMetrics(
        optimizer_steps=128,
        global_step_losses=tuple(1.0 - index * 0.002 for index in range(128)),
        training_losses=tuple(1.0 - index * 0.0001 for index in range(1024)),
        initial_language_losses=initial,
        final_language_losses=final,
        validation_examples={language: 256 for language in CANONICAL_LANGUAGE_IDS},
    )


def test_authoritative_smoke_and_full_configs_are_pinned() -> None:
    smoke = MultilingualTrainingConfig.load(SMOKE_CONFIG)
    full = MultilingualTrainingConfig.load(FULL_CONFIG)

    assert smoke.languages == full.languages == CANONICAL_LANGUAGE_IDS
    assert smoke.families == full.families == TRAINING_FAMILIES
    assert smoke.target_modules == full.target_modules == LORA_TARGET_MODULES
    assert smoke.maximum_optimizer_steps == 128
    assert full.maximum_optimizer_steps is None
    assert full.validation_interval_optimizer_steps == 500
    assert full.checkpoint_interval_optimizer_steps == 500
    assert full.epochs == 1


def test_balanced_sampler_is_deterministic_and_near_equal_at_smoke_scale() -> None:
    rows = _records()
    sampler = LanguageBalancedSampler(rows, seed=1729)

    indices = sampler.sample_indices(128 * 8)
    repeated = LanguageBalancedSampler(rows, seed=1729).sample_indices(128 * 8)
    counts = sampler.balance_counts(indices)

    assert indices == repeated
    assert len(indices) == 1024
    assert max(counts["languages"].values()) - min(counts["languages"].values()) <= 1
    assert max(counts["families"].values()) - min(counts["families"].values()) <= 1
    cell_counts = Counter(
        (rows[index]["language"], rows[index]["family"]) for index in indices
    )
    assert len(cell_counts) == 14 * 6
    assert max(cell_counts.values()) - min(cell_counts.values()) <= 1


def test_balanced_sampler_refuses_a_missing_language_family_cell() -> None:
    rows = _records()
    rows = [
        row for row in rows
        if not (row["language"] == "bash" and row["family"] == "knowledge_boundary")
    ]

    with pytest.raises(ValueError, match="bash/knowledge_boundary"):
        LanguageBalancedSampler(rows)


def test_smoke_gate_requires_finite_improving_global_and_per_language_losses() -> None:
    config = MultilingualTrainingConfig.load(SMOKE_CONFIG)

    decision = evaluate_smoke_gate(_passing_metrics(), config)

    assert decision.promoted
    assert not decision.failure_codes
    assert decision.global_improvement is not None
    assert decision.global_improvement >= 0.02


def test_smoke_gate_reports_every_numerical_failure() -> None:
    config = MultilingualTrainingConfig.load(SMOKE_CONFIG)
    passing = _passing_metrics()
    initial = dict(passing.initial_language_losses)
    final = dict(passing.final_language_losses)
    final["rust"] = 1.26
    examples = dict(passing.validation_examples)
    examples["bash"] = 255
    metrics = SmokeMetrics(
        optimizer_steps=127,
        global_step_losses=tuple([1.0] * 127 + [math.inf]),
        training_losses=passing.training_losses,
        initial_language_losses=initial,
        final_language_losses=final,
        validation_examples=examples,
    )

    decision = evaluate_smoke_gate(metrics, config)

    assert not decision.promoted
    assert "optimizer_step_count" in decision.failure_codes
    assert "non_finite_loss" in decision.failure_codes
    assert "global_loss_improvement_below_threshold" in decision.failure_codes
    assert "language_loss_regression:rust" in decision.failure_codes
    assert "validation_examples:bash" in decision.failure_codes


def test_optimizer_step_means_aggregate_exact_accumulation_windows() -> None:
    assert optimizer_step_means(tuple(range(16)), 8) == (3.5, 11.5)


def test_packed_smoke_audit_requires_every_cell_and_256_validation_rows(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.records.jsonl"
    validation = tmp_path / "validation.records.jsonl"
    train.write_text(
        "".join(
            json.dumps({"language": language, "family": family}) + "\n"
            for language in CANONICAL_LANGUAGE_IDS
            for family in TRAINING_FAMILIES
        ),
        encoding="utf-8",
    )
    validation.write_text(
        "".join(
            json.dumps({"language": language, "family": TRAINING_FAMILIES[0]}) + "\n"
            for language in CANONICAL_LANGUAGE_IDS
            for _ in range(256)
        ),
        encoding="utf-8",
    )
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "shards": [
                    {"split": "train", "records_path": train.name, "token_count": 1000},
                    {
                        "split": "validation",
                        "records_path": validation.name,
                        "token_count": 1000,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_packed_multilingual_dataset(
        tmp_path, MultilingualTrainingConfig.load(SMOKE_CONFIG)
    )

    assert report["passed"] is True
    assert len(report["language_reports"]) == 14


def test_orchestration_uses_one_shared_adapter_and_defaults_full_run_to_validation() -> None:
    corpus = Path("scripts/build_bcnlt_multilingual_corpus.ps1").read_text()
    smoke = Path("scripts/run_bcnlt_multilingual_smoke.ps1").read_text()
    launcher = Path("scripts/launch_bcnlt_multilingual_full_run.ps1").read_text()

    assert "bcnlt_corpus_sources_v2.json" in corpus
    assert "sample_bcnlt_multilingual_sources" in corpus
    assert "bcnlt.multilingual_training" in corpus
    assert "runeforge_mk2_bcnlt_multilingual_smoke_v1.json" in smoke
    assert "--maximum-optimizer-steps 1" in smoke
    assert all(f"$Output-{language}" not in smoke for language in CANONICAL_LANGUAGE_IDS)
    assert "[switch]$StartTraining" in launcher
    assert "if ($StartTraining)" in launcher
    assert "--start-training" in launcher
