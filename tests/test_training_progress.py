from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def _feature():
    if importlib.util.find_spec("bosscrafts_model.training_progress") is None:
        pytest.fail("typed training-progress reader is not implemented")
    return importlib.import_module("bosscrafts_model.training_progress")


def _payload() -> dict[str, object]:
    return {
        "schema_version": "bosscrafts-training-progress-v1",
        "checkpoint_written": True,
        "gradients_finite": True,
        "full_model": True,
        "task_database_identity_sha256": "a" * 64,
        "model_fingerprint": "b" * 64,
        "training_plan_sha256": "c" * 64,
        "planned_optimizer_steps": 100,
        "initialization": {"type": "random_seed_7"},
        "total_optimizer_steps": 40,
        "optimizer_steps_remaining": 60,
        "total_examples_seen": 320,
        "mean_loss": 1.25,
        "learning_rate": 0.0001,
        "training_seconds": 12.5,
        "training_complete": False,
    }


def test_loads_valid_progress_and_returns_narrowed_values(tmp_path: Path) -> None:
    feature = _feature()
    path = tmp_path / "training-progress.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    progress = feature.load_training_progress(
        path,
        expected_fields={
            "task_database_identity_sha256": "a" * 64,
            "model_fingerprint": "b" * 64,
            "training_plan_sha256": "c" * 64,
            "planned_optimizer_steps": 100,
            "initialization": {"type": "random_seed_7"},
        },
    )

    assert progress["total_optimizer_steps"] == 40
    assert progress["optimizer_steps_remaining"] == 60
    assert progress["training_complete"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("schema_version", "wrong-schema"),
        ("checkpoint_written", False),
        ("gradients_finite", False),
        ("full_model", False),
        ("planned_optimizer_steps", 0),
        ("planned_optimizer_steps", True),
        ("total_optimizer_steps", True),
        ("optimizer_steps_remaining", -1),
        ("total_examples_seen", -1),
        ("mean_loss", float("nan")),
        ("learning_rate", float("inf")),
        ("training_seconds", -0.1),
        ("training_complete", 0),
    ),
)
def test_rejects_invalid_progress_field_types_and_values(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    feature = _feature()
    payload = _payload()
    payload[key] = value
    path = tmp_path / "training-progress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training progress contract"):
        feature.load_training_progress(path, expected_fields={})


@pytest.mark.parametrize("key", tuple(_payload()))
def test_rejects_missing_required_progress_field(tmp_path: Path, key: str) -> None:
    feature = _feature()
    payload = _payload()
    del payload[key]
    path = tmp_path / "training-progress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training progress contract"):
        feature.load_training_progress(path, expected_fields={})


def test_rejects_inconsistent_optimizer_counters(tmp_path: Path) -> None:
    feature = _feature()
    payload = _payload()
    payload["optimizer_steps_remaining"] = 59
    path = tmp_path / "training-progress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training progress counters"):
        feature.load_training_progress(path, expected_fields={})


@pytest.mark.parametrize(
    ("total", "remaining", "complete"),
    ((40, 60, True), (100, 0, False)),
)
def test_rejects_training_completion_that_disagrees_with_remaining_steps(
    tmp_path: Path,
    total: int,
    remaining: int,
    complete: bool,
) -> None:
    feature = _feature()
    payload = _payload()
    payload["total_optimizer_steps"] = total
    payload["optimizer_steps_remaining"] = remaining
    payload["training_complete"] = complete
    path = tmp_path / "training-progress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training progress completion"):
        feature.load_training_progress(path, expected_fields={})


@pytest.mark.parametrize(
    "key",
    (
        "task_database_identity_sha256",
        "model_fingerprint",
        "training_plan_sha256",
    ),
)
def test_rejects_invalid_progress_lineage_digest(
    tmp_path: Path,
    key: str,
) -> None:
    feature = _feature()
    payload = _payload()
    payload[key] = "NOT-A-SHA256"
    path = tmp_path / "training-progress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training progress contract"):
        feature.load_training_progress(path, expected_fields={})


@pytest.mark.parametrize(
    "module_name",
    (
        "scripts.run_bosscrafts_v7_polish_training_until_complete",
        "scripts.run_bosscrafts_v8_consolidation_training_until_complete",
    ),
)
def test_historical_training_controllers_import_with_frozen_lineage(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    assert len(module.EXPECTED_DATABASE_IDENTITY) == 64
    assert len(module.EXPECTED_MODEL_FINGERPRINT) == 64
    assert module.PLANNED_OPTIMIZER_STEPS > 0
