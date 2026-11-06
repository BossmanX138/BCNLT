from __future__ import annotations

from pathlib import Path

import pytest

from bcnlt.evaluation_cache import (
    BaselineCacheKey,
    load_baseline_cache,
    selection_sha256,
    write_baseline_cache,
)


def _key(*, selection_sha256: str = "a" * 64) -> BaselineCacheKey:
    return BaselineCacheKey(
        base_model_sha256="b" * 64,
        tokenizer_manifest_sha256="c" * 64,
        packed_dataset_manifest_sha256="d" * 64,
        suite_sha256="e" * 64,
        selection_sha256=selection_sha256,
        seed=1729,
    )


def test_baseline_cache_round_trips_only_for_exact_provenance(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    baseline = {"identity": "base", "heldout_loss": 3.25, "results": []}

    write_baseline_cache(path, key=_key(), baseline=baseline)

    assert load_baseline_cache(path, expected_key=_key()) == baseline
    assert load_baseline_cache(
        path, expected_key=_key(selection_sha256="f" * 64)
    ) is None


def test_baseline_cache_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline_cache(path, key=_key(), baseline={"identity": "base"})

    with pytest.raises(FileExistsError):
        write_baseline_cache(path, key=_key(), baseline={"identity": "other"})


def test_selection_digest_covers_order_and_record_metadata() -> None:
    rows = [
        {"record_id": "one", "language": "python", "family": "mixed_instruction"},
        {"record_id": "two", "language": "rust", "family": "semantic_to_code"},
    ]

    assert selection_sha256(rows, (0, 1)) == selection_sha256(rows, (0, 1))
    assert selection_sha256(rows, (0, 1)) != selection_sha256(rows, (1, 0))
