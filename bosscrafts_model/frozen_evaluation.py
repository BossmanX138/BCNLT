"""Exact retrieval for a pre-training frozen evaluation inventory."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .model_acceptance import GroundedHeldoutSuite
from .resumable_task_packing import ResumableTaskDataset, TaskExample

MAX_HELDOUT_SUITE_MANIFEST_BYTES = 16 * 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate held-out suite manifest key: {key}")
        payload[key] = value
    return payload


def load_heldout_suite_manifest(path: str | Path) -> dict[str, object]:
    """Read a bounded UTF-8 manifest while rejecting duplicate JSON keys."""
    try:
        content = Path(path).resolve().read_bytes()
        if not content or len(content) > MAX_HELDOUT_SUITE_MANIFEST_BYTES:
            raise ValueError("held-out suite manifest size is invalid")
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("held-out suite manifest is unreadable") from error
    if type(payload) is not dict:
        raise ValueError("held-out suite manifest must be a JSON object")
    return payload


def iter_frozen_evaluation_examples(
    dataset: ResumableTaskDataset,
    suite: GroundedHeldoutSuite,
    *,
    split: str,
) -> Iterator[TaskExample]:
    """Yield only manifest-pinned examples and prove every row belongs to its cell."""
    if type(dataset) is not ResumableTaskDataset:
        raise TypeError("dataset must be a ResumableTaskDataset")
    if type(suite) is not GroundedHeldoutSuite:
        raise TypeError("suite must be a verified GroundedHeldoutSuite")
    if split not in {"validation", "test"}:
        raise ValueError("frozen evaluation split must be validation or test")
    split_cells = suite.evaluation_example_ids.get(split)
    if not split_cells:
        raise ValueError("frozen evaluation split inventory is empty")

    expected_cell_by_id: dict[str, str] = {}
    requested_ids: list[str] = []
    for cell in sorted(split_cells):
        language, separator, family = cell.partition("|")
        if not separator or not language or not family:
            raise ValueError("frozen evaluation cell name is invalid")
        for example_id in split_cells[cell]:
            if example_id in expected_cell_by_id:
                raise ValueError("frozen evaluation cells contain a duplicate example ID")
            expected_cell_by_id[example_id] = cell
            requested_ids.append(example_id)
    if not requested_ids:
        raise ValueError("frozen evaluation split inventory is empty")

    for row in dataset.iter_examples_by_id(requested_ids, split=split):
        observed_cell = f'{row["language"]}|{row["family"]}'
        if observed_cell != expected_cell_by_id[row["example_id"]]:
            raise ValueError("frozen evaluation example does not belong to its declared cell")
        yield row
