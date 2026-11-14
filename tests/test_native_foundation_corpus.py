from __future__ import annotations

import json
from pathlib import Path

from bcnlt.native_foundation.corpus import (
    _ByteReservoir,
    _atomic_write_json,
    materialize_balanced_sample,
)


def test_atomic_progress_write_retries_windows_sharing_violation(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "progress.json"
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(path: Path, target: Path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(5, "sharing violation")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    _atomic_write_json(destination, {"status": "running"})

    assert attempts == 2
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "running"}


def test_byte_reservoir_selection_is_order_independent_for_unequal_payloads():
    keys = ["candidate-a", "candidate-b", "candidate-c"]
    import hashlib

    low, middle, high = sorted(
        keys,
        key=lambda key: int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big"),
    )
    payloads = {low: b"c" * 3, middle: b"a" * 8, high: b"b" * 5}

    first = _ByteReservoir(10)
    for key in (middle, high, low):
        first.consider(key=key, payload=payloads[key])
    second = _ByteReservoir(10)
    for key in (low, middle, high):
        second.consider(key=key, payload=payloads[key])

    assert b"".join(first.ordered_payloads()) == b"".join(second.ordered_payloads())


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_balanced_sample_is_deterministic_and_excludes_held_out_records(tmp_path: Path):
    records = [
        {
            "record_id": f"python-{index}",
            "language": "python",
            "split": "train",
            "prompt": f"python prompt {index}",
            "target": f"python target {index} " + ("x" * (index * 19 + 1)),
        }
        for index in range(5)
    ] + [
        {
            "record_id": f"javascript-{index}",
            "language": "javascript",
            "split": "train",
            "prompt": f"javascript prompt {index}",
            "target": f"javascript target {index} " + ("y" * (index * 23 + 1)),
        }
        for index in range(5)
    ] + [
        {
            "record_id": "held-out",
            "language": "python",
            "split": "test",
            "prompt": "DO NOT INDUCE",
            "target": "HELD OUT",
        }
    ]
    accepted_a = tmp_path / "accepted-a.jsonl"
    accepted_b = tmp_path / "accepted-b.jsonl"
    _write_jsonl(accepted_a, records)
    _write_jsonl(accepted_b, list(reversed(records)))
    english = tmp_path / "english.json"
    english.write_text(
        json.dumps(
            [
                {
                    "instruction": "approved instruction",
                    "output": "approved answer",
                    "review_status": "approved",
                },
                {
                    "instruction": "rejected instruction",
                    "output": "rejected answer",
                    "review_status": "rejected",
                },
            ]
        ),
        encoding="utf-8",
    )

    first = materialize_balanced_sample(
        accepted_corpus_path=accepted_a,
        english_dataset_path=english,
        output_dir=tmp_path / "first",
        languages=("python", "javascript"),
        bytes_per_language=160,
        english_bytes=160,
        progress_every=2,
    )
    second = materialize_balanced_sample(
        accepted_corpus_path=accepted_b,
        english_dataset_path=english,
        output_dir=tmp_path / "second",
        languages=("python", "javascript"),
        bytes_per_language=160,
        english_bytes=160,
        progress_every=2,
    )

    for name in ("python.txt", "javascript.txt", "english.txt"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    combined = b"".join((tmp_path / "first" / name).read_bytes() for name in (
        "python.txt", "javascript.txt", "english.txt"
    ))
    assert b"DO NOT INDUCE" not in combined
    assert b"rejected instruction" not in combined
    assert b"approved instruction" in combined
    assert first["status"] == "complete"
    assert first["selected_records_by_domain"]["python"] > 0
    assert first["selected_records_by_domain"]["javascript"] > 0
    assert first["selected_records_by_domain"]["english"] == 1
    assert first["sample_sha256"] == second["sample_sha256"]

