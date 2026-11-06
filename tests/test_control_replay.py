from __future__ import annotations

from collections import Counter

from scripts.build_bcnlt_control_replay import build_replay_records


def test_control_replay_is_balanced_deterministic_and_does_not_copy_frozen_prompts():
    first = build_replay_records(256, seed=1729)
    second = build_replay_records(256, seed=1729)
    categories = Counter(dict(item.provenance)["replay_category"] for item in first)

    assert first == second
    assert len(first) == 256
    assert max(categories.values()) - min(categories.values()) <= 1
    assert all(item.prompt.startswith(f"Language: {item.language}\n") for item in first)
    assert all(item.split == "train" for item in first)
    assert all(item.target for item in first)
    assert "Reply with the single word ready." not in {item.prompt.split("\n", 1)[1] for item in first}
    assert "Reconstruct parseable Python for a function named add with parameters a and b whose RETURN record contains the operator a + b." not in {
        item.prompt.split("\n", 1)[1] for item in first
    }


def test_control_replay_covers_each_failure_mode_with_correct_target_stream():
    records = build_replay_records(100, seed=1729)
    by_category = {}
    for item in records:
        by_category.setdefault(dict(item.provenance)["replay_category"], item)

    assert set(by_category) == {
        "ordinary_instruction",
        "raw_code",
        "semantic_to_code",
        "structural",
        "unicode_roundtrip",
    }
    assert by_category["ordinary_instruction"].target_mode == "text"
    assert by_category["structural"].target_mode == "text"
    assert by_category["raw_code"].target_mode == "raw_code"
    assert by_category["semantic_to_code"].target_mode == "raw_code"
    assert by_category["unicode_roundtrip"].target_mode == "raw_code"
