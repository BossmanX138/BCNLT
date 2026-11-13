from __future__ import annotations

from pathlib import Path

import pytest

from bosscrafts_model.audited_balanced_curriculum import (
    _Candidate,
    _PromptClaimIndex,
    _push_candidate,
)


def _candidate(
    number: int,
    *,
    prompt: str = "a",
    target: str = "b",
    split: str = "train",
    target_tokens: int = 8,
) -> _Candidate:
    return _Candidate(
        example_id=f"{number:064x}",
        document_id=f"document-{number}",
        language="python",
        family="semantic_to_code",
        record_index=number,
        split=split,
        input_ids_zlib=b"payload",
        token_count=16,
        loss_start_index=8,
        semantic_spec_sha256="c" * 64,
        parser_runtime_identity_sha256="d" * 64,
        source_sha256="e" * 64,
        prompt_key=("semantic_to_code", prompt * 64, 8),
        target_key=(target * 64, target_tokens),
        target_tokens=target_tokens,
    )


def test_prompt_claim_index_detects_global_conflicts_and_cross_split_leakage(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "claims.sqlite"
    conflict_key = _candidate(1, prompt="a").prompt_key
    cross_split_key = _candidate(3, prompt="b").prompt_key
    both_key = _candidate(5, prompt="c").prompt_key

    with _PromptClaimIndex(scratch) as claims:
        claims.add(_candidate(1, prompt="a", target="a"))
        claims.add(_candidate(2, prompt="a", target="b"))
        claims.add(_candidate(3, prompt="b", target="c", split="train"))
        claims.add(_candidate(4, prompt="b", target="c", split="test"))
        claims.add(_candidate(5, prompt="c", target="d", split="train"))
        claims.add(_candidate(6, prompt="c", target="e", split="test"))
        claims.add(_candidate(7, prompt="d", target="f"))
        claims.add(_candidate(8, prompt="d", target="f"))
        conflicting, cross_split = claims.exclusions()
        assert scratch.is_file()

    assert conflicting == {conflict_key, both_key}
    assert cross_split == {cross_split_key}
    assert not scratch.exists()
    assert not scratch.with_name(f"{scratch.name}-journal").exists()


def test_prompt_claim_index_removes_scratch_after_failure(tmp_path: Path) -> None:
    scratch = tmp_path / "claims.sqlite"
    with pytest.raises(RuntimeError, match="deliberate"):
        with _PromptClaimIndex(scratch) as claims:
            claims.add(_candidate(1))
            raise RuntimeError("deliberate")
    assert not scratch.exists()


def test_bounded_candidate_heap_keeps_lowest_content_addresses() -> None:
    heap: list[tuple[int, str, _Candidate]] = []
    for number in (5, 1, 4, 2, 3):
        _push_candidate(heap, _candidate(number), 3)
    assert sorted(item.example_id for _, _, item in heap) == [
        f"{number:064x}" for number in (1, 2, 3)
    ]
