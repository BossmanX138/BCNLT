"""Bounded deterministic selection of review-only held-out source documents."""
from __future__ import annotations

from collections.abc import Iterable

from .current_corpus import SourceCandidate
from .current_task_packing import CurrentCorpusTaskPacker


def select_heldout_candidates(
    candidates: Iterable[SourceCandidate],
    *,
    languages: set[str] | frozenset[str],
    per_language: int,
) -> tuple[SourceCandidate, ...]:
    if not languages or any(type(language) is not str or not language for language in languages):
        raise ValueError("languages must be a non-empty set of names")
    if type(per_language) is not int or per_language < 1:
        raise ValueError("per_language must be a positive integer")
    selected: list[SourceCandidate] = []
    counts = {language: 0 for language in languages}
    for candidate in candidates:
        if type(candidate) is not SourceCandidate:
            raise TypeError("candidates must be SourceCandidate values")
        if candidate.language not in counts or counts[candidate.language] >= per_language:
            continue
        if CurrentCorpusTaskPacker._split(candidate.document_id) not in {"validation", "test"}:
            continue
        selected.append(candidate)
        counts[candidate.language] += 1
        if all(count == per_language for count in counts.values()):
            break
    missing = {language: per_language - count for language, count in counts.items() if count < per_language}
    if missing:
        raise ValueError(f"held-out candidate quota could not be filled: {missing}")
    return tuple(sorted(selected, key=lambda candidate: (candidate.language, candidate.document_id)))
