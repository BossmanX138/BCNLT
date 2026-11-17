from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

from bcnlt.corpus.curriculum import CurriculumRecord

from .manifest import AdaptiveVocabularyManifest


@dataclass(frozen=True, slots=True)
class PairedTrainingRecord:
    record: CurriculumRecord
    v1_input_ids: tuple[int, ...]
    adaptive_input_ids: tuple[int, ...]
    target_bytes: bytes


@dataclass(frozen=True, slots=True)
class PairedTrainingSelection:
    records: tuple[PairedTrainingRecord, ...]
    retained_macro_ids: tuple[int, ...]
    removed_macro_ids: tuple[int, ...]
    inventory_reevaluation_required: bool
    language_counts: tuple[tuple[str, int], ...]
    family_counts: tuple[tuple[str, int], ...]
    macro_counts: tuple[tuple[int, int], ...]
    v1_record_ids: tuple[str, ...]
    adaptive_record_ids: tuple[str, ...]
    target_sha256: str
    seed: int
    maximum_examples: int
    minimum_macro_occurrences: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "bcnlt-paired-training-selection-v1",
            "record_ids": list(self.v1_record_ids),
            "v1_record_ids": list(self.v1_record_ids),
            "adaptive_record_ids": list(self.adaptive_record_ids),
            "retained_macro_ids": list(self.retained_macro_ids),
            "removed_macro_ids": list(self.removed_macro_ids),
            "inventory_reevaluation_required": self.inventory_reevaluation_required,
            "language_counts": dict(self.language_counts),
            "family_counts": dict(self.family_counts),
            "macro_counts": {str(key): value for key, value in self.macro_counts},
            "target_sha256": self.target_sha256,
            "target_byte_lengths": [len(item.target_bytes) for item in self.records],
            "v1_input_sha256": _ids_digest(
                [item.v1_input_ids for item in self.records]
            ),
            "adaptive_input_sha256": _ids_digest(
                [item.adaptive_input_ids for item in self.records]
            ),
            "seed": self.seed,
            "maximum_examples": self.maximum_examples,
            "minimum_macro_occurrences": self.minimum_macro_occurrences,
        }


def _ids_digest(rows: Sequence[tuple[int, ...]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(len(row).to_bytes(8, "big"))
        for token_id in row:
            digest.update(int(token_id).to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def _rank(record: PairedTrainingRecord, seed: int) -> str:
    language = dict(record.record.provenance).get("canonical_language", "unknown")
    material = f"{seed}\0{language}\0{record.record.family}\0{record.record.record_id}"
    return hashlib.sha256(material.encode()).hexdigest()


def mutually_packable_records(
    records: Sequence[PairedTrainingRecord], *, maximum_sequence_tokens: int = 8192
) -> tuple[tuple[PairedTrainingRecord, ...], tuple[str, ...]]:
    if maximum_sequence_tokens <= 0:
        raise ValueError("maximum sequence tokens must be positive")
    kept = []
    rejected = []
    for item in records:
        if max(len(item.v1_input_ids), len(item.adaptive_input_ids)) > maximum_sequence_tokens:
            rejected.append(item.record.record_id)
        else:
            kept.append(item)
    return tuple(kept), tuple(rejected)


def select_paired_training_records(
    records: Sequence[PairedTrainingRecord],
    manifest: AdaptiveVocabularyManifest,
    *,
    minimum_macro_occurrences: int = 8,
    maximum_examples: int = 10_000,
    seed: int = 1729,
) -> PairedTrainingSelection:
    if minimum_macro_occurrences <= 0 or maximum_examples <= 0:
        raise ValueError("paired selection limits must be positive")
    macro_ids = {macro.id for macro in manifest.macros}
    training = tuple(item for item in records if item.record.split == "train")
    available = Counter(
        token_id
        for item in training
        for token_id in item.adaptive_input_ids
        if token_id in macro_ids
    )
    retained = {token_id for token_id in macro_ids if available[token_id] >= minimum_macro_occurrences}
    eligible = [
        item
        for item in training
        if any(token_id in retained for token_id in item.adaptive_input_ids)
    ]
    groups: dict[tuple[str, str], deque[PairedTrainingRecord]] = {}
    grouped: dict[tuple[str, str], list[PairedTrainingRecord]] = defaultdict(list)
    for item in eligible:
        language = str(dict(item.record.provenance).get("canonical_language", "unknown"))
        grouped[(language, item.record.family)].append(item)
    for key, values in grouped.items():
        groups[key] = deque(sorted(values, key=lambda item: (_rank(item, seed), item.record.record_id)))
    selected: list[PairedTrainingRecord] = []
    while groups and len(selected) < maximum_examples:
        for key in sorted(tuple(groups)):
            selected.append(groups[key].popleft())
            if not groups[key]:
                del groups[key]
            if len(selected) >= maximum_examples:
                break
    selected_macro_counts = Counter(
        token_id
        for item in selected
        for token_id in item.adaptive_input_ids
        if token_id in retained
    )
    retained = {
        token_id
        for token_id in retained
        if selected_macro_counts[token_id] >= minimum_macro_occurrences
    }
    selected = [
        item
        for item in selected
        if any(token_id in retained for token_id in item.adaptive_input_ids)
    ]
    language_counts = Counter(
        str(dict(item.record.provenance).get("canonical_language", "unknown"))
        for item in selected
    )
    family_counts = Counter(item.record.family for item in selected)
    target_digest = hashlib.sha256()
    for item in selected:
        target_digest.update(len(item.target_bytes).to_bytes(8, "big"))
        target_digest.update(item.target_bytes)
    ids = tuple(item.record.record_id for item in selected)
    removed = tuple(sorted(macro_ids.difference(retained)))
    return PairedTrainingSelection(
        records=tuple(selected),
        retained_macro_ids=tuple(sorted(retained)),
        removed_macro_ids=removed,
        inventory_reevaluation_required=bool(removed),
        language_counts=tuple(sorted(language_counts.items())),
        family_counts=tuple(sorted(family_counts.items())),
        macro_counts=tuple(sorted((key, selected_macro_counts[key]) for key in retained)),
        v1_record_ids=ids,
        adaptive_record_ids=ids,
        target_sha256=target_digest.hexdigest(),
        seed=seed,
        maximum_examples=maximum_examples,
        minimum_macro_occurrences=minimum_macro_occurrences,
    )
