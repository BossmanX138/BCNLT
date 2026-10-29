from __future__ import annotations

from dataclasses import dataclass

from bcnlt.adaptive_vocab.manifest import AdaptiveVocabularyManifest, MacroToken
from bcnlt.adaptive_vocab.training import (
    PairedTrainingRecord,
    mutually_packable_records,
    select_paired_training_records,
)
from bcnlt.corpus.curriculum import CurriculumRecord


def _manifest() -> AdaptiveVocabularyManifest:
    return AdaptiveVocabularyManifest.create(
        v1_manifest_sha256="a" * 64,
        candidate_report_sha256="b" * 64,
        config_sha256="c" * 64,
        lineage_sha256="d" * 64,
        base_vocab_size=10,
        v1_vocab_size=20,
        macros=(
            MacroToken("<|bcnlt:macro:0000|>", 20, (1, 2), ("surface",)),
            MacroToken("<|bcnlt:macro:0001|>", 21, (3, 4), ("surface",)),
        ),
    )


def _record(index: int, macro_id: int, *, split: str = "train") -> PairedTrainingRecord:
    curriculum = CurriculumRecord(
        record_id=f"{index:064x}",
        family="semantic_continuation" if index % 2 else "code_to_semantic",
        template_id="pair-v1",
        source_document_ids=(f"doc-{index}",),
        prompt_mode="semantic_code",
        target_mode="raw_code",
        prompt="{}\n",
        target=f"target-{index}\n",
        provenance=(("canonical_language", "python" if index % 2 else "javascript"),),
        split=split,
    )
    return PairedTrainingRecord(
        record=curriculum,
        v1_input_ids=(1, 2, 3),
        adaptive_input_ids=(macro_id, 3),
        target_bytes=curriculum.target.encode("utf-8"),
    )


def test_selection_is_deterministic_paired_and_removes_unreachable_macros():
    records = tuple(_record(index, 20) for index in range(12)) + (
        _record(99, 21),
        _record(100, 20, split="validation"),
    )

    first = select_paired_training_records(
        records, _manifest(), minimum_macro_occurrences=8, maximum_examples=10, seed=1729
    )
    second = select_paired_training_records(
        records, _manifest(), minimum_macro_occurrences=8, maximum_examples=10, seed=1729
    )

    assert first == second
    assert len(first.records) == 10
    assert first.retained_macro_ids == (20,)
    assert first.removed_macro_ids == (21,)
    assert first.inventory_reevaluation_required is True
    assert all(item.record.split == "train" for item in first.records)
    assert first.v1_record_ids == first.adaptive_record_ids
    assert first.target_sha256 == second.target_sha256


def test_shared_packability_rejects_a_record_from_both_arms():
    records = (
        _record(1, 20),
        PairedTrainingRecord(_record(2, 20).record, tuple(range(9)), (20,), b"x"),
    )
    kept, rejected = mutually_packable_records(records, maximum_sequence_tokens=8)
    assert tuple(item.record.record_id for item in kept) == (records[0].record.record_id,)
    assert rejected == (records[1].record.record_id,)
