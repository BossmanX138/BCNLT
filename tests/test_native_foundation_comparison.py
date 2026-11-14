from __future__ import annotations

import json
from pathlib import Path

from bcnlt.native_foundation.comparison import (
    compare_tokenizers,
    select_heldout_records,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_sample(path: Path, bodies: list[str]) -> None:
    path.mkdir(parents=True)
    (path / "python.txt").write_text(
        "".join(
            f"<|bcnlt:record:begin|>\n{body}\n<|bcnlt:record:end|>\n"
            for body in bodies
        ),
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "languages": ["python"],
                "sample_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )


def test_heldout_selection_is_order_independent_and_blocks_leakage(tmp_path: Path):
    records = [
        {
            "record_id": "train",
            "language": "python",
            "split": "train",
            "prompt": "train prompt",
            "target": "train target",
            "source_document_ids": ["train-doc"],
            "template_id": "task-v1",
        },
        {
            "record_id": "shared-doc",
            "language": "python",
            "split": "test",
            "prompt": "held prompt",
            "target": "held target",
            "source_document_ids": ["train-doc"],
            "template_id": "task-v1",
        },
        {
            "record_id": "exact-leak",
            "language": "python",
            "split": "validation",
            "prompt": "leaked prompt",
            "target": "leaked target 123",
            "source_document_ids": ["held-doc-1"],
            "template_id": "task-v1",
        },
        {
            "record_id": "near-leak",
            "language": "python",
            "split": "test",
            "prompt": "leaked   prompt",
            "target": "leaked target 999",
            "source_document_ids": ["held-doc-2"],
            "template_id": "task-v1",
        },
        {
            "record_id": "clean-a",
            "language": "python",
            "split": "test",
            "prompt": "unique alpha",
            "target": "answer alpha",
            "source_document_ids": ["held-doc-a"],
            "template_id": "task-v2",
        },
        {
            "record_id": "clean-b",
            "language": "python",
            "split": "validation",
            "prompt": "unique beta",
            "target": "answer beta",
            "source_document_ids": ["held-doc-b"],
            "template_id": "task-v2",
        },
    ]
    first_source = tmp_path / "first.jsonl"
    second_source = tmp_path / "second.jsonl"
    _write_jsonl(first_source, records)
    _write_jsonl(second_source, list(reversed(records)))
    sample = tmp_path / "sample"
    _write_sample(sample, ["leaked prompt\nleaked target 123"])

    first, first_manifest = select_heldout_records(
        accepted_corpus_path=first_source,
        induction_sample_dir=sample,
        languages=("python",),
        records_per_language=2,
        seed=77,
    )
    second, second_manifest = select_heldout_records(
        accepted_corpus_path=second_source,
        induction_sample_dir=sample,
        languages=("python",),
        records_per_language=2,
        seed=77,
    )

    assert [record["record_id"] for record in first] == ["clean-a", "clean-b"]
    assert [record["record_id"] for record in second] == ["clean-a", "clean-b"]
    assert first_manifest["selection_sha256"] == second_manifest["selection_sha256"]
    assert first_manifest["rejections"]["train_source_document_overlap"] == 1
    assert first_manifest["rejections"]["exact_induction_duplicate"] == 1
    assert first_manifest["rejections"]["near_induction_duplicate"] == 1
    assert first_manifest["english_status"] == "missing_true_english_holdout"


class _Encoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _LiteralTokenizer:
    def __init__(self, pieces: dict[str, int], *, vocab_size: int):
        self.pieces = pieces
        self.reverse = {value: key for key, value in pieces.items()}
        self.vocab_size = vocab_size

    def encode(self, text: str):
        ids = []
        cursor = 0
        ordered = sorted(self.pieces, key=len, reverse=True)
        while cursor < len(text):
            for piece in ordered:
                if text.startswith(piece, cursor):
                    ids.append(self.pieces[piece])
                    cursor += len(piece)
                    break
            else:
                ids.append(self.pieces["<|unk|>"])
                cursor += 1
        return _Encoding(ids)

    def decode(self, ids: list[int], *, skip_special_tokens: bool):
        return "".join(self.reverse.get(token_id, "") for token_id in ids)

    def token_to_id(self, token: str):
        return self.pieces.get(token)

    def get_vocab_size(self):
        return self.vocab_size


def test_comparison_reports_paired_metrics_and_fixed_seed_uncertainty(tmp_path: Path):
    structural = "<|bcnlt:stream:begin|>"
    primary = _LiteralTokenizer(
        {"<|unk|>": 0, structural: 1, "ab": 2, "a": 3, "b": 4, "\n": 5},
        vocab_size=65_536,
    )
    control = _LiteralTokenizer(
        {"<|unk|>": 0, structural: 1, "a": 2, "b": 3, "\n": 4},
        vocab_size=32_768,
    )
    records = [
        {
            "record_id": "one",
            "language": "python",
            "split": "test",
            "prompt": "ab",
            "target": "ab",
        },
        {
            "record_id": "two",
            "language": "python",
            "split": "validation",
            "prompt": "a",
            "target": "b",
        },
    ]

    first = compare_tokenizers(
        records=records,
        primary_tokenizer=primary,
        control_tokenizer=control,
        structural_tokens=(structural,),
        seed=19,
        bootstrap_samples=500,
        model_width=2048,
    )
    second = compare_tokenizers(
        records=records,
        primary_tokenizer=primary,
        control_tokenizer=control,
        structural_tokens=(structural,),
        seed=19,
        bootstrap_samples=500,
        model_width=2048,
    )

    assert first["overall"]["records"] == 2
    assert first["overall"]["primary_tokens"] == 6
    assert first["overall"]["control_tokens"] == 8
    assert first["overall"]["paired_token_savings"] == 2
    assert first["overall"]["primary_unknown_emissions"] == 0
    assert first["overall"]["control_unknown_emissions"] == 0
    assert first["overall"]["primary_roundtrip_failures"] == 0
    assert first["overall"]["control_roundtrip_failures"] == 0
    assert first["structural_atomicity"]["primary_passed"] is True
    assert first["structural_atomicity"]["control_passed"] is True
    assert first["paired_uncertainty"] == second["paired_uncertainty"]
    assert first["resource_implications"]["extra_vocabulary_rows"] == 32_768
    assert first["resource_implications"]["extra_tied_embedding_parameters"] == 67_108_864
    assert first["promotion_decision"] == "not_evaluated"
