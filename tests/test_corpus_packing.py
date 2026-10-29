from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.corpus.packing import PackedBCNLTDataset, pack_training_shards


class Manifest:
    manifest_sha256 = "a" * 64
    vocab_size = 512


class TinyTokenizer:
    manifest = Manifest()

    def _encode(self, value: str) -> list[int]:
        return [1, *(10 + (byte % 200) for byte in value.encode()), 2]

    encode_text = _encode
    encode_raw_code = _encode

    def encode_records(self, records) -> list[int]:
        return self._encode("".join(record.event for record in records))


def _record(index: int, *, split: str = "train", target: str | None = None):
    prompt = f"prompt {index}"
    answer = target if target is not None else f"answer {index}"
    digest = hashlib.sha256(f"{index}:{split}:{prompt}:{answer}".encode()).hexdigest()
    return CurriculumRecord(
        record_id=digest,
        family="mixed_instruction",
        template_id="test-v1",
        source_document_ids=(f"doc-{index}",),
        prompt_mode="text",
        target_mode="raw_code",
        prompt=prompt,
        target=answer,
        provenance=(("content_sha256", hashlib.sha256(prompt.encode()).hexdigest()),),
        split=split,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_packing_is_byte_identical_and_offsets_are_exact(tmp_path: Path):
    records = [_record(1), _record(2), _record(3, split="validation")]
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest = pack_training_shards(records, TinyTokenizer(), first)
    pack_training_shards(records, TinyTokenizer(), second)

    assert _tree_bytes(first) == _tree_bytes(second)
    assert manifest.tokenizer_manifest_sha256 == "a" * 64
    dataset = PackedBCNLTDataset(first, "train", expected_tokenizer_digest="a" * 64)
    first_item = dataset[0]
    second_item = dataset[1]
    expected_prompt = TinyTokenizer().encode_text("prompt 1")
    expected_target = TinyTokenizer().encode_raw_code("answer 1")
    assert len(dataset) == 2
    assert first_item["input_ids"] == expected_prompt + expected_target
    assert first_item["labels"] == [-100] * len(expected_prompt) + expected_target
    assert first_item["attention_mask"] == [1] * len(first_item["input_ids"])
    assert second_item["input_ids"][0] == 1
    assert manifest.shards[0].start_offset == 0
    assert manifest.shards[0].token_count == sum(
        len(dataset[index]["input_ids"]) for index in range(len(dataset))
    )


def test_long_rows_are_rejected_without_truncation_and_shards_roll_over(tmp_path: Path):
    records = [_record(1), _record(2), _record(3, target="x" * 100)]
    tokenizer = TinyTokenizer()
    one_row_tokens = len(tokenizer.encode_text("prompt 1")) + len(
        tokenizer.encode_raw_code("answer 1")
    )

    manifest = pack_training_shards(
        records,
        tokenizer,
        tmp_path / "packed",
        maximum_sequence_tokens=50,
        shard_token_limit=one_row_tokens,
    )

    assert len(manifest.shards) == 2
    assert manifest.rejected_record_count == 1
    assert manifest.accepted_record_count == 2
    assert "sequence_too_long" in (tmp_path / "packed" / "rejections.jsonl").read_text()
    assert all(shard.record_count == 1 for shard in manifest.shards)


def test_token_ids_outside_the_manifest_range_are_rejected(tmp_path: Path):
    class BrokenTokenizer(TinyTokenizer):
        def encode_text(self, value: str) -> list[int]:
            return [self.manifest.vocab_size]

    with pytest.raises(ValueError, match="outside tokenizer vocabulary"):
        pack_training_shards([_record(1)], BrokenTokenizer(), tmp_path / "packed")


def test_loader_checks_tokenizer_digest_and_detects_corruption(tmp_path: Path):
    output = tmp_path / "packed"
    pack_training_shards([_record(1)], TinyTokenizer(), output)

    with pytest.raises(ValueError, match="tokenizer manifest digest"):
        PackedBCNLTDataset(output, "train", expected_tokenizer_digest="b" * 64)

    binary = next(output.glob("train-*.bin"))
    binary.write_bytes(binary.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        PackedBCNLTDataset(output, "train", expected_tokenizer_digest="a" * 64)


def test_packing_accepts_adaptive_manifest_digest_and_vocabulary_ceiling(tmp_path: Path):
    class AdaptiveManifest:
        manifest_sha256 = "c" * 64
        vocab_size = 1024

    class AdaptiveTokenizer(TinyTokenizer):
        manifest = AdaptiveManifest()

        def encode_text(self, value: str) -> list[int]:
            return [1, 900, 2]

    output = tmp_path / "adaptive-packed"
    manifest = pack_training_shards([_record(1)], AdaptiveTokenizer(), output)
    dataset = PackedBCNLTDataset(
        output, "train", expected_tokenizer_digest="c" * 64
    )

    assert manifest.tokenizer_manifest_sha256 == "c" * 64
    assert manifest.tokenizer_vocab_size == 1024
    assert 900 in dataset[0]["input_ids"]


def test_resume_requires_matching_input_range_and_output_hashes(tmp_path: Path):
    output = tmp_path / "packed"
    records = [_record(1), _record(2)]
    pack_training_shards(records, TinyTokenizer(), output)
    binary = next(output.glob("train-*.bin"))
    before = binary.stat().st_mtime_ns

    pack_training_shards(records, TinyTokenizer(), output, resume=True)
    assert binary.stat().st_mtime_ns == before

    with pytest.raises(ValueError, match="input range"):
        pack_training_shards(records + [_record(3)], TinyTokenizer(), output, resume=True)
