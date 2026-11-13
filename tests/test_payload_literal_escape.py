from __future__ import annotations

import pytest

from bosscrafts_model.semantic import NativeSemanticRecordCodec, SemanticRecord
from bosscrafts_model.tokenizer import NativeTokenizerBinding


STANDARD_SPECIALS = ("<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>")


@pytest.fixture(scope="module")
def binding() -> NativeTokenizerBinding:
    return NativeTokenizerBinding.load()


def test_all_reserved_literals_are_lossless_payload_text(
    binding: NativeTokenizerBinding,
) -> None:
    literals = tuple(
        sorted((*STANDARD_SPECIALS, *binding.structural_token_ids), key=len, reverse=True)
    )
    cases = tuple(
        text
        for literal in literals
        for text in (
            literal,
            f"before {literal} after",
            literal + literal,
            f"\n{literal}\n",
        )
    )

    scalar = [binding.encode_payload(text) for text in cases]
    assert binding.encode_payload_batch(cases) == scalar
    for text, ids in zip(cases, scalar, strict=True):
        assert binding.decode(ids) == text
        assert all(token_id > 76 for token_id in ids)


def test_reserved_literal_is_data_inside_a_semantic_surface(
    binding: NativeTokenizerBinding,
) -> None:
    codec = NativeSemanticRecordCodec(binding)
    record = SemanticRecord(
        event="CALL",
        surface="print '<|bcnlt:record:end|>' literally",
        confidence_band="high",
    )
    ids = codec.encode(record)

    assert codec.decode(ids) == record
    structural = binding.structural_token_ids
    allowed = {
        structural["<|bcnlt:record:begin|>"],
        structural["<|bcnlt:mode:semantic_code|>"],
        structural["<|bcnlt:record:end|>"],
    }
    assert all(token_id > 76 or token_id in allowed for token_id in ids)
    assert structural["<|bcnlt:record:end|>"] not in ids[1:-1]


def test_unknown_control_like_text_remains_ordinary_payload(
    binding: NativeTokenizerBinding,
) -> None:
    text = "<|bcnlt:not-a-real-control|>"
    ids = binding.encode_payload(text)
    assert binding.decode(ids) == text
    assert all(token_id > 76 for token_id in ids)
