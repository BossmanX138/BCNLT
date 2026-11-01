from __future__ import annotations

from pathlib import Path

import pytest

from bcnlt.adapters.python import PythonAdapter
from bcnlt.canonical import canonical_record_bytes
from bcnlt.semantic_record import records_from_units
from bcnlt.tokenizer import BCNLTTokenizer
from bcnlt.tokenizer_manifest import BaseTokenizerIdentity, build_manifest


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("BCNLT payloads must never request base special tokens")
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(ids).decode("utf-8")

    def get_vocab(self) -> dict[str, int]:
        return {f"<byte:{value:02x}>": value for value in range(256)}


def _tokenizer() -> BCNLTTokenizer:
    base = ByteTokenizer()
    identity = BaseTokenizerIdentity.from_files(
        name="fixture/byte-tokenizer",
        revision="v1",
        files={"tokenizer.json": b"byte-tokenizer-v1"},
    )
    return BCNLTTokenizer(base, build_manifest(base, identity))


def _records(source: str = "caf\u00e9 = '<|bcnlt:event:call|>'\nprint(caf\u00e9)\n"):
    adapter = PythonAdapter()
    document = adapter.parse(source, path="unicode.py")
    return records_from_units(adapter.to_units(document), document.index)


@pytest.mark.parametrize("mode", ["text", "raw_code"])
def test_text_and_raw_code_modes_preserve_base_payload_ids(mode: str):
    tokenizer = _tokenizer()
    text = "hello, caf\u00e9\n"
    encoded = tokenizer.encode_text(text) if mode == "text" else tokenizer.encode_raw_code(text)

    assert encoded[2:-1] == tokenizer.base.encode(text, add_special_tokens=False)
    assert tokenizer.decode(encoded) == text


def test_semantic_records_round_trip_exactly_with_unicode_and_token_like_payloads():
    tokenizer = _tokenizer()
    records = _records()

    encoded = tokenizer.encode_records(records)
    decoded = tokenizer.decode_records(encoded)

    assert decoded == records
    assert tokenizer.decode(encoded).encode("utf-8") == b"".join(
        canonical_record_bytes(record) for record in records
    )


def test_multiple_records_keep_explicit_boundaries_and_empty_stream_is_valid():
    tokenizer = _tokenizer()
    records = _records("left = 1\nright = 2\n")
    token_ids = tokenizer.manifest.token_to_id

    encoded = tokenizer.encode_records(records)
    empty = tokenizer.encode_records(())

    assert encoded.count(token_ids["<|bcnlt:record:begin|>"]) == len(records)
    assert encoded.count(token_ids["<|bcnlt:record:end|>"]) == len(records)
    assert tokenizer.decode_records(empty) == ()
    assert tokenizer.decode(empty) == ""


def test_encode_source_matches_precomputed_record_encoding():
    tokenizer = _tokenizer()
    source = "def add(left, right):\n    return left + right\n"

    assert tokenizer.encode_source(source, "add.py") == tokenizer.encode_records(_records(source))


def test_encode_source_routes_non_python_through_the_multilingual_carousel():
    tokenizer = _tokenizer()
    source = "function add(left, right) { return left + right; }\n"

    encoded = tokenizer.encode_source(source, "add.js", language="javascript")
    records = tokenizer.decode_records(encoded)

    assert records
    assert any(record.event == "DEFINE" and record.surface == "add" for record in records)


def test_decoder_rejects_truncated_semantic_field():
    tokenizer = _tokenizer()
    encoded = tokenizer.encode_records(_records("answer = 42\n"))
    field_end = tokenizer.manifest.token_to_id["<|bcnlt:field:surface:end|>"]
    encoded.remove(field_end)

    with pytest.raises(ValueError, match="surface"):
        tokenizer.decode_records(encoded)


def test_decoder_rejects_unexpected_mode_marker():
    tokenizer = _tokenizer()
    encoded = tokenizer.encode_text("hello")
    encoded[1] = tokenizer.manifest.token_to_id["<|bcnlt:task:semantic_continuation|>"]

    with pytest.raises(ValueError, match="mode"):
        tokenizer.decode(encoded)


def test_decoder_rejects_added_id_absent_from_manifest():
    tokenizer = _tokenizer()
    encoded = tokenizer.encode_text("hello")
    encoded.insert(-1, tokenizer.manifest.vocab_size)

    with pytest.raises(ValueError, match="unknown token ID"):
        tokenizer.decode(encoded)


def test_decoder_rejects_duplicated_or_out_of_order_fields():
    tokenizer = _tokenizer()
    encoded = tokenizer.encode_records(_records("answer = 42\n"))
    token_ids = tokenizer.manifest.token_to_id
    scope_begin = token_ids["<|bcnlt:field:scope:begin|>"]
    encoded[encoded.index(scope_begin)] = token_ids["<|bcnlt:field:surface:begin|>"]

    with pytest.raises(ValueError, match="duplicated or out-of-order"):
        tokenizer.decode_records(encoded)


@pytest.mark.parametrize("bad_id", [-1, 10_000_000])
def test_decoder_rejects_ids_outside_declared_vocabulary(bad_id: int):
    tokenizer = _tokenizer()
    encoded = tokenizer.encode_text("hello")
    encoded.insert(-1, bad_id)

    with pytest.raises(ValueError, match="token ID"):
        tokenizer.decode(encoded)


def test_constructor_rejects_a_manifest_for_a_different_base_vocabulary():
    original = ByteTokenizer()
    identity = BaseTokenizerIdentity.from_files(
        name="fixture/byte-tokenizer",
        revision="v1",
        files={"tokenizer.json": b"byte-tokenizer-v1"},
    )
    manifest = build_manifest(original, identity)

    class ShiftedByteTokenizer(ByteTokenizer):
        def get_vocab(self) -> dict[str, int]:
            vocab = super().get_vocab()
            del vocab["<byte:ff>"]
            vocab["<different>"] = 255
            return vocab

    with pytest.raises(ValueError, match="base vocabulary"):
        BCNLTTokenizer(ShiftedByteTokenizer(), manifest)
