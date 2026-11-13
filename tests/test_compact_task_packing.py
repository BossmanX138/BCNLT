from __future__ import annotations

import hashlib
import json

import pytest

from bosscrafts_model.compact_semantic_wire import CompactSemanticWireV2
from bosscrafts_model.compact_task_packing import CompactCurrentCorpusTaskPacker
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload(surface: str, history: list[str], line: int) -> dict[str, object]:
    return {
        "ast_relation": "identifier.self",
        "confidence_band": "high",
        "defines": None,
        "event": "REFERENCE",
        "ir": {
            "children": [],
            "kind": "variable",
            "metadata": {},
            "name": surface,
            "value": None,
        },
        "metadata": {
            "knowledge_state": "known",
            "uncertainty_signals": [],
            "uyr_rounds": 0,
        },
        "recent_history": history,
        "scope": "module",
        "span": [line, 0, line, len(surface)],
        "surface": surface,
        "symbol_id": None,
        "uncertainty_band": "low",
    }


def test_compact_task_fragments_are_independently_decodable() -> None:
    binding = NativeTokenizerBinding.load()
    packer = CompactCurrentCorpusTaskPacker(binding, max_sequence=2048)
    codec = CompactSemanticWireV2(binding)
    lines = (
        _canonical(_payload("alpha", [], 1)),
        _canonical(_payload("beta", ["alpha"], 2)),
    )
    row = {
        "canonical_jsonl": "".join(f"{line}\n" for line in lines),
        "language": "python",
        "semantic_record_count": 2,
    }

    fragments = packer._semantic_record_fragments(row)

    assert len(fragments) == 2
    for line, fragment in zip(lines, fragments, strict=True):
        stream = [
            binding.structural_token_ids["<|bcnlt:stream:begin|>"],
            binding.structural_token_ids["<|bcnlt:mode:semantic_code|>"],
            *fragment,
            binding.structural_token_ids["<|bcnlt:stream:end|>"],
        ]
        assert codec.decode(stream) == f"{line}\n"


def test_compact_task_packer_builds_all_task_families() -> None:
    binding = NativeTokenizerBinding.load()
    packer = CompactCurrentCorpusTaskPacker(binding, max_sequence=2048)
    payloads = (_payload("alpha", [], 1), _payload("beta", ["alpha"], 2))
    canonical_jsonl = "".join(f"{_canonical(payload)}\n" for payload in payloads)
    source = "alpha\nbeta\n"
    row = {
        "document_id": "d" * 64,
        "language": "python",
        "canonical_jsonl": canonical_jsonl,
        "semantic_record_count": 2,
        "source": source,
        "semantic_spec_sha256": "a" * 64,
        "parser_runtime_identity_sha256": "b" * 64,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }

    examples, rejections = packer.pack_row(row)

    assert {example["family"] for example in examples} == {
        "code_to_semantic",
        "semantic_continuation",
        "semantic_to_code",
    }
    assert not rejections


def test_continuation_packing_does_not_copy_fragment_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoSliceFragments(list[list[int]]):
        def __getitem__(self, key: object) -> list[int] | list[list[int]]:
            if isinstance(key, slice):
                raise AssertionError("quadratic fragment-prefix slice was used")
            return super().__getitem__(key)  # type: ignore[index]

    binding = NativeTokenizerBinding.load()
    packer = CompactCurrentCorpusTaskPacker(binding, max_sequence=2048)
    payloads = (_payload("alpha", [], 1), _payload("beta", ["alpha"], 2))
    canonical_jsonl = "".join(f"{_canonical(payload)}\n" for payload in payloads)
    source = "alpha\nbeta\n"
    row = {
        "document_id": "d" * 64,
        "language": "python",
        "canonical_jsonl": canonical_jsonl,
        "semantic_record_count": 2,
        "source": source,
        "semantic_spec_sha256": "a" * 64,
        "parser_runtime_identity_sha256": "b" * 64,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    fragments = _NoSliceFragments(packer._semantic_record_fragments(row))
    monkeypatch.setattr(packer, "_semantic_record_fragments", lambda _: fragments)

    examples, rejections = packer.pack_row(row)

    assert examples
    assert not rejections
