from __future__ import annotations

from pathlib import Path

import pytest

from bcnlt.adaptive_vocab.hf_artifact import (
    load_adaptive_tokenizer_artifact,
    save_adaptive_tokenizer_artifact,
)
from bcnlt.adaptive_vocab.manifest import AdaptiveVocabularyManifest, MacroToken
from bcnlt.adaptive_vocab.tokenizer import AdaptiveBCNLTTokenizer
from bcnlt.hf_tokenizer import HuggingFaceBaseTokenizer, identity_from_directory
from bcnlt.semantic_record import ScoreBand, SemanticTokenRecord
from bcnlt.tokenizer import BCNLTTokenizer
from bcnlt.tokenizer_manifest import build_manifest
from tests.tokenizer_fixtures import write_byte_level_tokenizer


def _v1(tmp_path: Path):
    base_path = write_byte_level_tokenizer(tmp_path / "base")
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    identity = identity_from_directory(base_path, name="fixture/base", revision="local-v1")
    return BCNLTTokenizer(base, build_manifest(base, identity)), base_path


def _manifest(v1: BCNLTTokenizer, expansion: tuple[int, ...], field: str = "surface"):
    return AdaptiveVocabularyManifest.create(
        v1_manifest_sha256=v1.manifest.manifest_sha256,
        candidate_report_sha256="b" * 64,
        config_sha256="c" * 64,
        lineage_sha256="d" * 64,
        base_vocab_size=v1.manifest.base_vocab_size,
        v1_vocab_size=v1.manifest.vocab_size,
        macros=(
            MacroToken(
                "<|bcnlt:macro:0000|>",
                v1.manifest.vocab_size,
                expansion,
                (field,),
            ),
        ),
    )


def _record(surface: str, *, scope: str = "", symbol_id=None) -> SemanticTokenRecord:
    return SemanticTokenRecord(
        event="REFERENCE",
        surface=surface,
        scope=scope,
        span=(1, 0, 1, max(1, len(surface))),
        symbol_id=symbol_id,
        defines=None,
        ast_relation=None,
        ir=None,
        recent_history=(),
        metadata=(),
        confidence_band=ScoreBand.HIGH,
        uncertainty_band=ScoreBand.LOW,
    )


def test_text_and_raw_code_delegate_exactly_while_semantic_payload_uses_macro(tmp_path: Path):
    v1, _ = _v1(tmp_path)
    expansion = tuple(v1.base.encode("value", add_special_tokens=False))
    adaptive = AdaptiveBCNLTTokenizer(v1, _manifest(v1, expansion))
    record = _record("valuevalue")

    assert adaptive.encode_text("value") == v1.encode_text("value")
    assert adaptive.encode_raw_code("value\r\n") == v1.encode_raw_code("value\r\n")
    assert adaptive.manifest.v1_vocab_size in adaptive.encode_records((record,))
    assert adaptive.decode_records(adaptive.encode_records((record,))) == (record,)


def test_macro_never_crosses_structural_field_boundary(tmp_path: Path):
    v1, _ = _v1(tmp_path)
    surface_ids = tuple(v1.base.encode("A", add_special_tokens=False))
    scope_ids = tuple(v1.base.encode("B", add_special_tokens=False))
    adaptive = AdaptiveBCNLTTokenizer(
        v1,
        _manifest(v1, surface_ids + scope_ids),
    )

    encoded = adaptive.encode_records((_record("A", scope="B"),))

    assert adaptive.manifest.v1_vocab_size not in encoded
    assert adaptive.decode_records(encoded) == (_record("A", scope="B"),)


@pytest.mark.parametrize(
    "surface",
    [
        "",
        "é",
        "🧙‍♂️",
        "line one\nline two\n",
        "quote: '\"' and slash: \\",
        "<|bcnlt:event:call|>",
    ],
)
def test_exact_surface_forms_round_trip_through_expansion(tmp_path: Path, surface: str):
    v1, _ = _v1(tmp_path)
    expansion = tuple(v1.base.encode("line", add_special_tokens=False))
    adaptive = AdaptiveBCNLTTokenizer(v1, _manifest(v1, expansion))
    record = _record(surface, symbol_id=None)

    encoded = adaptive.encode_records((record,))

    assert adaptive.decode_records(encoded) == (record,)
    assert adaptive.decode(encoded) == v1.decode(v1.encode_records((record,)))


@pytest.mark.parametrize("value", ["e\u0301", "line one\r\nline two\r\n"])
def test_noncanonical_surface_forms_remain_exact_in_delegated_modes(
    tmp_path: Path, value: str
):
    v1, _ = _v1(tmp_path)
    expansion = tuple(v1.base.encode("line", add_special_tokens=False))
    adaptive = AdaptiveBCNLTTokenizer(v1, _manifest(v1, expansion))

    text_ids = adaptive.encode_text(value)
    code_ids = adaptive.encode_raw_code(value)

    assert text_ids == v1.encode_text(value)
    assert code_ids == v1.encode_raw_code(value)
    assert adaptive.decode(text_ids) == value
    assert adaptive.decode(code_ids) == value


def test_save_load_preserves_adaptive_and_v1_hashes(tmp_path: Path):
    v1, base_path = _v1(tmp_path)
    expansion = tuple(v1.base.encode("value", add_special_tokens=False))
    adaptive = AdaptiveBCNLTTokenizer(v1, _manifest(v1, expansion))
    artifact = tmp_path / "adaptive"

    save_adaptive_tokenizer_artifact(artifact, adaptive)
    loaded = load_adaptive_tokenizer_artifact(artifact, base_path)

    assert loaded.manifest == adaptive.manifest
    assert loaded.v1.manifest == v1.manifest
    assert loaded.encode_records((_record("value"),)) == adaptive.encode_records((_record("value"),))


def test_reload_rejects_wrong_base_and_unknown_macro_ids(tmp_path: Path):
    v1, base_path = _v1(tmp_path)
    expansion = tuple(v1.base.encode("value", add_special_tokens=False))
    adaptive = AdaptiveBCNLTTokenizer(v1, _manifest(v1, expansion))
    artifact = tmp_path / "adaptive"
    save_adaptive_tokenizer_artifact(artifact, adaptive)
    wrong_base = write_byte_level_tokenizer(tmp_path / "wrong", corpus="different corpus")

    with pytest.raises(ValueError, match="base tokenizer|base vocabulary"):
        load_adaptive_tokenizer_artifact(artifact, wrong_base)
    with pytest.raises(ValueError, match="outside the adaptive vocabulary"):
        adaptive.decode([adaptive.manifest.vocab_size])
