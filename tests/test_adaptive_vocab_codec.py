from __future__ import annotations

from bcnlt.adaptive_vocab.codec import MacroCodec
from bcnlt.adaptive_vocab.manifest import AdaptiveVocabularyManifest, MacroToken


def _manifest() -> AdaptiveVocabularyManifest:
    macros = (
        MacroToken("<|bcnlt:macro:0000|>", 100, (1, 2), ("surface",)),
        MacroToken("<|bcnlt:macro:0001|>", 101, (1, 2, 3), ("surface",)),
        MacroToken("<|bcnlt:macro:0002|>", 102, (1, 2, 3), ("surface",)),
        MacroToken("<|bcnlt:macro:0003|>", 103, (4, 5), ("scope",)),
    )
    return AdaptiveVocabularyManifest.create(
        v1_manifest_sha256="a" * 64,
        candidate_report_sha256="b" * 64,
        config_sha256="c" * 64,
        lineage_sha256="d" * 64,
        base_vocab_size=50,
        v1_vocab_size=100,
        macros=macros,
    )


def test_longest_match_then_lower_macro_id_wins():
    codec = MacroCodec(_manifest())

    assert codec.substitute([1, 2, 3, 1, 2], "surface") == [101, 100]


def test_fields_outside_allowed_set_remain_base_ids():
    codec = MacroCodec(_manifest())

    assert codec.substitute([1, 2, 3], "scope") == [1, 2, 3]
    assert codec.substitute([4, 5], "surface") == [4, 5]


def test_substitution_round_trips_through_direct_expansion():
    codec = MacroCodec(_manifest())
    original = [9, 1, 2, 3, 4, 5, 1, 2]

    substituted = codec.substitute(original, "surface")

    assert codec.expand(substituted) == original
