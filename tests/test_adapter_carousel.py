from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bcnlt.adapters.carousel import BCNLTMultilingualAdapter, LanguageCarousel
from bcnlt.adapters.coverage import SemanticCoverage, SemanticExtraction
from bcnlt.adapters.spec import LanguageSemanticSpec, NodeRule
from bcnlt.core import ContextualUnit, SourceDocument
from bcnlt.languages import CANONICAL_LANGUAGE_IDS, resolve_language_id
from bcnlt.semantic_schema import SEMANTIC_SCHEMA_VERSION, UNCERTAINTY_METADATA_KEYS


EXPECTED_LANGUAGES = (
    "python",
    "javascript",
    "typescript",
    "c",
    "cpp",
    "c_sharp",
    "java",
    "go",
    "rust",
    "php",
    "ruby",
    "kotlin",
    "swift",
    "bash",
)


class _FakeEngine:
    def __init__(self) -> None:
        self.parsed = []
        self.extracted = []

    def parse(self, source, *, cartridge, path, dialect):
        self.parsed.append((source, cartridge.definition.canonical_id, path, dialect))
        return SourceDocument(
            path=path,
            source=source,
            tree={"language": cartridge.definition.grammar_id},
            index=None,
            language=cartridge.definition.canonical_id,
            dialect=dialect,
        )

    def extract(self, document, *, cartridge):
        self.extracted.append((document.path, cartridge.definition.canonical_id))
        unit = ContextualUnit("answer", "DEFINE", "module", (0, 0, 0, 6))
        return SemanticExtraction(
            schema_version=SEMANTIC_SCHEMA_VERSION,
            language=document.language,
            dialect=document.dialect,
            document=document,
            units=(unit,),
            coverage=SemanticCoverage(total_units=1, common_core_units=1, unknown_units=0),
        )


def test_carousel_lists_exactly_14_canonical_language_ids():
    carousel = LanguageCarousel.default()

    assert CANONICAL_LANGUAGE_IDS == EXPECTED_LANGUAGES
    assert carousel.language_ids == EXPECTED_LANGUAGES
    assert tuple(item.definition.canonical_id for item in carousel.cartridges) == EXPECTED_LANGUAGES


def test_carousel_requires_exact_ids_while_registry_owns_aliases():
    carousel = LanguageCarousel.default()

    assert resolve_language_id("csharp") == "c_sharp"
    assert resolve_language_id("shell") == "bash"
    with pytest.raises(ValueError, match="canonical language id"):
        carousel.for_language("csharp")
    with pytest.raises(ValueError, match="canonical language id"):
        carousel.for_language("python3")


def test_public_adapter_selects_only_from_explicit_metadata():
    engine = _FakeEngine()
    adapter = BCNLTMultilingualAdapter(engine=engine)

    document = adapter.parse("let answer = 42;", language="javascript", path="misleading.py")
    extraction = adapter.extract(document)

    assert document.language == "javascript"
    assert engine.parsed == [("let answer = 42;", "javascript", "misleading.py", None)]
    assert extraction.schema_version == "bcnlt-semantic-record-v2"
    assert extraction.language == "javascript"
    assert extraction.coverage.ratio == 1.0
    assert engine.extracted == [("misleading.py", "javascript")]


def test_cartridges_are_semantic_rules_not_models_or_loras():
    adapter = BCNLTMultilingualAdapter(engine=_FakeEngine())

    assert type(adapter).__name__ == "BCNLTMultilingualAdapter"
    for cartridge in adapter.carousel.cartridges:
        assert not hasattr(cartridge, "model")
        assert not hasattr(cartridge, "lora")
        assert not hasattr(cartridge, "adapter")
        assert cartridge.spec.semantic_spec_version


def test_semantic_spec_hash_resolves_inherited_family_rules():
    family = LanguageSemanticSpec(
        semantic_spec_version="family-v1",
        node_rules=(NodeRule("call_expression", "CALL", "call"),),
    )
    child = LanguageSemanticSpec(
        semantic_spec_version="child-v1",
        inherited=(family,),
        node_rules=(NodeRule("return_statement", "RETURN", "return"),),
    )
    same_resolved_rules = LanguageSemanticSpec(
        semantic_spec_version="child-v1",
        node_rules=(
            NodeRule("call_expression", "CALL", "call"),
            NodeRule("return_statement", "RETURN", "return"),
        ),
    )

    assert child.resolved_node_rules == same_resolved_rules.resolved_node_rules
    assert child.canonical_hash == same_resolved_rules.canonical_hash


def test_v2_schema_adds_uncertainty_keys_without_changing_token_inventory():
    inventory = Path(__file__).parents[1] / "bcnlt" / "vocab" / "bcnlt-tokenizer-v1.tokens.json"

    assert SEMANTIC_SCHEMA_VERSION == "bcnlt-semantic-record-v2"
    assert UNCERTAINTY_METADATA_KEYS == (
        "dynamic_call",
        "dynamic_binding",
        "macro_expansion",
        "overload_unresolved",
        "preprocessor_condition",
        "unknown_node",
        "knowledge_state",
        "uyr_rounds",
        "uncertainty_signals",
        "wildcard_category",
    )
    assert hashlib.sha256(inventory.read_bytes()).hexdigest() == (
        "1745a81d29dbe617bd7c2c87f83567a1aa056953a5ff62e7e5946ebb054f64a0"
    )
