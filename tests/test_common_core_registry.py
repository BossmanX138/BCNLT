from pathlib import Path

import pytest

from bcnlt.common_core_registry import (
    COMMON_CORE_REGISTRY_VERSION,
    REQUIRED_GRAMMAR_CONCEPTS,
    build_semantic_spec,
    registry_digest,
    syntax_aliases_for,
)
from bcnlt.events import CURRENT_EVENT_VALUES
from bcnlt.ir import COMMON_CODING_CONCEPTS, IRKind
from bcnlt.languages import CANONICAL_LANGUAGE_IDS


EXPECTED_CONCEPTS = (
    "literal",
    "variable_definition",
    "reference",
    "parameter",
    "type_definition",
    "function",
    "call",
    "assignment",
    "state",
    "loop",
    "branch",
    "comparison",
    "data_structure",
    "io",
    "error",
    "return",
    "operator",
    "member_access",
    "import",
)


def test_common_coding_concepts_are_defined_once_in_one_catalog():
    assert COMMON_CORE_REGISTRY_VERSION == "bcnlt-common-core-v1"
    assert tuple(concept.concept_id for concept in COMMON_CODING_CONCEPTS) == EXPECTED_CONCEPTS
    assert len({concept.concept_id for concept in COMMON_CODING_CONCEPTS}) == len(
        COMMON_CODING_CONCEPTS
    )
    assert all(concept.event in CURRENT_EVENT_VALUES for concept in COMMON_CODING_CONCEPTS)
    assert all(concept.ir_kind is not IRKind.EXTENSION for concept in COMMON_CODING_CONCEPTS)
    assert len(registry_digest()) == 64


@pytest.mark.parametrize("language", CANONICAL_LANGUAGE_IDS)
def test_every_grammar_projects_required_shared_syntax_into_the_same_core(language: str):
    aliases = syntax_aliases_for(language)
    mapped = {alias.concept_id for alias in aliases}
    spec = build_semantic_spec(language)

    assert REQUIRED_GRAMMAR_CONCEPTS <= mapped
    assert spec.node_rules
    assert all(rule.ir_kind != IRKind.EXTENSION.value for rule in spec.node_rules)


def test_registry_builds_specs_without_per_language_implementation_modules():
    adapters = Path(__file__).parents[1] / "bcnlt" / "adapters"
    forbidden = {
        "javascript.py",
        "typescript.py",
        "c.py",
        "cpp.py",
        "csharp.py",
        "java.py",
        "go.py",
        "rust.py",
        "php.py",
        "ruby.py",
        "kotlin.py",
        "swift.py",
        "shell.py",
    }

    assert not forbidden.intersection(path.name for path in adapters.iterdir())
    assert build_semantic_spec("python").canonical_hash != build_semantic_spec(
        "javascript"
    ).canonical_hash


def test_registry_requires_exact_canonical_language_ids():
    with pytest.raises(ValueError, match="canonical language"):
        build_semantic_spec("c++")
