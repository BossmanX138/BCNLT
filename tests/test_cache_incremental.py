from bcnlt.adapters.carousel import BCNLTMultilingualAdapter
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine
from bcnlt.cache import (
    ProjectIndex,
    SemanticCacheIdentity,
    adapter_cache_identity,
    semantic_cache_key,
)
from bcnlt.incremental import RegionInvalidator


def test_project_index_tracks_symbol_definitions_and_references():
    index = ProjectIndex()
    index.define("sample.py", "module.fn.x", "x", (2, 4, 2, 5))
    index.reference("sample.py", "module.fn.x", "x", (3, 11, 3, 12))

    assert index.find_definition("x").symbol_id == "module.fn.x"
    assert index.references_to("module.fn.x")[0].surface == "x"


def test_region_invalidator_marks_only_changed_files_and_line_ranges():
    invalidator = RegionInvalidator()

    invalidator.mark_changed("sample.py", start_line=4, end_line=6)

    assert invalidator.changed_files() == {"sample.py"}
    assert invalidator.changed_ranges("sample.py") == [(4, 6)]


def test_cache_identity_separates_languages_dialects_and_semantic_specs():
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    python = adapter_cache_identity(adapter, "python", "python3")
    javascript = adapter_cache_identity(adapter, "javascript", "ecmascript")

    assert python.language == "python"
    assert python.parser_bundle_sha256 == javascript.parser_bundle_sha256
    assert python.semantic_spec_sha256 != javascript.semantic_spec_sha256
    assert semantic_cache_key(path="shared", source_sha256="a" * 64, identity=python) != (
        semantic_cache_key(path="shared", source_sha256="a" * 64, identity=javascript)
    )
    base_key = semantic_cache_key(path="shared", source_sha256="a" * 64, identity=python)
    variants = (
        SemanticCacheIdentity(
            "javascript", python.dialect, python.parser_bundle_sha256,
            python.semantic_spec_sha256,
        ),
        SemanticCacheIdentity(
            python.language, "other", python.parser_bundle_sha256,
            python.semantic_spec_sha256,
        ),
        SemanticCacheIdentity(
            python.language, python.dialect, "c" * 64,
            python.semantic_spec_sha256,
        ),
        SemanticCacheIdentity(
            python.language, python.dialect, python.parser_bundle_sha256,
            "d" * 64,
        ),
    )
    assert all(
        semantic_cache_key(path="shared", source_sha256="a" * 64, identity=item) != base_key
        for item in variants
    )


def test_region_invalidator_invalidates_the_whole_document_when_identity_changes():
    invalidator = RegionInvalidator()
    first = SemanticCacheIdentity("python", None, "a" * 64, "b" * 64)
    changed = SemanticCacheIdentity("python", None, "c" * 64, "b" * 64)

    invalidator.mark_changed("sample.py", 4, 6, identity=first)
    invalidator.mark_changed("sample.py", 8, 9, identity=changed)

    assert invalidator.changed_ranges("sample.py") == [(0, 2**31 - 1)]
    assert invalidator.identity("sample.py") == changed
