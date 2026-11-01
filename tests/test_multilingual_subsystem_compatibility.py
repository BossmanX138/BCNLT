from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bcnlt.adapters.carousel import BCNLTMultilingualAdapter
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine
from bcnlt.benchmark import benchmark_source
from bcnlt.cache import adapter_cache_identity, semantic_cache_key
from bcnlt.incremental import RegionInvalidator
from bcnlt.languages import LANGUAGE_BY_ID
from bcnlt.preprocessing import process_source
from bcnlt.semantic_db import build_database
from tests.test_common_core_language_projection import SOURCES


@pytest.mark.parametrize("language", tuple(SOURCES))
def test_every_subsystem_routes_all_languages_through_one_carousel(
    language: str, tmp_path: Path
) -> None:
    source = SOURCES[language]
    extension = LANGUAGE_BY_ID[language].accepted_extensions[0]
    source_path = tmp_path / f"sample{extension}"
    source_path.write_text(source, encoding="utf-8")
    adapter = BCNLTMultilingualAdapter(
        engine=TreeSitterSemanticEngine(wildcard_memory_root=tmp_path / "wildcards")
    )
    identity = adapter_cache_identity(adapter, language)

    document = adapter.parse(source, language=language, path=str(source_path))
    extraction = adapter.extract(document)
    preprocessed = process_source(source, language=language, path=str(source_path))
    benchmark = benchmark_source(source, path=str(source_path), language=language)

    assert document.language == language
    assert document.index.language == language
    assert document.index.parser_bundle_sha256 == identity.parser_bundle_sha256
    assert document.index.semantic_spec_sha256 == identity.semantic_spec_sha256
    assert extraction.units
    assert extraction.assessments
    assert len(extraction.units) == len(extraction.assessments)
    assert preprocessed
    assert benchmark.to_dict()["language"] == language
    assert benchmark.parser_bundle_sha256 == identity.parser_bundle_sha256
    assert benchmark.semantic_spec_sha256 == identity.semantic_spec_sha256

    cache_key = semantic_cache_key(
        path=str(source_path), source_sha256="a" * 64, identity=identity
    )
    invalidator = RegionInvalidator()
    invalidator.mark_changed(str(source_path), 1, 1, identity=identity)
    assert len(cache_key) == 64
    assert invalidator.identity(str(source_path)) == identity

    database = tmp_path / "semantic-pairs.sqlite3"
    stats = build_database(tmp_path, database, language=language)
    assert stats.documents == 1
    assert stats.semantic_units > 0
    assert stats.semantic_pairs > 0
    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT language, parser_bundle_sha256, semantic_spec_sha256 "
            "FROM documents"
        ).fetchone()
        provenance = connection.execute(
            "SELECT provenance_json FROM semantic_pairs LIMIT 1"
        ).fetchone()
    assert stored == (
        language,
        identity.parser_bundle_sha256,
        identity.semantic_spec_sha256,
    )
    assert f'"language": "{language}"' in provenance[0]


def test_parser_identity_change_forces_full_incremental_invalidation() -> None:
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    identity = adapter_cache_identity(adapter, "python")
    changed = type(identity)(
        language=identity.language,
        dialect=identity.dialect,
        parser_bundle_sha256="f" * 64,
        semantic_spec_sha256=identity.semantic_spec_sha256,
    )
    invalidator = RegionInvalidator()

    invalidator.mark_changed("sample.py", 2, 3, identity=identity)
    invalidator.mark_changed("sample.py", 4, 5, identity=changed)

    assert invalidator.changed_ranges("sample.py") == [(0, 2**31 - 1)]
