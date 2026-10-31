from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bcnlt.corpus.database import CorpusDatabase, stable_document_id
from bcnlt.corpus.filters import evaluate_source_document
from bcnlt.corpus.sources import SourceSpec, ingest_stack_stream
from bcnlt.languages import CANONICAL_LANGUAGE_IDS
from scripts.sample_bcnlt_multilingual_sources import sample_languages, source_spec_from_entry


STACK_REVISION = "17cad72c886a2858e08d4c349a00d6466f54df63"
NVIDIA_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
EXPECTED_PARTITIONS = {
    "python": "data/python",
    "javascript": "data/javascript",
    "typescript": "data/typescript",
    "c": "data/c",
    "cpp": "data/cpp",
    "c_sharp": "data/c-sharp",
    "java": "data/java",
    "go": "data/go",
    "rust": "data/rust",
    "php": "data/php",
    "ruby": "data/ruby",
    "kotlin": "data/kotlin",
    "swift": "data/swift",
    "bash": "data/shell",
}


class _AcceptingAdapter:
    def parse(self, source: str, *, language: str, path: str, dialect: str | None = None):
        return SimpleNamespace(language=language, dialect=dialect, source=source, path=path)

    def extract(self, document):
        coverage = SimpleNamespace(
            admitted=True,
            mapped_ratio=0.90,
            unknown_ratio=0.05,
            scopes_balanced=True,
            links_valid=True,
        )
        units = tuple(
            SimpleNamespace(event=event)
            for event in (
                "define",
                "call",
                "reference",
                "branch",
                "scope_enter",
                "scope_enter",
            )
        )
        return SimpleNamespace(coverage=coverage, units=units)


def _content() -> str:
    return "function example() { return helper(42); }\n" + ("// safe fixture\n" * 40)


def _spec(language: str = "javascript") -> SourceSpec:
    return SourceSpec(
        provider="huggingface",
        name="bigcode/the-stack-dedup",
        revision=STACK_REVISION,
        upstream_url="https://huggingface.co/datasets/bigcode/the-stack-dedup",
        canonical_language=language,
        data_dir=EXPECTED_PARTITIONS[language],
    )


def test_v2_registry_pins_all_stack_partitions_and_python_only_nvidia():
    path = Path("configs/bcnlt_corpus_sources_v2.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "bcnlt-corpus-sources-v2"
    stack = [item for item in payload["sources"] if item["name"] == "bigcode/the-stack-dedup"]
    assert {item["canonical_language"]: item["data_dir"] for item in stack} == EXPECTED_PARTITIONS
    assert {item["revision"] for item in stack} == {STACK_REVISION}
    assert tuple(item["canonical_language"] for item in stack) == CANONICAL_LANGUAGE_IDS
    assert all(len(item["registry_entry_sha256"]) == 64 for item in stack)
    assert all(len(item["parser_bundle_sha256"]) == 64 for item in stack)
    assert all(len(item["semantic_spec_sha256"]) == 64 for item in stack)
    assert all(source_spec_from_entry(item).registry_entry_sha256 == item["registry_entry_sha256"] for item in stack)
    nvidia = [item for item in payload["sources"] if item["name"] == "nvidia/OpenCodeInstruct"]
    assert len(nvidia) == 1
    assert nvidia[0]["canonical_language"] == "python"
    assert nvidia[0]["revision"] == NVIDIA_REVISION


@pytest.mark.parametrize(
    ("language", "path"),
    [
        ("python", "module.pyx"),
        ("javascript", "component.jsx"),
        ("typescript", "component.tsx"),
        ("c", "include/common.h"),
        ("cpp", "include/common.h"),
        ("bash", "script.zsh"),
    ],
)
def test_deferred_extensions_fail_before_parser_selection(language: str, path: str):
    adapter = SimpleNamespace(
        parse=lambda *args, **kwargs: pytest.fail("deferred form reached parser"),
        extract=lambda *args, **kwargs: pytest.fail("deferred form reached extractor"),
    )

    decision = evaluate_source_document(
        relative_path=path,
        content=_content(),
        license_value="MIT",
        language=language,
        adapter=adapter,
    )

    assert decision.accepted is False
    assert decision.reason_code == "ambiguous_or_deferred_extension"


def test_generic_filter_parses_and_applies_semantic_coverage_gate():
    decision = evaluate_source_document(
        relative_path="src/example.js",
        content=_content(),
        license_value="MIT",
        language="javascript",
        adapter=_AcceptingAdapter(),
    )

    assert decision.accepted is True
    assert decision.event_names == (
        "define",
        "call",
        "reference",
        "branch",
        "scope_enter",
        "scope_enter",
    )
    assert decision.mapped_ratio == pytest.approx(0.90)
    assert decision.unknown_ratio == pytest.approx(0.05)


def test_stack_ingestion_persists_multilingual_lineage(tmp_path: Path):
    database = CorpusDatabase(tmp_path / "corpus.sqlite3")
    spec = _spec()
    admitted = ingest_stack_stream(
        database,
        spec,
        [
            {
                "content": _content(),
                "path": "src/example.js",
                "repository": "owner/project",
                "license": "MIT",
            }
        ],
        adapter=_AcceptingAdapter(),
    )

    assert len(admitted) == 1
    record = admitted[0]
    assert record.canonical_language == "javascript"
    assert record.language == "javascript"
    assert record.registry_entry_sha256 == spec.registry_entry_sha256
    assert record.parser_bundle_sha256 == spec.parser_bundle_sha256
    assert record.semantic_spec_sha256 == spec.semantic_spec_sha256
    assert database.documents() == admitted


def test_document_identity_changes_with_language_and_registry_identity():
    common = dict(
        provider="huggingface",
        revision=STACK_REVISION,
        repository_identity="owner/project",
        relative_path="src/example.txt",
        content_sha256=hashlib.sha256(_content().encode()).hexdigest(),
    )
    python = stable_document_id(**common, language="python", registry_entry_sha256="a" * 64)
    rust = stable_document_id(**common, language="rust", registry_entry_sha256="a" * 64)
    changed_registry = stable_document_id(
        **common, language="python", registry_entry_sha256="b" * 64
    )

    assert len({python, rust, changed_registry}) == 3


def test_bounded_sample_requires_and_reports_all_languages():
    registry = json.loads(Path("configs/bcnlt_corpus_sources_v2.json").read_text(encoding="utf-8"))
    specs = tuple(
        source_spec_from_entry(item)
        for item in registry["sources"]
        if item["name"] == "bigcode/the-stack-dedup"
    )

    def rows(spec: SourceSpec, limit: int):
        extension = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "c": ".c",
            "cpp": ".cpp",
            "c_sharp": ".cs",
            "java": ".java",
            "go": ".go",
            "rust": ".rs",
            "php": ".php",
            "ruby": ".rb",
            "kotlin": ".kt",
            "swift": ".swift",
            "bash": ".sh",
        }[spec.canonical_language]
        return [
            {
                "content": _content(),
                "path": f"src/example{extension}",
                "repository": f"owner/{spec.canonical_language}",
                "license": "MIT",
            }
        ][:limit]

    report = sample_languages(
        specs,
        row_provider=rows,
        adapter=_AcceptingAdapter(),
        rows_per_language=1,
        minimum_admitted=1,
    )

    assert report.passed is True
    assert tuple(item.language for item in report.language_reports) == CANONICAL_LANGUAGE_IDS
    assert all(item.admitted == 1 and item.passed for item in report.language_reports)
