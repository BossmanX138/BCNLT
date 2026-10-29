from __future__ import annotations

import ast
import hashlib
import json

from bcnlt.adapters.python import PythonAdapter
from bcnlt.canonical import canonical_record_bytes, record_from_canonical_bytes
from bcnlt.corpus.curriculum import generate_document_curriculum
from bcnlt.corpus.database import SourceDocumentRecord, stable_document_id
from bcnlt.semantic_record import records_from_units


SOURCE = "rate = 2\n\ndef price(value):\n    return value * rate\n"


def _fixture(split: str = "train"):
    content_hash = hashlib.sha256(SOURCE.encode()).hexdigest()
    document = SourceDocumentRecord(
        document_id=stable_document_id(
            provider="local",
            revision="a" * 40,
            repository_identity="BossCrafts/example",
            relative_path="price.py",
            content_sha256=content_hash,
        ),
        source_provider="local",
        source_name="BossCrafts/example",
        source_revision="a" * 40,
        canonical_language="python",
        repository_identity="BossCrafts/example",
        relative_path="price.py",
        content_sha256=content_hash,
        content=SOURCE,
        license_spdx="owner_provided",
        ownership="owner_provided",
        attribution="BossCrafts/example",
        upstream_url="owner-provided",
    )
    adapter = PythonAdapter()
    parsed = adapter.parse(SOURCE, path="price.py")
    units = tuple(adapter.to_units(parsed))
    records = records_from_units(units, parsed.index)
    return document, units, records, split


def test_one_deterministic_golden_record_per_core_family():
    document, units, records, split = _fixture()
    first = list(generate_document_curriculum(document, units, records, split=split))
    second = list(generate_document_curriculum(document, units, records, split=split))
    by_family = {record.family: record for record in first}
    canonical = b"".join(canonical_record_bytes(record) for record in records).decode()

    assert first == second
    assert len({record.record_id for record in first}) == len(first)
    assert {
        "semantic_continuation",
        "semantic_to_code",
        "code_to_semantic",
        "structural_reasoning",
        "mixed_instruction",
    } <= set(by_family)
    assert by_family["semantic_continuation"].template_id == "semantic-next-record-v1"
    assert by_family["semantic_continuation"].prompt.startswith(
        "Continue this canonical BCNLT semantic stream with the next record:\n"
    )
    assert by_family["semantic_to_code"].prompt == (
        "Reconstruct parseable Python from these canonical BCNLT records:\n" + canonical
    )
    assert by_family["semantic_to_code"].target == SOURCE
    assert by_family["code_to_semantic"].prompt == (
        "Convert this Python source into canonical BCNLT semantic records.\n"
        "Source path: price.py\n"
        "Original line start: 1\n"
        "Prior semantic history: []\n"
        "Python source:\n"
        + SOURCE
    )
    assert by_family["code_to_semantic"].target == canonical
    assert by_family["mixed_instruction"].prompt.startswith(
        "Use the semantic context to return the original parseable Python.\n"
    )


def test_semantic_targets_round_trip_and_code_targets_parse():
    document, units, records, split = _fixture()
    generated = list(generate_document_curriculum(document, units, records, split=split))

    for item in generated:
        if item.target_mode == "raw_code":
            ast.parse(item.target)
    semantic = next(item for item in generated if item.family == "code_to_semantic")
    decoded = tuple(
        record_from_canonical_bytes((line + "\n").encode())
        for line in semantic.target.splitlines()
    )
    assert decoded == records


def test_structural_reasoning_answers_use_definition_links_and_inherit_split():
    document, units, records, _ = _fixture()
    generated = list(
        generate_document_curriculum(document, units, records, split="validation")
    )
    reasoning = [item for item in generated if item.family == "structural_reasoning"]
    linked_records = [record for record in records if record.defines]

    assert reasoning
    assert {item.target for item in reasoning} == {
        record.defines for record in linked_records
    }
    assert all(item.split == "validation" for item in generated)
    assert all(item.source_document_ids == (document.document_id,) for item in generated)
    assert all(dict(item.provenance)["content_sha256"] == document.content_sha256 for item in generated)
    assert all(dict(item.provenance)["canonical_language"] == "python" for item in generated)


def test_generation_does_not_multiply_unchanged_facts_by_paraphrase():
    document, units, records, split = _fixture()
    generated = list(generate_document_curriculum(document, units, records, split=split))

    template_fact_pairs = [(item.template_id, item.target) for item in generated]
    assert len(template_fact_pairs) == len(set(template_fact_pairs))
    assert sum(item.family == "semantic_continuation" for item in generated) == 1
    assert sum(item.family == "semantic_to_code" for item in generated) == 1
    assert sum(item.family == "code_to_semantic" for item in generated) == 1
    assert sum(item.family == "mixed_instruction" for item in generated) == 1


def test_large_modules_generate_parseable_top_level_windows_for_all_core_families():
    source = "\n\n".join(
        f"def function_{index}(value):\n    result = value + {index}\n    return result"
        for index in range(80)
    ) + "\n"
    content_hash = hashlib.sha256(source.encode()).hexdigest()
    document = SourceDocumentRecord(
        document_id=stable_document_id(
            provider="local",
            revision="b" * 40,
            repository_identity="BossCrafts/large-example",
            relative_path="large.py",
            content_sha256=content_hash,
        ),
        source_provider="local",
        source_name="BossCrafts/large-example",
        source_revision="b" * 40,
        canonical_language="python",
        repository_identity="BossCrafts/large-example",
        relative_path="large.py",
        content_sha256=content_hash,
        content=source,
        license_spdx="owner_provided",
        ownership="owner_provided",
        attribution="BossCrafts/large-example",
        upstream_url="owner-provided",
    )
    adapter = PythonAdapter()
    parsed = adapter.parse(source, path="large.py")
    units = tuple(adapter.to_units(parsed))
    records = records_from_units(units, parsed.index)

    generated = list(generate_document_curriculum(document, units, records))
    windowed = [
        item
        for item in generated
        if item.family in {"semantic_to_code", "code_to_semantic", "mixed_instruction"}
    ]

    assert {item.family for item in windowed} == {
        "semantic_to_code",
        "code_to_semantic",
        "mixed_instruction",
    }
    assert len(windowed) > 3
    assert all("source_line_start" in dict(item.provenance) for item in windowed)
    for item in windowed:
        if item.target_mode == "raw_code":
            ast.parse(item.target)
        if item.family == "code_to_semantic":
            target_records = tuple(
                record_from_canonical_bytes((line + "\n").encode())
                for line in item.target.splitlines()
            )
            provenance = dict(item.provenance)
            assert (
                f"Original line start: {provenance['source_line_start']}\n"
                in item.prompt
            )
            assert (
                "Prior semantic history: "
                + json.dumps(
                    list(target_records[0].recent_history),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            ) in item.prompt


def test_code_to_semantic_windows_disclose_hidden_context_needed_by_targets():
    document, units, records, split = _fixture()
    generated = list(generate_document_curriculum(document, units, records, split=split))
    item = next(row for row in generated if row.family == "code_to_semantic")
    target_records = tuple(
        record_from_canonical_bytes((line + "\n").encode())
        for line in item.target.splitlines()
    )

    assert f"Source path: {document.relative_path}\n" in item.prompt
    assert "Original line start: 1\n" in item.prompt
    assert (
        "Prior semantic history: "
        + json.dumps(
            list(target_records[0].recent_history),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ) in item.prompt
