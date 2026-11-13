from __future__ import annotations

import hashlib
import sqlite3
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest

from bosscrafts_model.audited_balanced_curriculum import (
    _Candidate,
    _PromptClaimIndex,
    _push_candidate,
    build_audited_balanced_curriculum,
)
from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.current_task_packing import FAMILIES
from bosscrafts_model.resumable_current_corpus import RESUMABLE_SCHEMA
from bosscrafts_model.resumable_task_packing import _canonical_json, _encode_u16
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _candidate(
    number: int,
    *,
    prompt: str = "a",
    target: str = "b",
    split: str = "train",
    target_tokens: int = 8,
) -> _Candidate:
    return _Candidate(
        example_id=f"{number:064x}",
        document_id=f"document-{number}",
        language="python",
        family="semantic_to_code",
        record_index=number,
        split=split,
        input_ids_zlib=b"payload",
        token_count=16,
        loss_start_index=8,
        semantic_spec_sha256="c" * 64,
        parser_runtime_identity_sha256="d" * 64,
        source_sha256="e" * 64,
        prompt_key=("semantic_to_code", prompt * 64, 8),
        target_key=(target * 64, target_tokens),
        target_tokens=target_tokens,
    )


def test_prompt_claim_index_detects_global_conflicts_and_cross_split_leakage(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "claims.sqlite"
    conflict_key = _candidate(1, prompt="a").prompt_key
    cross_split_key = _candidate(3, prompt="b").prompt_key
    both_key = _candidate(5, prompt="c").prompt_key

    with _PromptClaimIndex(scratch) as claims:
        claims.add(_candidate(1, prompt="a", target="a"))
        claims.add(_candidate(2, prompt="a", target="b"))
        claims.add(_candidate(3, prompt="b", target="c", split="train"))
        claims.add(_candidate(4, prompt="b", target="c", split="test"))
        claims.add(_candidate(5, prompt="c", target="d", split="train"))
        claims.add(_candidate(6, prompt="c", target="e", split="test"))
        claims.add(_candidate(7, prompt="d", target="f"))
        claims.add(_candidate(8, prompt="d", target="f"))
        conflicting, cross_split = claims.exclusions()
        assert scratch.is_file()

    assert conflicting == {conflict_key, both_key}
    assert cross_split == {cross_split_key}
    assert not scratch.exists()
    assert not scratch.with_name(f"{scratch.name}-journal").exists()


def test_prompt_claim_index_removes_scratch_after_failure(tmp_path: Path) -> None:
    scratch = tmp_path / "claims.sqlite"
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        _PromptClaimIndex(scratch) as claims,
    ):
        claims.add(_candidate(1))
        raise RuntimeError("deliberate")
    assert not scratch.exists()


def test_bounded_candidate_heap_keeps_lowest_content_addresses() -> None:
    heap: list[tuple[int, str, _Candidate]] = []
    for number in (5, 1, 4, 2, 3):
        _push_candidate(heap, _candidate(number), 3)
    assert sorted(item.example_id for _, _, item in heap) == [
        f"{number:064x}" for number in (1, 2, 3)
    ]


class _FixturePacker:
    def __init__(self, binding: NativeTokenizerBinding, *, max_sequence: int) -> None:
        self.binding = binding
        self.max_sequence = max_sequence

    def pack_row(self, row: dict[str, object]):
        document_id = str(row["document_id"])
        marker = int(document_id.rsplit("-", 1)[1])
        examples = []
        for record_index, family in enumerate(FAMILIES):
            input_ids = [100, marker, 200 + record_index, 300 + record_index]
            identity = {
                "document_id": document_id,
                "family": family,
                "record_index": record_index,
                "input_ids_sha256": hashlib.sha256(
                    _canonical_json(input_ids).encode()
                ).hexdigest(),
            }
            examples.append(
                {
                    "example_id": hashlib.sha256(
                        _canonical_json(identity).encode()
                    ).hexdigest(),
                    "family": family,
                    "record_index": record_index,
                    "input_ids": input_ids,
                    "loss_start_index": 2,
                    "semantic_spec_sha256": row["semantic_spec_sha256"],
                    "parser_runtime_identity_sha256": row[
                        "parser_runtime_identity_sha256"
                    ],
                }
            )
        return examples, []


def _write_balanced_fixture(root: Path, binding: NativeTokenizerBinding):
    catalog_path = root / "catalog.sqlite"
    documents_by_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    sources: dict[str, str] = {}
    marker = 1
    for split, split_documents in documents_by_split.items():
        for _ in range(8):
            document_id = f"{split}-document-{marker}"
            split_documents.add(document_id)
            sources[document_id] = f"value_{marker} = {marker}\n"
            marker += 1

    catalog = sqlite3.connect(catalog_path)
    try:
        catalog.execute(
            """CREATE TABLE documents(
                   document_id TEXT PRIMARY KEY,canonical_language TEXT NOT NULL,
                   content TEXT NOT NULL,relative_path TEXT NOT NULL,
                   content_sha256 TEXT NOT NULL)"""
        )
        catalog.executemany(
            "INSERT INTO documents VALUES(?,?,?,?,?)",
            [
                (
                    document_id,
                    "python",
                    source,
                    f"{document_id}.py",
                    hashlib.sha256(source.encode()).hexdigest(),
                )
                for document_id, source in sources.items()
            ],
        )
        catalog.commit()
    finally:
        catalog.close()

    admission_paths: dict[str, Path] = {}
    canonical_jsonl = '{"fixture":true}\n'
    canonical_sha256 = hashlib.sha256(canonical_jsonl.encode()).hexdigest()
    semantic_ids = _encode_u16([1, 2, 3])
    for split, document_ids in documents_by_split.items():
        admission_path = root / f"{split}-admission.sqlite"
        admission_paths[split] = admission_path
        admission = sqlite3.connect(admission_path)
        try:
            admission.execute(
                """CREATE TABLE documents(
                       document_id TEXT PRIMARY KEY,content_sha256 TEXT NOT NULL,
                       relative_path TEXT NOT NULL,semantic_spec_sha256 TEXT NOT NULL,
                       parser_runtime_identity_sha256 TEXT NOT NULL,
                       canonical_sha256 TEXT NOT NULL,canonical_jsonl_zlib BLOB NOT NULL,
                       semantic_ids_zlib BLOB NOT NULL,semantic_record_count INTEGER NOT NULL,
                       semantic_token_count INTEGER NOT NULL,status TEXT NOT NULL,
                       language TEXT NOT NULL)"""
            )
            admission.executemany(
                "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        document_id,
                        hashlib.sha256(sources[document_id].encode()).hexdigest(),
                        f"{document_id}.py",
                        CURRENT_RUNTIME_SPEC_SHA256["python"],
                        PARSER_RUNTIME_IDENTITY_SHA256,
                        canonical_sha256,
                        zlib.compress(canonical_jsonl.encode(), 6),
                        zlib.compress(semantic_ids, 6),
                        1,
                        3,
                        "admitted",
                        "python",
                    )
                    for document_id in sorted(document_ids)
                ],
            )
            admission.commit()
        finally:
            admission.close()
    return catalog_path, admission_paths, documents_by_split


def test_materializes_exact_balanced_cells_and_is_deterministic(tmp_path: Path) -> None:
    binding = NativeTokenizerBinding.load()
    catalog, admission_paths, documents_by_split = _write_balanced_fixture(
        tmp_path, binding
    )
    common_metadata = {
        "schema_version": RESUMABLE_SCHEMA,
        "catalog_exhausted": "true",
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "vocabulary_size": str(binding.vocab_size),
        "semantic_spec_sha256": _canonical_json(CURRENT_RUNTIME_SPEC_SHA256),
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "review_manifest_sha256": "a" * 64,
        "source_freeze_manifest_sha256": "b" * 64,
        "semantic_audit_qualification_sha256": "c" * 64,
    }

    def load_eligibility(
        _admission_path,
        _eligibility_root,
        expected_manifest_sha256,
        *,
        split,
        expected_catalog_sha256,
    ):
        assert (
            expected_catalog_sha256 == hashlib.sha256(catalog.read_bytes()).hexdigest()
        )
        manifest = {
            "manifest_sha256": expected_manifest_sha256,
            "source_audit_identity_sha256": hashlib.sha256(split.encode()).hexdigest(),
        }
        metadata = dict(common_metadata)
        if split != "train":
            metadata["frozen_source_split"] = split
        return (
            documents_by_split[split],
            manifest,
            metadata,
            hashlib.sha256(f"admission:{split}".encode()).hexdigest(),
        )

    arguments = {
        "admission_databases": admission_paths,
        "eligibility_roots": {
            split: tmp_path / f"{split}-eligibility" for split in documents_by_split
        },
        "eligibility_manifest_sha256": {
            split: hashlib.sha256(f"manifest:{split}".encode()).hexdigest()
            for split in documents_by_split
        },
        "source_catalog": catalog,
        "languages": ("python",),
        "quotas": {"train": 8, "validation": 8, "test": 8},
        "oversample_factor": 2,
        "max_sequence": 256,
    }
    prompt_sentinel = tmp_path / ".first.sqlite.prompt-claims.tmp"
    output_sentinel = tmp_path / ".first.sqlite.tmp"
    prompt_sentinel.write_bytes(b"unrelated prompt sentinel")
    output_sentinel.write_bytes(b"unrelated output sentinel")
    with (
        patch(
            "bosscrafts_model.audited_balanced_curriculum._load_eligibility",
            side_effect=load_eligibility,
        ),
        patch(
            "bosscrafts_model.audited_balanced_curriculum.CompactCurrentCorpusTaskPacker",
            _FixturePacker,
        ),
    ):
        first = build_audited_balanced_curriculum(
            binding,
            destination_database=tmp_path / "first.sqlite",
            **arguments,
        )
        second = build_audited_balanced_curriculum(
            binding,
            destination_database=tmp_path / "second.sqlite",
            **arguments,
        )

    overlapping_document = next(iter(documents_by_split["train"]))

    def load_overlapping_eligibility(*args, **kwargs):
        eligible, manifest, metadata, identity = load_eligibility(*args, **kwargs)
        if kwargs["split"] == "validation":
            eligible = {*eligible, overlapping_document}
        return eligible, manifest, metadata, identity

    with (
        patch(
            "bosscrafts_model.audited_balanced_curriculum._load_eligibility",
            side_effect=load_overlapping_eligibility,
        ),
        patch(
            "bosscrafts_model.audited_balanced_curriculum.CompactCurrentCorpusTaskPacker",
            _FixturePacker,
        ),
        pytest.raises(ValueError, match="multiple frozen splits"),
    ):
        build_audited_balanced_curriculum(
            binding,
            destination_database=tmp_path / "overlap.sqlite",
            **arguments,
        )

    assert prompt_sentinel.read_bytes() == b"unrelated prompt sentinel"
    assert output_sentinel.read_bytes() == b"unrelated output sentinel"
    assert first.example_count == 72
    assert first.processed_document_count == 24
    assert first.selected_document_count == 24
    assert set(first.cell_counts.values()) == {8}
    assert len(first.cell_counts) == 9
    assert first.database_identity_sha256 == second.database_identity_sha256
    assert first.policy_sha256 == second.policy_sha256
    assert not first.training_ready
