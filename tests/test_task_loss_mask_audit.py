from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.current_corpus import SourceCandidate
from bosscrafts_model.resumable_current_corpus import ResumableCurrentCorpusAdmission
from bosscrafts_model.resumable_task_packing import ResumableCurrentTaskPacking
from bosscrafts_model.task_database_audit_manifest import (
    materialize_task_database_audit_manifest,
)
from bosscrafts_model.task_loss_mask_audit import (
    TaskLossMaskAuditor,
    _assert_cross_split_example_ids_disjoint,
    _read_loss_mask_summary,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _candidate(name: str) -> SourceCandidate:
    source = f"def {name}(value):\n    return value + 1\n"
    return SourceCandidate(
        document_id=f"{name}-document",
        language="python",
        source=source,
        relative_path=f"{name}.py",
        content_sha256=hashlib.sha256(source.encode()).hexdigest(),
        license_spdx="MIT",
        attribution="fixture",
        ownership="third_party",
        source_provider="fixture",
        source_name="fixture/loss-mask",
        source_revision="v1",
        repository_identity=f"fixture/{name}",
        upstream_url="https://example.invalid/fixture",
    )


def _catalog(path: Path, candidates: tuple[SourceCandidate, ...]) -> str:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """CREATE TABLE documents(
                   document_id TEXT PRIMARY KEY, canonical_language TEXT NOT NULL,
                   content TEXT NOT NULL, relative_path TEXT NOT NULL,
                   content_sha256 TEXT NOT NULL, license_spdx TEXT NOT NULL,
                   attribution TEXT NOT NULL, ownership TEXT NOT NULL,
                   source_provider TEXT NOT NULL, source_name TEXT NOT NULL,
                   source_revision TEXT NOT NULL, repository_identity TEXT NOT NULL,
                   upstream_url TEXT NOT NULL, dialect TEXT)"""
        )
        connection.executemany(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    row.document_id,
                    row.language,
                    row.source,
                    row.relative_path,
                    row.content_sha256,
                    row.license_spdx,
                    row.attribution,
                    row.ownership,
                    row.source_provider,
                    row.source_name,
                    row.source_revision,
                    row.repository_identity,
                    row.upstream_url,
                    row.dialect,
                )
                for row in candidates
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id_database(path: Path, example_ids: tuple[str, ...]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE tasks(example_id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO tasks VALUES(?)", ((example_id,) for example_id in example_ids)
        )
        connection.commit()
    finally:
        connection.close()


def test_streaming_cross_split_identity_check_is_disjoint_and_count_bound(
    tmp_path: Path,
) -> None:
    databases = {
        split: tmp_path / f"{split}.sqlite"
        for split in ("train", "validation", "test")
    }
    _id_database(databases["train"], ("1", "4"))
    _id_database(databases["validation"], ("2", "5"))
    _id_database(databases["test"], ("3", "6"))
    counts = {"train": 2, "validation": 2, "test": 2}

    _assert_cross_split_example_ids_disjoint(databases, counts)
    with pytest.raises(ValueError, match="changed during loss-mask audit"):
        _assert_cross_split_example_ids_disjoint(
            databases, {"train": 1, "validation": 2, "test": 2}
        )


def test_streaming_cross_split_identity_check_rejects_overlap(tmp_path: Path) -> None:
    databases = {
        split: tmp_path / f"{split}.sqlite"
        for split in ("train", "validation", "test")
    }
    _id_database(databases["train"], ("1", "4"))
    _id_database(databases["validation"], ("2", "4"))
    _id_database(databases["test"], ("3", "6"))

    with pytest.raises(ValueError, match="identity crosses frozen splits"):
        _assert_cross_split_example_ids_disjoint(
            databases, {"train": 2, "validation": 2, "test": 2}
        )


def test_loss_mask_summary_rejects_document_task_count_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "tasks.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE tasks(
                split TEXT NOT NULL,
                language TEXT NOT NULL,
                family TEXT NOT NULL,
                document_id TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                loss_start_index INTEGER NOT NULL
            );
            CREATE TABLE documents(
                document_id TEXT PRIMARY KEY,
                task_count INTEGER NOT NULL,
                rejected_count INTEGER NOT NULL
            );
            INSERT INTO tasks VALUES('train','python','code_to_semantic','doc',8,4);
            INSERT INTO documents VALUES('doc',2,0);
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="document accounting"):
        _read_loss_mask_summary(database)


def test_materializes_complete_disjoint_loss_mask_receipt(tmp_path: Path):
    binding = NativeTokenizerBinding.load()
    pipeline = BCNLTSourcePipeline(binding)
    candidates = tuple(_candidate(split) for split in ("train", "validation", "test"))
    catalog = tmp_path / "catalog.sqlite"
    catalog_sha = _catalog(catalog, candidates)
    databases: dict[str, Path] = {}
    try:
        for split, candidate in zip(
            ("train", "validation", "test"), candidates, strict=True
        ):
            admission_path = tmp_path / f"{split}-admission.sqlite"
            task_path = tmp_path / f"{split}-tasks.sqlite"
            admission = ResumableCurrentCorpusAdmission(
                pipeline,
                admission_path,
                source_catalog_sha256=catalog_sha,
                review_manifest_sha256="a" * 64,
                host_code_identity_sha256="b" * 64,
            )
            assert admission.process((candidate,)).catalog_exhausted is True
            packing = ResumableCurrentTaskPacking(
                binding,
                admission_path,
                catalog,
                task_path,
                max_sequence=2048,
                forced_split=split,
            )
            assert packing.process().packing_complete is True
            databases[split] = task_path
    finally:
        pipeline.close()

    output = tmp_path / "loss-mask-receipt.json"
    audit_manifests = {
        split: tmp_path / f"{split}-task-audit.json" for split in databases
    }
    for split, database in databases.items():
        materialize_task_database_audit_manifest(
            binding, database, audit_manifests[split]
        )
    result = TaskLossMaskAuditor(
        binding, expected_languages=("python",)
    ).materialize(databases, output, audit_manifests=audit_manifests)
    assert result.audit_passed is True
    assert result.training_ready is False
    assert result.minimum_target_tokens > 0
    assert set(result.split_example_counts) == {"train", "validation", "test"}
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["cross_split_document_count"] == 0
    assert receipt["cross_split_example_count"] == 0
    assert receipt["receipt_sha256"] == result.receipt_sha256
