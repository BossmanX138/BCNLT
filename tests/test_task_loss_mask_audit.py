from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.current_corpus import SourceCandidate
from bosscrafts_model.resumable_current_corpus import ResumableCurrentCorpusAdmission
from bosscrafts_model.resumable_task_packing import ResumableCurrentTaskPacking
from bosscrafts_model.task_loss_mask_audit import TaskLossMaskAuditor
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
    result = TaskLossMaskAuditor(
        binding, expected_languages=("python",)
    ).materialize(databases, output)
    assert result.audit_passed is True
    assert result.training_ready is False
    assert result.minimum_target_tokens > 0
    assert set(result.split_example_counts) == {"train", "validation", "test"}
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["cross_split_document_count"] == 0
    assert receipt["cross_split_example_count"] == 0
    assert receipt["receipt_sha256"] == result.receipt_sha256
