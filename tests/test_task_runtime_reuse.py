from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from bosscrafts_model.admission_runtime_delta import (
    compare_admission_runtime_delta,
    publish_delta_receipt,
)
from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    BCNLTSourcePipeline,
)
from bosscrafts_model.current_corpus import SourceCandidate
from bosscrafts_model.resumable_current_corpus import ResumableCurrentCorpusAdmission
from bosscrafts_model.resumable_task_packing import (
    ResumableCurrentTaskPacking,
    audit_task_database,
)
from bosscrafts_model.resumable_training_semantic_audit import (
    ResumableTrainingSemanticAudit,
)
from bosscrafts_model.task_database_audit_manifest import (
    materialize_task_database_audit_manifest,
)
from bosscrafts_model.task_runtime_reuse import (
    load_task_runtime_reuse_receipt,
    migrate_reusable_tasks,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding

OLD_RUNTIME = "1" * 64
OLD_PYTHON_SPEC = "2" * 64


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _candidate(document_id: str, source: str) -> SourceCandidate:
    return SourceCandidate(
        document_id=document_id,
        language="python",
        source=source,
        relative_path=f"{document_id}.py",
        content_sha256=hashlib.sha256(source.encode()).hexdigest(),
        license_spdx="MIT",
        attribution="fixture-author",
        ownership="third_party",
        source_provider="fixture",
        source_name="fixture/runtime-reuse",
        source_revision="fixture-v1",
        repository_identity="fixture/repository",
        upstream_url="https://example.invalid/fixture",
    )


def _catalog(path: Path, candidates: tuple[SourceCandidate, ...]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """CREATE TABLE documents (
                   document_id TEXT PRIMARY KEY,
                   canonical_language TEXT NOT NULL,
                   content TEXT NOT NULL,
                   relative_path TEXT NOT NULL,
                   content_sha256 TEXT NOT NULL,
                   license_spdx TEXT NOT NULL,
                   attribution TEXT NOT NULL,
                   ownership TEXT NOT NULL,
                   source_provider TEXT NOT NULL,
                   source_name TEXT NOT NULL,
                   source_revision TEXT NOT NULL,
                   repository_identity TEXT NOT NULL,
                   upstream_url TEXT NOT NULL,
                   dialect TEXT
               )"""
        )
        connection.executemany(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                (
                    item.document_id,
                    item.language,
                    item.source,
                    item.relative_path,
                    item.content_sha256,
                    item.license_spdx,
                    item.attribution,
                    item.ownership,
                    item.source_provider,
                    item.source_name,
                    item.source_revision,
                    item.repository_identity,
                    item.upstream_url,
                    item.dialect,
                )
                for item in candidates
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _admission_identity(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        return ResumableTrainingSemanticAudit._admission_identity(
            connection, metadata
        )
    finally:
        connection.close()


def _packing_admission_identity(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        counts = connection.execute(
            """SELECT SUM(CASE WHEN status='admitted' THEN 1 ELSE 0 END),
                      SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END)
               FROM documents"""
        ).fetchone()
    finally:
        connection.close()
    assert counts is not None
    payload = {
        "metadata": metadata,
        "admitted_count": int(counts[0] or 0),
        "rejected_count": int(counts[1] or 0),
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _eligibility(
    root: Path,
    admission: Path,
    catalog_sha256: str,
    document_ids: tuple[str, ...],
) -> str:
    root.mkdir()
    eligible_text = "".join(
        f'{_canonical({"schema_version": "bosscrafts-training-eligible-document-v1", "document_id": document_id})}\n'
        for document_id in document_ids
    )
    manifest = {
        "schema_version": "bosscrafts-training-eligibility-manifest-v1",
        "eligibility_complete": True,
        "training_ready": False,
        "source_catalog_sha256": catalog_sha256,
        "source_split": "train",
        "source_admission_identity_sha256": _admission_identity(admission),
        "source_audit_identity_sha256": "3" * 64,
        "eligible_file": "eligible.jsonl",
        "eligible_sha256": hashlib.sha256(eligible_text.encode()).hexdigest(),
        "eligible_document_count": len(document_ids),
    }
    manifest_sha256 = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    manifest["manifest_sha256"] = manifest_sha256
    (root / "eligible.jsonl").write_text(eligible_text, encoding="utf-8")
    (root / "manifest.json").write_text(
        f"{_canonical(manifest)}\n", encoding="utf-8"
    )
    return manifest_sha256


def _convert_to_historical_admission(path: Path) -> None:
    specs = dict(CURRENT_RUNTIME_SPEC_SHA256)
    specs["python"] = OLD_PYTHON_SPEC
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='parser_runtime_identity_sha256'",
            (OLD_RUNTIME,),
        )
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='host_code_identity_sha256'",
            ("4" * 64,),
        )
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='semantic_spec_sha256'",
            (_canonical(specs),),
        )
        connection.execute(
            """UPDATE documents
               SET parser_runtime_identity_sha256=?,semantic_spec_sha256=?
               WHERE status='admitted'""",
            (OLD_RUNTIME, OLD_PYTHON_SPEC),
        )
        connection.commit()
    finally:
        connection.close()


def _convert_to_historical_tasks(path: Path, admission: Path) -> None:
    specs = dict(CURRENT_RUNTIME_SPEC_SHA256)
    specs["python"] = OLD_PYTHON_SPEC
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='parser_runtime_identity_sha256'",
            (OLD_RUNTIME,),
        )
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='semantic_spec_sha256'",
            (_canonical(specs),),
        )
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='source_admission_identity_sha256'",
            (_packing_admission_identity(admission),),
        )
        connection.execute(
            "UPDATE tasks SET parser_runtime_identity_sha256=?,semantic_spec_sha256=?",
            (OLD_RUNTIME, OLD_PYTHON_SPEC),
        )
        connection.commit()
    finally:
        connection.close()


def test_reuse_then_repack_matches_fresh_current_pack(tmp_path: Path) -> None:
    binding = NativeTokenizerBinding.load()
    pipeline = BCNLTSourcePipeline(binding)
    candidates = (
        _candidate("doc-a", "x = 1\nprint(x)\n"),
        _candidate("doc-b", "def add(a, b):\n    return a + b\n"),
    )
    catalog = tmp_path / "catalog.sqlite"
    candidate_admission = tmp_path / "candidate-admission.sqlite"
    baseline_admission = tmp_path / "baseline-admission.sqlite"
    baseline_tasks = tmp_path / "baseline-tasks.sqlite"
    baseline_audit = tmp_path / "baseline-task-audit.json"
    delta_receipt = tmp_path / "delta.json"
    target_tasks = tmp_path / "target-tasks.sqlite"
    reuse_receipt = tmp_path / "reuse.json"
    fresh_tasks = tmp_path / "fresh-tasks.sqlite"
    _catalog(catalog, candidates)
    catalog_sha256 = hashlib.sha256(catalog.read_bytes()).hexdigest()
    admission = ResumableCurrentCorpusAdmission(
        pipeline,
        candidate_admission,
        source_catalog_sha256=catalog_sha256,
        review_manifest_sha256="5" * 64,
        host_code_identity_sha256="6" * 64,
        source_freeze_manifest_sha256="7" * 64,
        semantic_audit_qualification_sha256="8" * 64,
        expected_source_document_count=2,
    )
    assert admission.process(candidates).catalog_exhausted is True
    shutil.copy2(candidate_admission, baseline_admission)
    _convert_to_historical_admission(baseline_admission)

    baseline_eligibility = tmp_path / "baseline-eligibility"
    baseline_eligibility_sha = _eligibility(
        baseline_eligibility,
        candidate_admission,
        catalog_sha256,
        ("doc-a",),
    )
    baseline_packer = ResumableCurrentTaskPacking(
        binding,
        candidate_admission,
        catalog,
        baseline_tasks,
        semantic_wire="compact_v2",
        eligibility_root=baseline_eligibility,
        expected_eligibility_manifest_sha256=baseline_eligibility_sha,
    )
    assert baseline_packer.process().packing_complete is True
    _convert_to_historical_tasks(baseline_tasks, baseline_admission)
    with patch(
        "bosscrafts_model.resumable_task_packing.PARSER_RUNTIME_IDENTITY_SHA256",
        OLD_RUNTIME,
    ):
        materialize_task_database_audit_manifest(
            binding, baseline_tasks, baseline_audit
        )

    delta = compare_admission_runtime_delta(
        baseline_admission, candidate_admission
    )
    assert delta.reusable_document_count == 2
    publish_delta_receipt(delta, delta_receipt)
    candidate_eligibility = tmp_path / "candidate-eligibility"
    candidate_eligibility_sha = _eligibility(
        candidate_eligibility,
        candidate_admission,
        catalog_sha256,
        ("doc-a", "doc-b"),
    )

    result = migrate_reusable_tasks(
        binding,
        baseline_task_database=baseline_tasks,
        baseline_task_audit_manifest=baseline_audit,
        baseline_admission_database=baseline_admission,
        candidate_admission_database=candidate_admission,
        admission_delta_receipt=delta_receipt,
        source_catalog=catalog,
        candidate_eligibility_root=candidate_eligibility,
        candidate_eligibility_manifest_sha256=candidate_eligibility_sha,
        target_database=target_tasks,
        output_receipt=reuse_receipt,
    )
    assert result.copied_document_count == 1
    assert result.repack_document_count == 1
    assert load_task_runtime_reuse_receipt(reuse_receipt)[
        "copied_document_count"
    ] == 1

    resumed = ResumableCurrentTaskPacking(
        binding,
        candidate_admission,
        catalog,
        target_tasks,
        semantic_wire="compact_v2",
        eligibility_root=candidate_eligibility,
        expected_eligibility_manifest_sha256=candidate_eligibility_sha,
        defer_cell_index=True,
    )
    resumed_status = resumed.process()
    assert resumed_status.processed_this_run == 1
    assert resumed_status.packing_complete is True
    migrated_audit = audit_task_database(binding, target_tasks)

    fresh = ResumableCurrentTaskPacking(
        binding,
        candidate_admission,
        catalog,
        fresh_tasks,
        semantic_wire="compact_v2",
        eligibility_root=candidate_eligibility,
        expected_eligibility_manifest_sha256=candidate_eligibility_sha,
        defer_cell_index=True,
    )
    assert fresh.process().packing_complete is True
    fresh_audit = audit_task_database(binding, fresh_tasks)
    assert migrated_audit.database_identity_sha256 == (
        fresh_audit.database_identity_sha256
    )
    reuse_receipt.write_text(
        reuse_receipt.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical"):
        load_task_runtime_reuse_receipt(reuse_receipt)
