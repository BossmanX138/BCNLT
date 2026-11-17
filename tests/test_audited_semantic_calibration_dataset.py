from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from bosscrafts_model.audited_semantic_calibration_dataset import (
    AuditedSemanticCalibrationDatasetBuilder,
)
from bosscrafts_model.bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from bosscrafts_model.resumable_training_semantic_audit import (
    AUDIT_SCHEMA,
    SOURCE_ADMISSION_SCHEMA,
    ResumableTrainingSemanticAudit,
)
from bosscrafts_model.semantic_calibration_promotion import (
    CalibrationThresholds,
    SemanticCalibrationPromotionPolicy,
    promote_semantic_uncertainty,
)
from bosscrafts_model.strict_uncertainty_acceptance import StrictUncertaintyPolicy


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record(index: int) -> dict[str, object]:
    incorrect = index % 2 == 1
    return {
        "ast_relation": "identifier",
        "confidence_band": "low" if incorrect else "high",
        "defines": None,
        "event": "READ",
        "ir": {
            "children": [],
            "kind": "symbol",
            "metadata": {},
            "name": f"value_{index}",
            "value": None,
        },
        "metadata": {
            "knowledge_state": "provisional" if incorrect else "known",
            "uncertainty_signals": ["unknown semantic"] if incorrect else [],
            "uyr_rounds": 1,
        },
        "recent_history": [],
        "scope": "module",
        "span": [index + 1, 0, index + 1, 7],
        "surface": f"value_{index}",
        "symbol_id": None,
        "uncertainty_band": "high" if incorrect else "low",
    }


def _build_split(
    root: Path,
    split: str,
    *,
    include_findings: bool = True,
    document_failure: bool = False,
) -> tuple[Path, Path]:
    split_root = root / split
    split_root.mkdir(parents=True)
    admission_path = split_root / "admission.sqlite"
    audit_path = split_root / "audit.sqlite"
    document_id = f"{split}-document"
    records = [_record(index) for index in range(4)]
    canonical_jsonl = "".join(f"{_canonical(record)}\n" for record in records)
    canonical_blob = zlib.compress(canonical_jsonl.encode("utf-8"))
    semantic_blob = zlib.compress(b"semantic-ids")
    canonical_sha = _sha(canonical_jsonl.encode("utf-8"))
    admission_metadata = {
        "schema_version": SOURCE_ADMISSION_SCHEMA,
        "catalog_exhausted": "true",
        "expected_source_document_count": "1",
        "source_catalog_sha256": "a" * 64,
        "source_freeze_manifest_sha256": "b" * 64,
        "semantic_audit_qualification_sha256": "c" * 64,
        "host_code_identity_sha256": "d" * 64,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "frozen_source_split": split,
        "frozen_source_split_identity_sha256": ("e" if split == "validation" else "f")
        * 64,
    }
    admission = sqlite3.connect(admission_path)
    try:
        admission.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents(
                document_id TEXT PRIMARY KEY,
                content_sha256 TEXT NOT NULL,
                language TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                status TEXT NOT NULL,
                reason_code TEXT,
                detail TEXT,
                semantic_spec_sha256 TEXT,
                parser_runtime_identity_sha256 TEXT,
                canonical_sha256 TEXT,
                canonical_jsonl_zlib BLOB,
                semantic_ids_zlib BLOB,
                semantic_record_count INTEGER NOT NULL,
                semantic_token_count INTEGER NOT NULL,
                raw_token_count INTEGER NOT NULL
            );
            """
        )
        admission.executemany(
            "INSERT INTO metadata VALUES(?,?)", sorted(admission_metadata.items())
        )
        admission.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                document_id,
                "1" * 64,
                "python",
                f"{split}.py",
                "admitted",
                None,
                None,
                "2" * 64,
                PARSER_RUNTIME_IDENTITY_SHA256,
                canonical_sha,
                canonical_blob,
                semantic_blob,
                len(records),
                100,
                20,
            ),
        )
        admission.commit()
        admission_identity = ResumableTrainingSemanticAudit._admission_identity(
            admission, admission_metadata
        )
    finally:
        admission.close()

    deterministic_sha = "3" * 64
    raw_ast_sha = "4" * 64
    finding_indices = (1, 3) if include_findings and not document_failure else ()
    finding_count = 1 if document_failure else len(finding_indices)
    failed_record_count = len(records) if document_failure else len(finding_indices)
    summary = {
        "document_id": document_id,
        "source_status": "admitted",
        "language": "python",
        "canonical_sha256": canonical_sha,
        "canonical_payload_sha256": _sha(canonical_blob),
        "semantic_payload_sha256": _sha(semantic_blob),
        "semantic_record_count": len(records),
        "deterministic_evidence_sha256": deterministic_sha,
        "raw_ast_evidence_sha256": raw_ast_sha,
        "failed_record_count": failed_record_count,
        "finding_count": finding_count,
    }
    audit_metadata = {
        "schema_version": AUDIT_SCHEMA,
        "source_admission_identity_sha256": admission_identity,
        "source_catalog_sha256": admission_metadata["source_catalog_sha256"],
        "source_freeze_manifest_sha256": admission_metadata[
            "source_freeze_manifest_sha256"
        ],
        "semantic_audit_qualification_sha256": admission_metadata[
            "semantic_audit_qualification_sha256"
        ],
        "admission_host_code_identity_sha256": admission_metadata[
            "host_code_identity_sha256"
        ],
        "audit_host_code_identity_sha256": "5" * 64,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "expected_source_document_count": "1",
        "source_split": split,
        "frozen_source_split_identity_sha256": admission_metadata[
            "frozen_source_split_identity_sha256"
        ],
        "audit_complete": "true",
    }
    audit = sqlite3.connect(audit_path)
    try:
        audit.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents(
                document_id TEXT PRIMARY KEY,
                source_status TEXT NOT NULL,
                language TEXT NOT NULL,
                canonical_sha256 TEXT,
                canonical_payload_sha256 TEXT,
                semantic_payload_sha256 TEXT,
                semantic_record_count INTEGER NOT NULL,
                deterministic_evidence_sha256 TEXT,
                raw_ast_evidence_sha256 TEXT,
                failed_record_count INTEGER NOT NULL,
                finding_count INTEGER NOT NULL,
                receipt_sha256 TEXT NOT NULL
            );
            CREATE TABLE findings(
                document_id TEXT NOT NULL,
                record_index INTEGER NOT NULL,
                review_id TEXT NOT NULL,
                gate TEXT NOT NULL,
                codes_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                PRIMARY KEY(document_id,record_index,gate)
            );
            """
        )
        audit.executemany(
            "INSERT INTO metadata VALUES(?,?)", sorted(audit_metadata.items())
        )
        audit.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                document_id,
                "admitted",
                "python",
                canonical_sha,
                _sha(canonical_blob),
                _sha(semantic_blob),
                len(records),
                deterministic_sha,
                raw_ast_sha,
                failed_record_count,
                finding_count,
                _sha(_canonical(summary).encode("utf-8")),
            ),
        )
        if document_failure:
            codes = ["raw_ast_witness_failure"]
            evidence = {
                "detail": "fixture witness unavailable",
                "document_id": document_id,
                "error_type": "OSError",
                "unwitnessed_record_count": len(records),
            }
            audit.execute(
                "INSERT INTO findings VALUES(?,?,?,?,?,?)",
                (
                    document_id,
                    -1,
                    f"{document_id}:raw_ast",
                    "raw_ast",
                    _canonical(codes),
                    _canonical(evidence),
                ),
            )
        for record_index in finding_indices:
            codes = ["ast_relation_span_mismatch"]
            evidence = {"finding_codes": codes, "passed": False}
            audit.execute(
                "INSERT INTO findings VALUES(?,?,?,?,?,?)",
                (
                    document_id,
                    record_index,
                    f"{document_id}:{record_index}",
                    "raw_ast",
                    _canonical(codes),
                    _canonical(evidence),
                ),
            )
        audit.commit()
    finally:
        audit.close()
    return admission_path, audit_path


def test_builds_balanced_disjoint_dataset_from_frozen_audits(tmp_path: Path) -> None:
    validation_admission, validation_audit = _build_split(tmp_path, "validation")
    test_admission, test_audit = _build_split(tmp_path, "test")
    output = tmp_path / "output"
    result = AuditedSemanticCalibrationDatasetBuilder(
        samples_per_language=2,
        expected_languages=("python",),
    ).materialize(
        validation_admission,
        validation_audit,
        test_admission,
        test_audit,
        output,
    )

    assert result.calibration_ready is True
    assert result.calibration_count == 2
    assert result.heldout_count == 2
    calibration = [
        json.loads(line)
        for line in (output / "calibration_observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    heldout = [
        json.loads(line)
        for line in (output / "heldout_observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["sample_id"] for row in calibration}.isdisjoint(
        row["sample_id"] for row in heldout
    )
    assert {row["source_split"] for row in calibration} == {"validation"}
    assert {row["source_split"] for row in heldout} == {"test"}
    assert {row["decision"] for row in calibration} == {"keep", "reject"}
    assert {row["decision"] for row in heldout} == {"keep", "reject"}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["calibration_blockers"] == []
    assert manifest["label_counts"]["per_language"]["python"] == {
        "calibration": {"keep": 1, "reject": 1},
        "heldout": {"keep": 1, "reject": 1},
    }
    assert manifest["training_ready"] is False


def test_built_dataset_is_accepted_by_existing_promotion_pipeline(
    tmp_path: Path,
) -> None:
    validation_admission, validation_audit = _build_split(tmp_path, "validation")
    test_admission, test_audit = _build_split(tmp_path, "test")
    dataset = tmp_path / "dataset"
    result = AuditedSemanticCalibrationDatasetBuilder(
        samples_per_language=2,
        expected_languages=("python",),
    ).materialize(
        validation_admission,
        validation_audit,
        test_admission,
        test_audit,
        dataset,
    )
    assert result.calibration_ready is True

    promotion = promote_semantic_uncertainty(
        dataset,
        tmp_path / "promotion",
        policy=SemanticCalibrationPromotionPolicy(
            contained_thresholds=CalibrationThresholds(
                minimum_heldout_samples=2,
                minimum_languages=1,
                minimum_samples_per_language=2,
                maximum_calibrated_ece=1.0,
                maximum_calibrated_brier=1.0,
                maximum_brier_degradation=1.0,
            ),
            strict_policy=StrictUncertaintyPolicy(
                minimum_global_selective_coverage_at_025=0.0,
                maximum_global_selective_error_rate_at_025=1.0,
                minimum_per_language_selective_coverage_at_025=0.0,
                maximum_per_language_selective_error_rate_at_025=1.0,
            ),
        ),
    )

    assert promotion.promoted is True
    assert promotion.failure_codes == ()


def test_missing_reject_class_is_explicit_blocker(tmp_path: Path) -> None:
    validation_admission, validation_audit = _build_split(
        tmp_path, "validation", include_findings=False
    )
    test_admission, test_audit = _build_split(tmp_path, "test", include_findings=False)
    output = tmp_path / "output"
    result = AuditedSemanticCalibrationDatasetBuilder(
        samples_per_language=2,
        expected_languages=("python",),
    ).materialize(
        validation_admission,
        validation_audit,
        test_admission,
        test_audit,
        output,
    )

    assert result.calibration_ready is False
    assert result.calibration_blockers == (
        "language_split_missing_reject:python:calibration",
        "language_split_missing_reject:python:heldout",
    )


def test_payload_tampering_after_audit_is_rejected(tmp_path: Path) -> None:
    validation_admission, validation_audit = _build_split(tmp_path, "validation")
    test_admission, test_audit = _build_split(tmp_path, "test")
    connection = sqlite3.connect(validation_admission)
    try:
        connection.execute(
            "UPDATE documents SET canonical_jsonl_zlib=?",
            (zlib.compress(b"{}\n"),),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="audited admission payload lineage mismatch"):
        AuditedSemanticCalibrationDatasetBuilder(
            samples_per_language=2,
            expected_languages=("python",),
        ).materialize(
            validation_admission,
            validation_audit,
            test_admission,
            test_audit,
            tmp_path / "output",
        )


def test_document_level_audit_failure_is_excluded_not_labeled_reject(
    tmp_path: Path,
) -> None:
    validation_admission, validation_audit = _build_split(
        tmp_path,
        "validation",
        document_failure=True,
    )
    test_admission, test_audit = _build_split(
        tmp_path,
        "test",
        document_failure=True,
    )
    output = tmp_path / "output"
    result = AuditedSemanticCalibrationDatasetBuilder(
        samples_per_language=2,
        expected_languages=("python",),
    ).materialize(
        validation_admission,
        validation_audit,
        test_admission,
        test_audit,
        output,
    )

    assert result.calibration_ready is False
    assert result.calibration_count == 0
    assert result.heldout_count == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_validation_unlabeled_document_failure_count"] == 1
    assert manifest["source_test_unlabeled_document_failure_count"] == 1
    assert manifest["label_counts"]["global"] == {
        "calibration": {"keep": 0, "reject": 0},
        "heldout": {"keep": 0, "reject": 0},
    }


def test_cli_is_importable_and_exposes_frozen_defaults() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_v10_audited_semantic_calibration_v3_runtime_repair.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--validation-admission" in completed.stdout
    assert "--test-audit" in completed.stdout
    assert "--samples-per-language" in completed.stdout

    promotion = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "promote_v10_audited_semantic_uncertainty_v3_runtime_repair.py"
    )
    completed = subprocess.run(
        [sys.executable, str(promotion), "--print-policy"],
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    policy = json.loads(completed.stdout)
    assert policy["contained_thresholds"]["minimum_heldout_samples"] == 140
    assert policy["contained_thresholds"]["minimum_languages"] == 14
    assert policy["strict_policy"][
        "maximum_global_selective_error_rate_at_025"
    ] == pytest.approx(0.1)
