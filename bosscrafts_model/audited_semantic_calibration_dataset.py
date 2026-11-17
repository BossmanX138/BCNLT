"""Build disjoint uncertainty calibration data from qualified semantic audits."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bcnlt_adapter import MAX_CANONICAL_OUTPUT_BYTES
from .bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from .compression import decompress_bounded
from .immutable_artifacts import publish_new_files
from .jsonl import split_jsonl_records
from .resumable_training_semantic_audit import (
    AUDIT_SCHEMA,
    SOURCE_ADMISSION_SCHEMA,
    ResumableTrainingSemanticAudit,
)
from .semantic_calibration_dataset import SemanticCalibrationDatasetBuilder
from .training_eligibility import AuditedTrainingEligibilityBuilder

DATASET_SCHEMA = "bosscrafts-semantic-calibration-dataset-v1"
OBSERVATION_SCHEMA = "bosscrafts-semantic-uncertainty-observation-v1"
SELECTION_POLICY = (
    "frozen_validation_calibration_test_heldout_sha256_balanced_audit_labels_v1"
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class AuditedSemanticCalibrationDatasetResult:
    output_root: Path
    calibration_count: int
    heldout_count: int
    calibration_ready: bool
    calibration_blockers: tuple[str, ...]
    manifest_sha256: str


class AuditedSemanticCalibrationDatasetBuilder:
    """Select uncertainty observations from validation/test semantic audits."""

    def __init__(
        self,
        *,
        samples_per_language: int = 10,
        expected_languages: Iterable[str] | None = None,
    ) -> None:
        if type(samples_per_language) is not int or samples_per_language < 2:
            raise ValueError("samples_per_language must be an integer of at least two")
        languages = tuple(
            sorted(
                CURRENT_RUNTIME_SPEC_SHA256
                if expected_languages is None
                else expected_languages
            )
        )
        if (
            not languages
            or len(languages) != len(set(languages))
            or any(type(language) is not str or not language for language in languages)
        ):
            raise ValueError("expected_languages must be unique non-empty strings")
        unknown = sorted(set(languages) - set(CURRENT_RUNTIME_SPEC_SHA256))
        if unknown:
            raise ValueError(f"unsupported calibration languages: {unknown}")
        self.samples_per_language = samples_per_language
        self.expected_languages = languages

    def materialize(
        self,
        validation_admission: str | Path,
        validation_audit: str | Path,
        test_admission: str | Path,
        test_audit: str | Path,
        output_root: str | Path,
    ) -> AuditedSemanticCalibrationDatasetResult:
        output = Path(output_root).resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                "audited calibration output already exists and is not empty"
            )

        validation_rows, validation_lineage = self._load_split(
            Path(validation_admission).resolve(),
            Path(validation_audit).resolve(),
            source_split="validation",
        )
        test_rows, test_lineage = self._load_split(
            Path(test_admission).resolve(),
            Path(test_audit).resolve(),
            source_split="test",
        )
        calibration, calibration_counts, calibration_blockers = self._select_split(
            validation_rows, split_name="calibration"
        )
        heldout, heldout_counts, heldout_blockers = self._select_split(
            test_rows, split_name="heldout"
        )
        overlap = {row["sample_id"] for row in calibration}.intersection(
            row["sample_id"] for row in heldout
        )
        if overlap:
            raise ValueError("calibration and held-out audit observations overlap")

        blockers = [*calibration_blockers, *heldout_blockers]
        ready = not blockers
        calibration.sort(key=lambda row: str(row["sample_id"]))
        heldout.sort(key=lambda row: str(row["sample_id"]))
        calibration_text = "".join(f"{_canonical(row)}\n" for row in calibration)
        heldout_text = "".join(f"{_canonical(row)}\n" for row in heldout)
        manual_text = ""
        language_counts = {
            language: {
                "calibration": calibration_counts[language]["total"],
                "heldout": heldout_counts[language]["total"],
            }
            for language in self.expected_languages
        }
        label_counts = {
            "global": {
                "calibration": self._global_counts(calibration_counts),
                "heldout": self._global_counts(heldout_counts),
            },
            "per_language": {
                language: {
                    "calibration": {
                        "keep": calibration_counts[language]["keep"],
                        "reject": calibration_counts[language]["reject"],
                    },
                    "heldout": {
                        "keep": heldout_counts[language]["keep"],
                        "reject": heldout_counts[language]["reject"],
                    },
                }
                for language in self.expected_languages
            },
        }
        unsigned = {
            "schema_version": DATASET_SCHEMA,
            "selection_policy": SELECTION_POLICY,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "expected_languages": list(self.expected_languages),
            "samples_per_language": self.samples_per_language,
            "source_validation_admission_identity_sha256": validation_lineage[
                "admission_identity_sha256"
            ],
            "source_validation_audit_identity_sha256": validation_lineage[
                "audit_identity_sha256"
            ],
            "source_test_admission_identity_sha256": test_lineage[
                "admission_identity_sha256"
            ],
            "source_test_audit_identity_sha256": test_lineage["audit_identity_sha256"],
            "source_validation_audit_host_code_identity_sha256": validation_lineage[
                "audit_host_code_identity_sha256"
            ],
            "source_validation_unlabeled_document_failure_count": int(
                validation_lineage["unlabeled_document_failure_count"]
            ),
            "source_test_audit_host_code_identity_sha256": test_lineage[
                "audit_host_code_identity_sha256"
            ],
            "source_test_unlabeled_document_failure_count": int(
                test_lineage["unlabeled_document_failure_count"]
            ),
            "calibration_per_language": self.samples_per_language,
            "heldout_per_language": self.samples_per_language,
            "language_counts": language_counts,
            "label_counts": label_counts,
            "calibration_file": "calibration_observations.jsonl",
            "calibration_count": len(calibration),
            "calibration_sha256": _sha256_text(calibration_text),
            "heldout_file": "heldout_observations.jsonl",
            "heldout_count": len(heldout),
            "heldout_sha256": _sha256_text(heldout_text),
            "manual_decisions_file": "manual_decisions.jsonl",
            "manual_decisions_sha256": _sha256_text(manual_text),
            "calibration_ready": ready,
            "calibration_blockers": blockers,
            "training_ready": False,
            "training_blockers": [
                *blockers,
                "calibration metrics and strict selective-risk gate have not passed",
                "model acceptance categories remain incomplete",
            ],
        }
        manifest_sha256 = _sha256_text(_canonical(unsigned))
        manifest = {**unsigned, "manifest_sha256": manifest_sha256}
        publish_new_files(
            (
                (
                    output / "calibration_observations.jsonl",
                    calibration_text.encode("utf-8"),
                ),
                (
                    output / "heldout_observations.jsonl",
                    heldout_text.encode("utf-8"),
                ),
                (
                    output / "manual_decisions.jsonl",
                    manual_text.encode("utf-8"),
                ),
                (
                    output / "manifest.json",
                    f"{_canonical(manifest)}\n".encode("utf-8"),
                ),
            ),
            artifact_label="audited semantic calibration dataset",
        )
        return AuditedSemanticCalibrationDatasetResult(
            output_root=output,
            calibration_count=len(calibration),
            heldout_count=len(heldout),
            calibration_ready=ready,
            calibration_blockers=tuple(blockers),
            manifest_sha256=manifest_sha256,
        )

    def _load_split(
        self,
        admission_path: Path,
        audit_path: Path,
        *,
        source_split: str,
    ) -> tuple[list[dict[str, object]], dict[str, str]]:
        admission = self._ro_connect(admission_path, "admission")
        audit = self._ro_connect(audit_path, "audit")
        try:
            if admission.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError(f"{source_split} admission database integrity failed")
            if audit.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError(f"{source_split} audit database integrity failed")
            admission_metadata = dict(
                admission.execute("SELECT key,value FROM metadata")
            )
            audit_metadata = dict(audit.execute("SELECT key,value FROM metadata"))
            expected_count = self._validate_metadata(
                admission_metadata,
                audit_metadata,
                source_split=source_split,
            )
            admission_identity = ResumableTrainingSemanticAudit._admission_identity(
                admission, admission_metadata
            )
            if admission_identity != audit_metadata["source_admission_identity_sha256"]:
                raise ValueError(
                    f"{source_split} admission changed after semantic audit"
                )
            audit_identity = AuditedTrainingEligibilityBuilder._audit_identity(
                audit, audit_metadata
            )
            audit_rows = audit.execute("SELECT * FROM documents ORDER BY document_id")
            audit_columns = [str(column[0]) for column in audit_rows.description]
            audited = [dict(zip(audit_columns, row, strict=True)) for row in audit_rows]
            if len(audited) != expected_count:
                raise ValueError(
                    f"{source_split} audit document inventory is incomplete"
                )
            admission_rows = {
                str(row[0]): row
                for row in admission.execute(
                    """SELECT document_id,status,canonical_sha256,
                              semantic_record_count,canonical_jsonl_zlib,
                              semantic_ids_zlib,language,
                              parser_runtime_identity_sha256,reason_code,detail
                       FROM documents ORDER BY document_id"""
                )
            }
            if set(admission_rows) != {str(row["document_id"]) for row in audited}:
                raise ValueError(
                    f"{source_split} audit and admission inventories differ"
                )

            observations: list[dict[str, object]] = []
            unlabeled_document_failures = 0
            for row in audited:
                document_id = str(row["document_id"])
                source = admission_rows[document_id]
                if source[1] != row["source_status"]:
                    raise ValueError("semantic audit and admission status differ")
                finding_rows = list(
                    audit.execute(
                        """SELECT record_index,review_id,gate,codes_json,evidence_json
                           FROM findings WHERE document_id=?
                           ORDER BY record_index,gate""",
                        (document_id,),
                    )
                )
                finding_map, doc_codes, finding_count = self._validate_findings(
                    finding_rows,
                    document_id=document_id,
                    semantic_record_count=int(row["semantic_record_count"]),
                )
                if finding_count != int(row["finding_count"]):
                    raise ValueError("semantic audit finding inventory mismatch")
                expected_failed_records = (
                    int(row["semantic_record_count"]) if doc_codes else len(finding_map)
                )
                if expected_failed_records != int(row["failed_record_count"]):
                    raise ValueError("semantic audit failed-record inventory mismatch")
                if row["source_status"] == "rejected":
                    if int(row["semantic_record_count"]) != 0 or finding_rows:
                        raise ValueError(
                            "source-rejected audit row contains semantic evidence"
                        )
                    rejected_summary = {
                        "document_id": document_id,
                        "source_status": "rejected",
                        "language": source[6],
                        "reason_code": source[8],
                        "detail": source[9],
                        "semantic_record_count": 0,
                        "failed_record_count": 0,
                        "finding_count": 0,
                    }
                    if (
                        _sha256_text(_canonical(rejected_summary))
                        != row["receipt_sha256"]
                    ):
                        raise ValueError("source-rejected audit receipt mismatch")
                    continue
                if (
                    source[6] != row["language"]
                    or source[7] != PARSER_RUNTIME_IDENTITY_SHA256
                ):
                    raise ValueError(
                        "audited admission language or runtime lineage mismatch"
                    )
                AuditedTrainingEligibilityBuilder._verify_admitted_receipt(row, source)
                records = self._canonical_records(source[4], source[2], source[3])
                if doc_codes:
                    # Document-level findings describe unavailable audit evidence,
                    # not record-level semantic incorrectness.  Exclude rather than
                    # manufacture reject labels from an infrastructure failure.
                    unlabeled_document_failures += 1
                    continue
                for record_index, record in enumerate(records):
                    codes = list(finding_map.get(record_index, ()))
                    incorrect = bool(codes)
                    identity = {
                        "source_split": source_split,
                        "document_id": document_id,
                        "record_index": record_index,
                        "canonical_sha256": source[2],
                        "audit_receipt_sha256": row["receipt_sha256"],
                    }
                    sample_id = _sha256_text(_canonical(identity))
                    observations.append(
                        {
                            "schema_version": OBSERVATION_SCHEMA,
                            "sample_id": sample_id,
                            "review_id": f"{document_id}:{record_index}",
                            "language": row["language"],
                            "raw_uncertainty": (
                                SemanticCalibrationDatasetBuilder._raw_uncertainty(
                                    record
                                )
                            ),
                            "incorrect": incorrect,
                            "decision": "reject" if incorrect else "keep",
                            "resolution": "qualified_semantic_audit",
                            "resolver": audit_metadata[
                                "audit_host_code_identity_sha256"
                            ],
                            "reviewer_entry_sha256": [
                                row["deterministic_evidence_sha256"],
                                row["raw_ast_evidence_sha256"],
                            ],
                            "parser_runtime_identity_sha256": (
                                PARSER_RUNTIME_IDENTITY_SHA256
                            ),
                            "source_split": source_split,
                            "document_id": document_id,
                            "record_index": record_index,
                            "audit_receipt_sha256": row["receipt_sha256"],
                            "audit_finding_codes": codes,
                        }
                    )
            return observations, {
                "admission_identity_sha256": admission_identity,
                "audit_identity_sha256": audit_identity,
                "audit_host_code_identity_sha256": audit_metadata[
                    "audit_host_code_identity_sha256"
                ],
                "unlabeled_document_failure_count": str(unlabeled_document_failures),
            }
        except (KeyError, TypeError, sqlite3.Error) as exc:
            raise ValueError(f"{source_split} calibration evidence is invalid") from exc
        finally:
            audit.close()
            admission.close()

    @staticmethod
    def _ro_connect(path: Path, name: str) -> sqlite3.Connection:
        if not path.is_file():
            raise ValueError(f"{name} database does not exist")
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
        connection.execute("PRAGMA query_only=ON")
        return connection

    @staticmethod
    def _validate_metadata(
        admission: dict[str, str],
        audit: dict[str, str],
        *,
        source_split: str,
    ) -> int:
        if (
            admission.get("schema_version") != SOURCE_ADMISSION_SCHEMA
            or admission.get("catalog_exhausted") != "true"
            or admission.get("frozen_source_split") != source_split
            or not _is_sha256(admission.get("frozen_source_split_identity_sha256"))
            or admission.get("parser_runtime_identity_sha256")
            != PARSER_RUNTIME_IDENTITY_SHA256
        ):
            raise ValueError(f"{source_split} admission lineage is incomplete")
        if (
            audit.get("schema_version") != AUDIT_SCHEMA
            or audit.get("audit_complete") != "true"
            or audit.get("source_split") != source_split
            or audit.get("parser_runtime_identity_sha256")
            != PARSER_RUNTIME_IDENTITY_SHA256
            or not _is_sha256(audit.get("audit_host_code_identity_sha256"))
            or audit.get("source_catalog_sha256")
            != admission.get("source_catalog_sha256")
            or audit.get("source_freeze_manifest_sha256")
            != admission.get("source_freeze_manifest_sha256")
            or audit.get("semantic_audit_qualification_sha256")
            != admission.get("semantic_audit_qualification_sha256")
        ):
            raise ValueError(f"{source_split} semantic audit lineage is incomplete")
        try:
            admission_count = int(admission["expected_source_document_count"])
            audit_count = int(audit["expected_source_document_count"])
        except (KeyError, ValueError) as exc:
            raise ValueError("calibration source document count is invalid") from exc
        if admission_count < 1 or audit_count != admission_count:
            raise ValueError("calibration source document count mismatch")
        return admission_count

    @staticmethod
    def _validate_findings(
        rows: list[tuple[Any, ...]],
        *,
        document_id: str,
        semantic_record_count: int,
    ) -> tuple[dict[int, tuple[str, ...]], tuple[str, ...], int]:
        by_record: dict[int, list[str]] = defaultdict(list)
        document_codes: list[str] = []
        total = 0
        for record_index, review_id, gate, codes_json, evidence_json in rows:
            if (
                type(record_index) is not int
                or record_index < -1
                or record_index >= semantic_record_count
                or type(review_id) is not str
                or not review_id
                or type(gate) is not str
                or gate not in {"coverage", "deterministic", "raw_ast"}
                or (record_index >= 0 and review_id != f"{document_id}:{record_index}")
            ):
                raise ValueError("semantic audit finding row is invalid")
            try:
                codes = json.loads(codes_json)
                evidence = json.loads(evidence_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("semantic audit finding codes are invalid") from exc
            if (
                type(codes) is not list
                or not codes
                or any(type(code) is not str or not code for code in codes)
                or codes_json != _canonical(codes)
                or type(evidence) is not dict
                or evidence_json != _canonical(evidence)
            ):
                raise ValueError("semantic audit finding evidence is invalid")
            total += len(codes)
            target = document_codes if record_index == -1 else by_record[record_index]
            target.extend(codes)
        return (
            {index: tuple(sorted(set(codes))) for index, codes in by_record.items()},
            tuple(sorted(set(document_codes))),
            total,
        )

    @staticmethod
    def _canonical_records(
        payload: object,
        expected_sha256: object,
        expected_count: object,
    ) -> list[dict[str, object]]:
        if type(payload) is not bytes or not _is_sha256(expected_sha256):
            raise ValueError("admitted canonical payload is invalid")
        try:
            canonical_jsonl = decompress_bounded(
                payload, maximum_output_bytes=MAX_CANONICAL_OUTPUT_BYTES
            ).decode("utf-8")
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError("admitted canonical payload is invalid") from exc
        if _sha256_text(canonical_jsonl) != expected_sha256:
            raise ValueError("admitted canonical payload SHA-256 mismatch")
        records = []
        for line in split_jsonl_records(canonical_jsonl):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("canonical semantic record JSON is invalid") from exc
            if type(record) is not dict or line != _canonical(record):
                raise ValueError("canonical semantic record is not canonical")
            records.append(record)
        if type(expected_count) is not int or len(records) != expected_count:
            raise ValueError("canonical semantic record count mismatch")
        return records

    def _select_split(
        self,
        rows: list[dict[str, object]],
        *,
        split_name: str,
    ) -> tuple[
        list[dict[str, object]],
        dict[str, dict[str, int]],
        list[str],
    ]:
        selected: list[dict[str, object]] = []
        counts: dict[str, dict[str, int]] = {}
        blockers: list[str] = []
        for language in self.expected_languages:
            candidates = sorted(
                (row for row in rows if row["language"] == language),
                key=lambda row: str(row["sample_id"]),
            )
            keeps = [row for row in candidates if row["incorrect"] is False]
            rejects = [row for row in candidates if row["incorrect"] is True]
            chosen = self._balanced_choice(keeps, rejects)
            selected.extend(chosen)
            keep_count = sum(row["incorrect"] is False for row in chosen)
            reject_count = sum(row["incorrect"] is True for row in chosen)
            counts[language] = {
                "total": len(chosen),
                "keep": keep_count,
                "reject": reject_count,
            }
            if len(chosen) != self.samples_per_language:
                blockers.append(
                    f"language_split_insufficient_samples:{language}:{split_name}"
                )
            if keep_count < 1:
                blockers.append(f"language_split_missing_keep:{language}:{split_name}")
            if reject_count < 1:
                blockers.append(
                    f"language_split_missing_reject:{language}:{split_name}"
                )
        return selected, counts, blockers

    def _balanced_choice(
        self,
        keeps: list[dict[str, object]],
        rejects: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        quota = self.samples_per_language
        reject_target = min(len(rejects), quota // 2)
        keep_target = min(len(keeps), quota - reject_target)
        if keep_target + reject_target < quota:
            reject_target = min(len(rejects), quota - keep_target)
        if keep_target + reject_target < quota:
            keep_target = min(len(keeps), quota - reject_target)
        return sorted(
            [*keeps[:keep_target], *rejects[:reject_target]],
            key=lambda row: str(row["sample_id"]),
        )

    @staticmethod
    def _global_counts(counts: dict[str, dict[str, int]]) -> dict[str, int]:
        return {
            "keep": sum(row["keep"] for row in counts.values()),
            "reject": sum(row["reject"] for row in counts.values()),
        }
