"""Freeze the zero-finding subset of a completed full training semantic audit."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .immutable_artifacts import publish_new_files
from .resumable_training_semantic_audit import (
    AUDIT_SCHEMA,
    ResumableTrainingSemanticAudit,
)

ELIGIBILITY_MANIFEST_SCHEMA = "bosscrafts-training-eligibility-manifest-v1"
ELIGIBLE_DOCUMENT_SCHEMA = "bosscrafts-training-eligible-document-v1"
EXCLUDED_DOCUMENT_SCHEMA = "bosscrafts-training-eligibility-exclusion-v1"


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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class AuditedTrainingEligibilityResult:
    output_root: Path
    eligible_document_count: int
    excluded_document_count: int
    eligible_sha256: str
    excluded_sha256: str
    manifest_sha256: str
    training_ready: bool


class AuditedTrainingEligibilityBuilder:
    def __init__(
        self,
        *,
        expected_audit_host_code_identity_sha256: str,
        expected_source_split: str = "train",
    ) -> None:
        if not _is_sha256(expected_audit_host_code_identity_sha256):
            raise ValueError("expected audit host code identity SHA-256 is invalid")
        if expected_source_split not in {
            "train",
            "validation",
            "test",
            "reserved_heldout",
        }:
            raise ValueError("expected source split is invalid")
        self._expected_audit_host_identity = expected_audit_host_code_identity_sha256
        self._expected_source_split = expected_source_split

    def materialize(
        self,
        audit_database: str | Path,
        admission_database: str | Path,
        output_root: str | Path,
    ) -> AuditedTrainingEligibilityResult:
        audit_path = Path(audit_database).resolve()
        admission_path = Path(admission_database).resolve()
        output = Path(output_root).resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("training eligibility output already exists and is not empty")
        audit = self._ro_connect(audit_path)
        admission = self._ro_connect(admission_path)
        try:
            if str(audit.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
                raise ValueError("training semantic audit database integrity failed")
            if str(admission.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
                raise ValueError("training admission database integrity failed")
            metadata = dict(audit.execute("SELECT key,value FROM metadata"))
            if (
                metadata.get("schema_version") != AUDIT_SCHEMA
                or metadata.get("audit_complete") != "true"
                or metadata.get("source_split") != self._expected_source_split
                or metadata.get("audit_host_code_identity_sha256")
                != self._expected_audit_host_identity
            ):
                raise ValueError("training semantic audit is incomplete or has wrong lineage")
            expected_count = int(metadata["expected_source_document_count"])
            admission_metadata = dict(
                admission.execute("SELECT key,value FROM metadata")
            )
            admission_identity = ResumableTrainingSemanticAudit._admission_identity(
                admission, admission_metadata
            )
            if admission_identity != metadata.get(
                "source_admission_identity_sha256"
            ):
                raise ValueError("training admission changed after semantic audit")
            rows = audit.execute("SELECT * FROM documents ORDER BY document_id")
            columns = [str(column[0]) for column in rows.description]
            audit_rows = [dict(zip(columns, row, strict=True)) for row in rows]
            if len(audit_rows) != expected_count:
                raise ValueError("training semantic audit document inventory is incomplete")
            admission_rows = {
                str(row[0]): row
                for row in admission.execute(
                    """SELECT document_id,status,canonical_sha256,
                              semantic_record_count,canonical_jsonl_zlib,
                              semantic_ids_zlib
                       FROM documents"""
                )
            }
            if set(admission_rows) != {
                str(row["document_id"]) for row in audit_rows
            }:
                raise ValueError("semantic audit and admission inventories differ")

            eligible: list[dict[str, object]] = []
            excluded: list[dict[str, object]] = []
            for row in audit_rows:
                document_id = str(row["document_id"])
                source = admission_rows[document_id]
                if source[1] != row["source_status"]:
                    raise ValueError("semantic audit and admission status differ")
                codes = self._finding_codes(audit, document_id)
                if len(codes) != int(row["finding_count"]):
                    raise ValueError("semantic audit finding inventory mismatch")
                if row["source_status"] == "admitted":
                    self._verify_admitted_receipt(row, source)
                if (
                    row["source_status"] == "admitted"
                    and int(row["semantic_record_count"]) > 0
                    and int(row["failed_record_count"]) == 0
                    and int(row["finding_count"]) == 0
                ):
                    eligible.append(
                        {
                            "schema_version": ELIGIBLE_DOCUMENT_SCHEMA,
                            "document_id": document_id,
                            "language": row["language"],
                            "canonical_sha256": row["canonical_sha256"],
                            "canonical_payload_sha256": row[
                                "canonical_payload_sha256"
                            ],
                            "semantic_payload_sha256": row[
                                "semantic_payload_sha256"
                            ],
                            "semantic_record_count": row["semantic_record_count"],
                            "audit_receipt_sha256": row["receipt_sha256"],
                        }
                    )
                else:
                    reason = (
                        "source_rejected"
                        if row["source_status"] == "rejected"
                        else "semantic_audit_findings"
                    )
                    excluded.append(
                        {
                            "schema_version": EXCLUDED_DOCUMENT_SCHEMA,
                            "document_id": document_id,
                            "reason": reason,
                            "semantic_record_count": int(
                                row["semantic_record_count"]
                            ),
                            "finding_count": int(row["finding_count"]),
                        }
                    )
            audit_identity = self._audit_identity(audit, metadata)
        except (KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("training eligibility input is invalid") from exc
        finally:
            admission.close()
            audit.close()

        eligible_text = "".join(f"{_canonical(row)}\n" for row in eligible)
        excluded_text = "".join(f"{_canonical(row)}\n" for row in excluded)
        eligible_sha = _sha256_bytes(eligible_text.encode("utf-8"))
        excluded_sha = _sha256_bytes(excluded_text.encode("utf-8"))
        unsigned = {
            "schema_version": ELIGIBILITY_MANIFEST_SCHEMA,
            "eligibility_complete": True,
            "source_audit_identity_sha256": audit_identity,
            "source_admission_identity_sha256": metadata[
                "source_admission_identity_sha256"
            ],
            "source_freeze_manifest_sha256": metadata[
                "source_freeze_manifest_sha256"
            ],
            "source_catalog_sha256": metadata["source_catalog_sha256"],
            "semantic_audit_qualification_sha256": metadata[
                "semantic_audit_qualification_sha256"
            ],
            "audit_host_code_identity_sha256": metadata[
                "audit_host_code_identity_sha256"
            ],
            "source_document_count": len(eligible) + len(excluded),
            "source_split": self._expected_source_split,
            "eligible_document_count": len(eligible),
            "excluded_document_count": len(excluded),
            "eligible_file": "eligible.jsonl",
            "eligible_sha256": eligible_sha,
            "excluded_file": "excluded.jsonl",
            "excluded_sha256": excluded_sha,
            "training_ready": False,
            "training_blockers": (
                [
                    "eligible task examples are not packed",
                    "loss masks and sequence bounds have not passed the no-training dry run",
                    "uncertainty calibration is not promoted",
                    "model training remains explicitly held",
                ]
                if self._expected_source_split == "train"
                else [
                    "eligible evaluation examples are not packed",
                    "evaluation task database audit has not passed",
                    "model training remains explicitly held",
                ]
            ),
        }
        manifest_sha = _sha256_bytes(_canonical(unsigned).encode("utf-8"))
        manifest = {**unsigned, "manifest_sha256": manifest_sha}
        publish_new_files(
            (
                (output / "eligible.jsonl", eligible_text.encode("utf-8")),
                (output / "excluded.jsonl", excluded_text.encode("utf-8")),
                (
                    output / "manifest.json",
                    f"{_canonical(manifest)}\n".encode("utf-8"),
                ),
            ),
            artifact_label="training eligibility",
        )
        return AuditedTrainingEligibilityResult(
            output_root=output,
            eligible_document_count=len(eligible),
            excluded_document_count=len(excluded),
            eligible_sha256=eligible_sha,
            excluded_sha256=excluded_sha,
            manifest_sha256=manifest_sha,
            training_ready=False,
        )

    @staticmethod
    def _ro_connect(path: Path) -> sqlite3.Connection:
        if not path.is_file():
            raise ValueError("eligibility input database does not exist")
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
        connection.execute("PRAGMA query_only=ON")
        return connection

    @staticmethod
    def _finding_codes(connection: sqlite3.Connection, document_id: str) -> list[str]:
        codes: list[str] = []
        for (encoded,) in connection.execute(
            "SELECT codes_json FROM findings WHERE document_id=? ORDER BY record_index,gate",
            (document_id,),
        ):
            parsed = json.loads(encoded)
            if type(parsed) is not list or any(type(code) is not str for code in parsed):
                raise ValueError("semantic audit finding codes are invalid")
            codes.extend(parsed)
        return codes

    @staticmethod
    def _verify_admitted_receipt(
        row: dict[str, object], source: tuple[object, ...]
    ) -> None:
        canonical_payload = source[4]
        semantic_payload = source[5]
        if (
            source[2] != row["canonical_sha256"]
            or source[3] != row["semantic_record_count"]
            or type(canonical_payload) is not bytes
            or type(semantic_payload) is not bytes
            or _sha256_bytes(canonical_payload) != row["canonical_payload_sha256"]
            or _sha256_bytes(semantic_payload) != row["semantic_payload_sha256"]
        ):
            raise ValueError("audited admission payload lineage mismatch")
        summary = {
            "document_id": row["document_id"],
            "source_status": "admitted",
            "language": row["language"],
            "canonical_sha256": row["canonical_sha256"],
            "canonical_payload_sha256": row["canonical_payload_sha256"],
            "semantic_payload_sha256": row["semantic_payload_sha256"],
            "semantic_record_count": row["semantic_record_count"],
            "deterministic_evidence_sha256": row[
                "deterministic_evidence_sha256"
            ],
            "raw_ast_evidence_sha256": row["raw_ast_evidence_sha256"],
            "failed_record_count": row["failed_record_count"],
            "finding_count": row["finding_count"],
        }
        if _sha256_bytes(_canonical(summary).encode("utf-8")) != row["receipt_sha256"]:
            raise ValueError("semantic audit receipt SHA-256 mismatch")

    @staticmethod
    def _audit_identity(
        connection: sqlite3.Connection, metadata: dict[str, str]
    ) -> str:
        digest = hashlib.sha256()
        digest.update(f"{_canonical(metadata)}\n".encode("utf-8"))
        for rows in (
            connection.execute("SELECT * FROM documents ORDER BY rowid"),
            connection.execute("SELECT * FROM findings ORDER BY rowid"),
        ):
            for row in rows:
                digest.update(f"{_canonical(list(row))}\n".encode("utf-8"))
        return digest.hexdigest()
