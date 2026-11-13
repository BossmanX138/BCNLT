"""Crash-resumable full-record semantic audit for a completed source admission."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .bcnlt_adapter import MAX_CANONICAL_OUTPUT_BYTES
from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from .compression import decompress_bounded
from .deterministic_semantic_audit import DeterministicSemanticAuditor
from .jsonl import split_jsonl_records
from .raw_ast_witness import RawAstWitnessClient

AUDIT_SCHEMA = "bosscrafts-resumable-training-semantic-audit-v1"
SOURCE_ADMISSION_SCHEMA = "bosscrafts-current-corpus-resumable-v4"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class ResumableTrainingSemanticAuditStatus:
    database_path: Path
    processed_this_run: int
    processed_total: int
    admitted_document_count: int
    source_rejected_document_count: int
    semantic_record_count: int
    failed_record_count: int
    finding_count: int
    audit_complete: bool
    training_ready: bool


class _WitnessClient(Protocol):
    def inspect(
        self,
        source: str,
        *,
        language: str,
        path: str,
        dialect: str | None,
        queries: tuple[dict[str, object], ...],
    ) -> dict[str, object]: ...


class ResumableTrainingSemanticAudit:
    """Audit every admitted record while committing one document at a time."""

    def __init__(
        self,
        admission_database: str | Path,
        source_catalog_database: str | Path,
        source_freeze_root: str | Path,
        audit_database: str | Path,
        *,
        expected_source_catalog_sha256: str,
        expected_source_freeze_manifest_sha256: str,
        expected_qualification_sha256: str,
        expected_admission_host_code_identity_sha256: str,
        audit_host_code_identity_sha256: str,
        source_split: str = "train",
        witness_client: object | None = None,
    ) -> None:
        hashes = {
            "expected_source_catalog_sha256": expected_source_catalog_sha256,
            "expected_source_freeze_manifest_sha256": (
                expected_source_freeze_manifest_sha256
            ),
            "expected_qualification_sha256": expected_qualification_sha256,
            "expected_admission_host_code_identity_sha256": (
                expected_admission_host_code_identity_sha256
            ),
            "audit_host_code_identity_sha256": audit_host_code_identity_sha256,
        }
        for name, value in hashes.items():
            if not _is_sha256(value):
                raise ValueError(f"{name} is invalid")
        if source_split not in {"train", "validation", "test", "reserved_heldout"}:
            raise ValueError("source_split is invalid")
        self.source_split = source_split
        self.admission_database = Path(admission_database).resolve()
        self.source_catalog_database = Path(source_catalog_database).resolve()
        self.source_freeze_root = Path(source_freeze_root).resolve()
        self.database_path = Path(audit_database).resolve()
        if not self.admission_database.is_file():
            raise ValueError("source admission database does not exist")
        if not self.source_catalog_database.is_file():
            raise ValueError("source catalog database does not exist")
        if _sha256_file(self.source_catalog_database) != expected_source_catalog_sha256:
            raise ValueError("source catalog SHA-256 mismatch")
        frozen_source_ids = self._load_frozen_source_ids(
            self.source_freeze_root,
            expected_manifest_sha256=expected_source_freeze_manifest_sha256,
            expected_catalog_sha256=expected_source_catalog_sha256,
            source_split=source_split,
        )
        if witness_client is not None and not callable(
            getattr(witness_client, "inspect", None)
        ):
            raise TypeError("witness_client must provide an inspect method")
        self._witness_client: _WitnessClient = cast(
            _WitnessClient,
            witness_client if witness_client is not None else RawAstWitnessClient(),
        )
        self._deterministic = DeterministicSemanticAuditor()

        admission = self._ro_connect(self.admission_database)
        try:
            if str(admission.execute("PRAGMA quick_check").fetchone()[0]) != "ok":
                raise ValueError("source admission database integrity check failed")
            metadata = dict(admission.execute("SELECT key,value FROM metadata"))
            expected_count = self._validate_admission_metadata(
                metadata,
                expected_source_catalog_sha256=expected_source_catalog_sha256,
                expected_source_freeze_manifest_sha256=(
                    expected_source_freeze_manifest_sha256
                ),
                expected_qualification_sha256=expected_qualification_sha256,
                expected_admission_host_code_identity_sha256=(
                    expected_admission_host_code_identity_sha256
                ),
                source_split=source_split,
            )
            actual_count = int(
                admission.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            )
            admitted_ids = {
                str(row[0])
                for row in admission.execute("SELECT document_id FROM documents")
            }
            if (
                actual_count != expected_count
                or expected_count != len(frozen_source_ids)
                or admitted_ids != frozen_source_ids
            ):
                raise ValueError("source admission document count is incomplete")
            admission_identity = self._admission_identity(admission, metadata)
        except sqlite3.Error as exc:
            raise ValueError("source admission database is invalid") from exc
        finally:
            admission.close()
        self.expected_source_document_count = expected_count
        self._expected_metadata = {
            "schema_version": AUDIT_SCHEMA,
            "source_admission_identity_sha256": admission_identity,
            "source_catalog_sha256": expected_source_catalog_sha256,
            "source_freeze_manifest_sha256": expected_source_freeze_manifest_sha256,
            "semantic_audit_qualification_sha256": expected_qualification_sha256,
            "admission_host_code_identity_sha256": (
                expected_admission_host_code_identity_sha256
            ),
            "audit_host_code_identity_sha256": audit_host_code_identity_sha256,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "expected_source_document_count": str(expected_count),
            "source_split": source_split,
        }
        if source_split != "train":
            self._expected_metadata["frozen_source_split_identity_sha256"] = metadata[
                "frozen_source_split_identity_sha256"
            ]
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _load_frozen_source_ids(
        root: Path,
        *,
        expected_manifest_sha256: str,
        expected_catalog_sha256: str,
        source_split: str,
    ) -> set[str]:
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("source freeze manifest is invalid") from exc
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if (
            manifest.get("schema_version") != "bosscrafts-training-source-freeze-v1"
            or claimed != _sha256_bytes(_canonical(unsigned).encode("utf-8"))
            or claimed != expected_manifest_sha256
            or manifest.get("source_catalog_sha256") != expected_catalog_sha256
            or manifest.get("source_freeze_complete") is not True
            or manifest.get("training_ready") is not False
        ):
            raise ValueError("source freeze manifest lineage mismatch")
        try:
            documents = (root / str(manifest["documents_file"])).read_bytes()
        except OSError as exc:
            raise ValueError("source freeze document inventory cannot be read") from exc
        if _sha256_bytes(documents) != manifest.get("documents_sha256"):
            raise ValueError("source freeze document inventory SHA-256 mismatch")
        all_ids: set[str] = set()
        selected_ids: set[str] = set()
        training_count = 0
        for line in split_jsonl_records(documents.decode("utf-8")):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("source freeze document inventory is invalid") from exc
            document_id = row.get("document_id") if type(row) is dict else None
            split = row.get("split") if type(row) is dict else None
            if (
                type(row) is not dict
                or line != _canonical(row)
                or row.get("schema_version")
                != "bosscrafts-training-source-document-v1"
                or type(document_id) is not str
                or not document_id
                or document_id in all_ids
                or split not in {"train", "validation", "test", "reserved_heldout"}
            ):
                raise ValueError("source freeze document inventory is invalid")
            all_ids.add(document_id)
            if split == "train":
                training_count += 1
            if split == source_split:
                selected_ids.add(document_id)
        if (
            len(all_ids) != manifest.get("source_document_count")
            or training_count != manifest.get("training_document_count")
        ):
            raise ValueError("source freeze document inventory count mismatch")
        if not selected_ids:
            raise ValueError("source freeze selected split is empty")
        return selected_ids

    @staticmethod
    def _ro_connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
        connection.execute("PRAGMA query_only=ON")
        return connection

    @staticmethod
    def _validate_admission_metadata(
        metadata: dict[str, str],
        *,
        expected_source_catalog_sha256: str,
        expected_source_freeze_manifest_sha256: str,
        expected_qualification_sha256: str,
        expected_admission_host_code_identity_sha256: str,
        source_split: str,
    ) -> int:
        if metadata.get("schema_version") != SOURCE_ADMISSION_SCHEMA:
            raise ValueError("source admission schema mismatch")
        if metadata.get("catalog_exhausted") != "true":
            raise ValueError("source admission is not complete")
        lineage = {
            "source_catalog_sha256": expected_source_catalog_sha256,
            "source_freeze_manifest_sha256": expected_source_freeze_manifest_sha256,
            "semantic_audit_qualification_sha256": expected_qualification_sha256,
            "host_code_identity_sha256": expected_admission_host_code_identity_sha256,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        }
        if any(metadata.get(key) != value for key, value in lineage.items()):
            raise ValueError("source admission lineage mismatch")
        if source_split != "train" and (
            metadata.get("frozen_source_split") != source_split
            or not _is_sha256(metadata.get("frozen_source_split_identity_sha256"))
        ):
            raise ValueError("source admission frozen split lineage mismatch")
        try:
            expected_count = int(metadata["expected_source_document_count"])
        except (KeyError, ValueError) as exc:
            raise ValueError("source admission expected document count is invalid") from exc
        if expected_count < 1:
            raise ValueError("source admission expected document count is invalid")
        return expected_count

    @staticmethod
    def _admission_identity(
        connection: sqlite3.Connection, metadata: dict[str, str]
    ) -> str:
        digest = hashlib.sha256()
        stable_metadata = {
            key: value for key, value in metadata.items() if key != "catalog_exhausted"
        }
        digest.update(f"{_canonical(stable_metadata)}\n".encode())
        for row in connection.execute(
            """SELECT document_id,content_sha256,language,relative_path,status,
                      reason_code,detail,semantic_spec_sha256,
                      parser_runtime_identity_sha256,canonical_sha256,
                      semantic_record_count,semantic_token_count,raw_token_count
               FROM documents ORDER BY document_id"""
        ):
            digest.update(f"{_canonical(list(row))}\n".encode())
        return digest.hexdigest()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents(
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
                CREATE TABLE IF NOT EXISTS findings(
                    document_id TEXT NOT NULL,
                    record_index INTEGER NOT NULL,
                    review_id TEXT NOT NULL,
                    gate TEXT NOT NULL,
                    codes_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    PRIMARY KEY(document_id,record_index,gate),
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                """
            )
            existing = dict(connection.execute("SELECT key,value FROM metadata"))
            if not existing:
                connection.executemany(
                    "INSERT INTO metadata(key,value) VALUES(?,?)",
                    sorted(self._expected_metadata.items()),
                )
                connection.execute(
                    "INSERT INTO metadata(key,value) VALUES('audit_complete','false')"
                )
                connection.commit()
            else:
                actual = {key: existing.get(key) for key in self._expected_metadata}
                if actual != self._expected_metadata:
                    raise ValueError("resumable semantic audit lineage mismatch")
        finally:
            connection.close()

    def process(
        self, *, max_documents: int | None = None
    ) -> ResumableTrainingSemanticAuditStatus:
        if max_documents is not None and (
            type(max_documents) is not int or max_documents < 1
        ):
            raise ValueError("max_documents must be a positive integer or None")
        source = self._ro_connect(self.admission_database)
        catalog = self._ro_connect(self.source_catalog_database)
        output = self._connect()
        processed = 0
        try:
            rows = source.execute(
                "SELECT * FROM documents ORDER BY document_id"
            )
            columns = [str(item[0]) for item in rows.description]
            for raw in rows:
                row = dict(zip(columns, raw, strict=True))
                document_id = str(row["document_id"])
                existing = output.execute(
                    """SELECT source_status,canonical_payload_sha256,
                              semantic_payload_sha256
                       FROM documents WHERE document_id=?""",
                    (document_id,),
                ).fetchone()
                if existing is not None:
                    canonical_payload = row.get("canonical_jsonl_zlib")
                    semantic_payload = row.get("semantic_ids_zlib")
                    current_canonical_sha = (
                        _sha256_bytes(canonical_payload)
                        if type(canonical_payload) is bytes
                        else None
                    )
                    current_semantic_sha = (
                        _sha256_bytes(semantic_payload)
                        if type(semantic_payload) is bytes
                        else None
                    )
                    if (
                        existing[0] != row["status"]
                        or existing[1] != current_canonical_sha
                        or existing[2] != current_semantic_sha
                    ):
                        raise ValueError("source payload changed after audit")
                    continue
                if max_documents is not None and processed >= max_documents:
                    break
                catalog_row = catalog.execute(
                    """SELECT canonical_language,content,relative_path,
                              content_sha256,dialect
                       FROM documents WHERE document_id=?""",
                    (document_id,),
                ).fetchone()
                if catalog_row is None:
                    raise ValueError("source admission document is absent from catalog")
                self._audit_document(output, row, catalog_row)
                output.commit()
                processed += 1
            processed_total = int(
                output.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            )
            if processed_total == self.expected_source_document_count:
                output.execute(
                    "UPDATE metadata SET value='true' WHERE key='audit_complete'"
                )
                output.commit()
        finally:
            output.close()
            catalog.close()
            source.close()
        return self.status(processed_this_run=processed)

    def _audit_document(
        self,
        output: sqlite3.Connection,
        admission: dict[str, Any],
        catalog_row: tuple[object, ...],
    ) -> None:
        document_id = str(admission["document_id"])
        language, source, relative_path, content_sha256, dialect = catalog_row
        if (
            language != admission["language"]
            or relative_path != admission["relative_path"]
            or content_sha256 != admission["content_sha256"]
            or _sha256_bytes(str(source).encode("utf-8")) != content_sha256
        ):
            raise ValueError("source admission and catalog document mismatch")
        if admission["status"] == "rejected":
            summary = {
                "document_id": document_id,
                "source_status": "rejected",
                "language": language,
                "reason_code": admission["reason_code"],
                "detail": admission["detail"],
                "semantic_record_count": 0,
                "failed_record_count": 0,
                "finding_count": 0,
            }
            output.execute(
                "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    document_id,
                    "rejected",
                    language,
                    None,
                    None,
                    None,
                    0,
                    None,
                    None,
                    0,
                    0,
                    _sha256_bytes(_canonical(summary).encode("utf-8")),
                ),
            )
            return
        if admission["status"] != "admitted":
            raise ValueError("source admission status is invalid")
        canonical_payload = admission["canonical_jsonl_zlib"]
        semantic_payload = admission["semantic_ids_zlib"]
        if type(canonical_payload) is not bytes or type(semantic_payload) is not bytes:
            raise ValueError("admitted source payload is incomplete")
        canonical_payload_sha256 = _sha256_bytes(canonical_payload)
        semantic_payload_sha256 = _sha256_bytes(semantic_payload)
        try:
            canonical_jsonl = decompress_bounded(
                canonical_payload,
                maximum_output_bytes=MAX_CANONICAL_OUTPUT_BYTES,
            ).decode("utf-8")
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError("canonical JSONL payload is invalid") from exc
        if _sha256_bytes(canonical_jsonl.encode("utf-8")) != admission["canonical_sha256"]:
            raise ValueError("canonical JSONL SHA-256 mismatch")
        records: list[dict[str, object]] = []
        for line in split_jsonl_records(canonical_jsonl):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("canonical semantic record JSON is invalid") from exc
            if type(record) is not dict or line != _canonical(record):
                raise ValueError("canonical semantic record is not canonical")
            records.append(record)
        if len(records) != admission["semantic_record_count"]:
            raise ValueError("canonical semantic record count mismatch")

        queries: list[dict[str, object]] = []
        deterministic_lines: list[str] = []
        findings: list[tuple[int, str, str, list[str], dict[str, object]]] = []
        failed_ids: set[str] = set()
        if not records:
            coverage_evidence = {
                "document_id": document_id,
                "semantic_record_count": 0,
            }
            findings.append(
                (
                    -1,
                    f"{document_id}:coverage",
                    "coverage",
                    ["no_semantic_records"],
                    coverage_evidence,
                )
            )
        for index, record in enumerate(records):
            review_id = f"{document_id}:{index}"
            item = {
                "review_id": review_id,
                "canonical_record": record,
                "source_excerpt": self._source_excerpt(
                    str(source), str(relative_path), record
                ),
            }
            result = self._deterministic.audit_item(item)
            codes = [finding.code for finding in result.findings]
            evidence = {
                "review_id": review_id,
                "extracted_surface": result.extracted_surface,
                "finding_codes": codes,
                "passed": result.passed,
            }
            deterministic_lines.append(f"{_canonical(evidence)}\n")
            if codes:
                failed_ids.add(review_id)
                findings.append((index, review_id, "deterministic", codes, evidence))
            queries.append(
                {
                    "query_id": review_id,
                    "span": record.get("span"),
                    "ast_relation": record.get("ast_relation"),
                    "scope": record.get("scope"),
                }
            )
        raw_lines: list[str] = []
        if queries:
            try:
                response = self._witness_client.inspect(
                    str(source),
                    language=str(language),
                    path=str(relative_path),
                    dialect=str(dialect) if dialect else None,
                    queries=tuple(queries),
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                detail = " ".join(str(exc).split())[:512]
                failure_evidence = {
                    "document_id": document_id,
                    "error_type": type(exc).__name__,
                    "detail": detail,
                    "unwitnessed_record_count": len(records),
                }
                raw_lines.append(f"{_canonical(failure_evidence)}\n")
                failed_ids.update(str(query["query_id"]) for query in queries)
                findings.append(
                    (
                        -1,
                        f"{document_id}:raw_ast",
                        "raw_ast",
                        ["raw_ast_witness_failure"],
                        failure_evidence,
                    )
                )
            else:
                witnesses = (
                    response.get("witnesses") if type(response) is dict else None
                )
                if (
                    type(response) is not dict
                    or response.get("source_sha256") != content_sha256
                    or type(witnesses) is not list
                    or len(witnesses) != len(records)
                    or [row.get("query_id") for row in witnesses]
                    != [query["query_id"] for query in queries]
                ):
                    raise ValueError("raw AST witness coverage or provenance mismatch")
                for index, witness in enumerate(witnesses):
                    if type(witness) is not dict:
                        raise ValueError("raw AST witness row is invalid")
                    raw_lines.append(f"{_canonical(witness)}\n")
                    witness_codes: list[str] = []
                    if witness.get("relation_span_match") is not True:
                        witness_codes.append("ast_relation_span_mismatch")
                    if witness.get("scope_match") is False:
                        witness_codes.append("raw_scope_mismatch")
                    if witness_codes:
                        review_id = str(witness["query_id"])
                        failed_ids.add(review_id)
                        findings.append(
                            (index, review_id, "raw_ast", witness_codes, witness)
                        )

        deterministic_sha = _sha256_bytes("".join(deterministic_lines).encode("utf-8"))
        raw_sha = _sha256_bytes("".join(raw_lines).encode("utf-8"))
        finding_count = sum(len(item[3]) for item in findings)
        summary = {
            "document_id": document_id,
            "source_status": "admitted",
            "language": language,
            "canonical_sha256": admission["canonical_sha256"],
            "canonical_payload_sha256": canonical_payload_sha256,
            "semantic_payload_sha256": semantic_payload_sha256,
            "semantic_record_count": len(records),
            "deterministic_evidence_sha256": deterministic_sha,
            "raw_ast_evidence_sha256": raw_sha,
            "failed_record_count": len(failed_ids),
            "finding_count": finding_count,
        }
        output.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                document_id,
                "admitted",
                language,
                admission["canonical_sha256"],
                canonical_payload_sha256,
                semantic_payload_sha256,
                len(records),
                deterministic_sha,
                raw_sha,
                len(failed_ids),
                finding_count,
                _sha256_bytes(_canonical(summary).encode("utf-8")),
            ),
        )
        output.executemany(
            "INSERT INTO findings VALUES(?,?,?,?,?,?)",
            (
                (
                    document_id,
                    index,
                    review_id,
                    gate,
                    _canonical(codes),
                    _canonical(evidence),
                )
                for index, review_id, gate, codes, evidence in findings
            ),
        )

    @staticmethod
    def _source_excerpt(
        source: str, relative_path: str, record: dict[str, object]
    ) -> str:
        span = record.get("span")
        if (
            type(span) is not list
            or len(span) != 4
            or any(type(value) is not int or value < 0 for value in span)
        ):
            raise ValueError("canonical semantic record span is invalid")
        start_line, _, end_line, _ = span
        lines = source.split("\n")
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            raise ValueError("canonical semantic record span is outside source")
        rendered = [f"source: {relative_path}"]
        for line_number in range(start_line, end_line + 1):
            rendered.append(f">{line_number:06d}| {lines[line_number - 1]}")
        return "\n".join(rendered)

    def status(
        self, *, processed_this_run: int = 0
    ) -> ResumableTrainingSemanticAuditStatus:
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT COUNT(*),
                          SUM(source_status='admitted'),
                          SUM(source_status='rejected'),
                          COALESCE(SUM(semantic_record_count),0),
                          COALESCE(SUM(failed_record_count),0),
                          COALESCE(SUM(finding_count),0)
                   FROM documents"""
            ).fetchone()
            complete = connection.execute(
                "SELECT value FROM metadata WHERE key='audit_complete'"
            ).fetchone()
        finally:
            connection.close()
        if row is None or complete is None:
            raise RuntimeError("resumable semantic audit status is incomplete")
        return ResumableTrainingSemanticAuditStatus(
            database_path=self.database_path,
            processed_this_run=processed_this_run,
            processed_total=int(row[0]),
            admitted_document_count=int(row[1] or 0),
            source_rejected_document_count=int(row[2] or 0),
            semantic_record_count=int(row[3]),
            failed_record_count=int(row[4]),
            finding_count=int(row[5]),
            audit_complete=complete[0] == "true",
            training_ready=False,
        )
