"""Crash-resumable current-spec admission backed by committed SQLite rows."""
from __future__ import annotations

from array import array
import atexit
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing
from pathlib import Path
import sqlite3
import sys
from typing import Iterable
import zlib

from .bcnlt_source_pipeline import (
    BCNLTPreprocessingError,
    BCNLTSourcePipeline,
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
    PreparedBCNLTInput,
)
from .current_corpus import CurrentCorpusAdmission, SourceCandidate
from .tokenizer import NativeTokenizerBinding


RESUMABLE_SCHEMA = "bosscrafts-current-corpus-resumable-v4"
_EXECUTOR_PREFETCH_MULTIPLIER = 4


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _u16_bytes(values: tuple[int, ...]) -> bytes:
    if any(type(value) is not int or not 0 <= value <= 65_535 for value in values):
        raise ValueError("semantic token IDs must fit uint16")
    encoded = array("H", values)
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


@dataclass(frozen=True)
class ResumableAdmissionStatus:
    database_path: Path
    processed_this_run: int
    processed_total: int
    admitted_count: int
    rejected_count: int
    semantic_token_count: int
    raw_token_count: int
    catalog_exhausted: bool


@dataclass(frozen=True)
class _PreparedCandidate:
    semantic: PreparedBCNLTInput | None
    raw: PreparedBCNLTInput | None
    rejection_detail: str | None


@dataclass(frozen=True)
class _CompressedPreparedCandidate:
    semantic_spec_sha256: str | None
    parser_runtime_identity_sha256: str | None
    canonical_sha256: str | None
    canonical_jsonl_zlib: bytes | None
    semantic_ids_zlib: bytes | None
    semantic_record_count: int
    semantic_token_count: int
    raw_token_count: int
    rejection_detail: str | None


@dataclass
class _ProcessWorkerState:
    pipeline: BCNLTSourcePipeline | None = None


_PROCESS_WORKER_STATE = _ProcessWorkerState()


def _initialize_process_worker(timeout_seconds: float) -> None:
    binding = NativeTokenizerBinding.load()
    pipeline = BCNLTSourcePipeline(
        binding,
        persistent_sidecars=True,
        timeout_seconds=timeout_seconds,
    )
    _PROCESS_WORKER_STATE.pipeline = pipeline
    atexit.register(pipeline.close)


def _prepare_candidate_in_process(
    candidate: SourceCandidate,
) -> _CompressedPreparedCandidate:
    pipeline = _PROCESS_WORKER_STATE.pipeline
    if pipeline is None:
        raise RuntimeError("BCNLT admission process worker is not initialized")
    try:
        semantic = pipeline.prepare_semantic(
            candidate.source,
            language=candidate.language,
            path=candidate.relative_path,
            dialect=candidate.dialect,
            allow_raw_fallback=False,
        )
    except (BCNLTPreprocessingError, ValueError) as error:
        return _CompressedPreparedCandidate(
            semantic_spec_sha256=None,
            parser_runtime_identity_sha256=None,
            canonical_sha256=None,
            canonical_jsonl_zlib=None,
            semantic_ids_zlib=None,
            semantic_record_count=0,
            semantic_token_count=0,
            raw_token_count=0,
            rejection_detail=" ".join(str(error).split())[:512],
        )
    raw = pipeline.prepare_raw(
        candidate.source,
        language=candidate.language,
        path=candidate.relative_path,
    )
    if semantic.canonical_jsonl is None or semantic.canonical_sha256 is None:
        raise RuntimeError("semantic process result lacks canonical payload")
    return _CompressedPreparedCandidate(
        semantic_spec_sha256=semantic.semantic_spec_sha256,
        parser_runtime_identity_sha256=semantic.parser_runtime_identity_sha256,
        canonical_sha256=semantic.canonical_sha256,
        canonical_jsonl_zlib=zlib.compress(semantic.canonical_jsonl.encode("utf-8"), 6),
        semantic_ids_zlib=zlib.compress(_u16_bytes(semantic.token_ids), 6),
        semantic_record_count=semantic.record_count,
        semantic_token_count=len(semantic.token_ids),
        raw_token_count=len(raw.token_ids),
        rejection_detail=None,
    )


class ResumableCurrentCorpusAdmission:
    def __init__(
        self,
        pipeline: BCNLTSourcePipeline,
        database_path: str | Path,
        *,
        source_catalog_sha256: str,
        review_manifest_sha256: str,
        host_code_identity_sha256: str,
        source_freeze_manifest_sha256: str | None = None,
        semantic_audit_qualification_sha256: str | None = None,
        expected_source_document_count: int | None = None,
    ) -> None:
        if type(pipeline) is not BCNLTSourcePipeline:
            raise TypeError("pipeline must be a BCNLTSourcePipeline")
        if not _is_sha256(source_catalog_sha256):
            raise ValueError("source_catalog_sha256 is invalid")
        if not _is_sha256(review_manifest_sha256):
            raise ValueError("review_manifest_sha256 is invalid")
        if not _is_sha256(host_code_identity_sha256):
            raise ValueError("host_code_identity_sha256 is invalid")
        frozen_lineage = (
            source_freeze_manifest_sha256,
            semantic_audit_qualification_sha256,
            expected_source_document_count,
        )
        if any(value is not None for value in frozen_lineage) and not all(
            value is not None for value in frozen_lineage
        ):
            raise ValueError("frozen source lineage must be supplied as a complete set")
        if source_freeze_manifest_sha256 is not None and not _is_sha256(
            source_freeze_manifest_sha256
        ):
            raise ValueError("source_freeze_manifest_sha256 is invalid")
        if semantic_audit_qualification_sha256 is not None and not _is_sha256(
            semantic_audit_qualification_sha256
        ):
            raise ValueError("semantic_audit_qualification_sha256 is invalid")
        if expected_source_document_count is not None and (
            type(expected_source_document_count) is not int
            or expected_source_document_count < 1
        ):
            raise ValueError("expected_source_document_count must be a positive integer")
        self._pipeline = pipeline
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._validator = CurrentCorpusAdmission(pipeline)
        self._expected_metadata = {
            "schema_version": RESUMABLE_SCHEMA,
            "binding_sha256": pipeline._binding.binding_sha256,
            "tokenizer_sha256": pipeline._binding.tokenizer_sha256,
            "vocabulary_size": "65536",
            "source_catalog_sha256": source_catalog_sha256,
            "review_manifest_sha256": review_manifest_sha256,
            "host_code_identity_sha256": host_code_identity_sha256,
            "semantic_spec_sha256": json.dumps(
                CURRENT_RUNTIME_SPEC_SHA256, sort_keys=True, separators=(",", ":")
            ),
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "sidecar_timeout_seconds": format(
                pipeline.sidecar_timeout_seconds,
                ".17g",
            ),
        }
        if source_freeze_manifest_sha256 is not None:
            if (
                semantic_audit_qualification_sha256 is None
                or expected_source_document_count is None
            ):
                raise RuntimeError("validated frozen source lineage became incomplete")
            self._expected_metadata.update(
                {
                    "source_freeze_manifest_sha256": source_freeze_manifest_sha256,
                    "semantic_audit_qualification_sha256": (
                        semantic_audit_qualification_sha256
                    ),
                    "expected_source_document_count": str(
                        expected_source_document_count
                    ),
                }
            )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                       document_id TEXT PRIMARY KEY,
                       content_sha256 TEXT NOT NULL,
                       language TEXT NOT NULL,
                       relative_path TEXT NOT NULL,
                       status TEXT NOT NULL CHECK(status IN ('admitted','rejected')),
                       reason_code TEXT,
                       detail TEXT,
                       semantic_spec_sha256 TEXT,
                       parser_runtime_identity_sha256 TEXT,
                       canonical_sha256 TEXT,
                       canonical_jsonl_zlib BLOB,
                       semantic_ids_zlib BLOB,
                       semantic_record_count INTEGER NOT NULL DEFAULT 0,
                       semantic_token_count INTEGER NOT NULL DEFAULT 0,
                       raw_token_count INTEGER NOT NULL DEFAULT 0
                   )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS documents_content_sha256 ON documents(content_sha256)"
            )
            existing = dict(connection.execute("SELECT key, value FROM metadata"))
            if not existing:
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES(?, ?)",
                    sorted(self._expected_metadata.items()),
                )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('catalog_exhausted', 'false')"
                )
                connection.commit()
            else:
                actual = {key: existing.get(key) for key in self._expected_metadata}
                if actual != self._expected_metadata:
                    raise ValueError("resumable admission lineage mismatch")
        finally:
            connection.close()

    def status(self, *, processed_this_run: int = 0) -> ResumableAdmissionStatus:
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT COUNT(*),
                          SUM(CASE WHEN status='admitted' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END),
                          COALESCE(SUM(semantic_token_count), 0),
                          COALESCE(SUM(raw_token_count), 0)
                   FROM documents"""
            ).fetchone()
            exhausted = connection.execute(
                "SELECT value FROM metadata WHERE key='catalog_exhausted'"
            ).fetchone()
        finally:
            connection.close()
        if row is None or exhausted is None:
            raise RuntimeError("resumable admission status metadata is incomplete")
        return ResumableAdmissionStatus(
            database_path=self.database_path,
            processed_this_run=processed_this_run,
            processed_total=int(row[0]),
            admitted_count=int(row[1] or 0),
            rejected_count=int(row[2] or 0),
            semantic_token_count=int(row[3]),
            raw_token_count=int(row[4]),
            catalog_exhausted=exhausted[0] == "true",
        )

    def process(
        self,
        candidates: Iterable[SourceCandidate],
        *,
        max_documents: int | None = None,
        workers: int = 1,
        worker_mode: str = "threads",
    ) -> ResumableAdmissionStatus:
        if max_documents is not None and (
            type(max_documents) is not int or max_documents < 1
        ):
            raise ValueError("max_documents must be a positive integer or None")
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("workers must be an integer between 1 and 16")
        if worker_mode not in {"threads", "processes"}:
            raise ValueError("worker_mode must be threads or processes")
        connection = self._connect()
        processed = 0
        exhausted = True
        iterator = iter(candidates)
        executor: ThreadPoolExecutor | ProcessPoolExecutor | None
        if worker_mode == "processes":
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_process_worker,
                initargs=(self._pipeline.sidecar_timeout_seconds,),
            )
        else:
            executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        batch_limit = (
            workers
            if executor is None
            else workers * _EXECUTOR_PREFETCH_MULTIPLIER
        )
        try:
            stop_after_batch = False
            while True:
                batch: list[
                    tuple[
                        SourceCandidate,
                        str | None,
                        _PreparedCandidate
                        | _CompressedPreparedCandidate
                        | Future[_PreparedCandidate]
                        | Future[_CompressedPreparedCandidate]
                        | None,
                    ]
                ] = []
                pending_content_hashes: set[str] = set()
                reached_catalog_end = False
                while len(batch) < batch_limit:
                    if max_documents is not None and processed + len(batch) >= max_documents:
                        exhausted = False
                        stop_after_batch = True
                        break
                    try:
                        candidate = next(iterator)
                    except StopIteration:
                        reached_catalog_end = True
                        break
                    if type(candidate) is not SourceCandidate:
                        raise TypeError("candidates must be SourceCandidate values")
                    if connection.execute(
                        "SELECT 1 FROM documents WHERE document_id=?",
                        (candidate.document_id,),
                    ).fetchone():
                        continue
                    reason = self._validator._validate_candidate(candidate)
                    if reason is None and (
                        candidate.content_sha256 in pending_content_hashes
                        or connection.execute(
                            "SELECT 1 FROM documents "
                            "WHERE content_sha256=? AND status='admitted' LIMIT 1",
                            (candidate.content_sha256,),
                        ).fetchone()
                    ):
                        reason = "duplicate_content_sha256"
                    if reason is None:
                        pending_content_hashes.add(candidate.content_sha256)
                    work: (
                        _PreparedCandidate
                        | _CompressedPreparedCandidate
                        | Future[_PreparedCandidate]
                        | Future[_CompressedPreparedCandidate]
                        | None
                    ) = None
                    if reason is None:
                        if executor is None:
                            work = self._prepare_candidate(candidate)
                        elif worker_mode == "processes":
                            work = executor.submit(
                                _prepare_candidate_in_process,
                                candidate,
                            )
                        else:
                            work = executor.submit(self._prepare_candidate, candidate)
                    batch.append((candidate, reason, work))

                for candidate, reason, work in batch:
                    if reason is not None:
                        self._insert_rejection(connection, candidate, reason, None)
                    else:
                        if work is None:
                            raise RuntimeError("admission batch lost its prepared work item")
                        prepared = work.result() if isinstance(work, Future) else work
                        if isinstance(prepared, _CompressedPreparedCandidate):
                            self._commit_compressed(connection, candidate, prepared)
                        else:
                            self._commit_prepared(connection, candidate, prepared)
                    connection.commit()
                    processed += 1

                if reached_catalog_end or stop_after_batch:
                    break
                if not batch:
                    continue
            if exhausted:
                connection.execute(
                    "UPDATE metadata SET value='true' WHERE key='catalog_exhausted'"
                )
                connection.commit()
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            connection.close()
        return self.status(processed_this_run=processed)

    def _prepare_candidate(self, candidate: SourceCandidate) -> _PreparedCandidate:
        try:
            semantic = self._pipeline.prepare_semantic(
                candidate.source,
                language=candidate.language,
                path=candidate.relative_path,
                dialect=candidate.dialect,
                allow_raw_fallback=False,
            )
        except (BCNLTPreprocessingError, ValueError) as error:
            return _PreparedCandidate(
                semantic=None,
                raw=None,
                rejection_detail=" ".join(str(error).split())[:512],
            )
        raw = self._pipeline.prepare_raw(
            candidate.source,
            language=candidate.language,
            path=candidate.relative_path,
        )
        return _PreparedCandidate(
            semantic=semantic,
            raw=raw,
            rejection_detail=None,
        )

    def _commit_prepared(
        self,
        connection: sqlite3.Connection,
        candidate: SourceCandidate,
        prepared: _PreparedCandidate,
    ) -> None:
        if prepared.rejection_detail is not None:
            self._insert_rejection(
                connection,
                candidate,
                "semantic_parse_failure",
                prepared.rejection_detail,
            )
            return
        semantic = prepared.semantic
        raw = prepared.raw
        if semantic is None or raw is None:
            raise RuntimeError("prepared admission result is incomplete")
        if semantic.canonical_jsonl is None or semantic.canonical_sha256 is None:
            raise RuntimeError("semantic admission result lacks canonical payload")
        if semantic.parser_runtime_identity_sha256 != PARSER_RUNTIME_IDENTITY_SHA256:
            raise ValueError("prepared semantic parser runtime identity mismatch")
        connection.execute(
            """INSERT INTO documents(
                   document_id, content_sha256, language, relative_path,
                   status, semantic_spec_sha256, parser_runtime_identity_sha256,
                   canonical_sha256,
                   canonical_jsonl_zlib, semantic_ids_zlib,
                   semantic_record_count, semantic_token_count, raw_token_count
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                candidate.document_id,
                candidate.content_sha256,
                candidate.language,
                candidate.relative_path,
                "admitted",
                semantic.semantic_spec_sha256,
                semantic.parser_runtime_identity_sha256,
                semantic.canonical_sha256,
                zlib.compress(semantic.canonical_jsonl.encode("utf-8"), 6),
                zlib.compress(_u16_bytes(semantic.token_ids), 6),
                semantic.record_count,
                len(semantic.token_ids),
                len(raw.token_ids),
            ),
        )

    def _commit_compressed(
        self,
        connection: sqlite3.Connection,
        candidate: SourceCandidate,
        prepared: _CompressedPreparedCandidate,
    ) -> None:
        if prepared.rejection_detail is not None:
            self._insert_rejection(
                connection,
                candidate,
                "semantic_parse_failure",
                prepared.rejection_detail,
            )
            return
        if (
            prepared.semantic_spec_sha256 != CURRENT_RUNTIME_SPEC_SHA256[candidate.language]
            or prepared.parser_runtime_identity_sha256
            != PARSER_RUNTIME_IDENTITY_SHA256
            or not _is_sha256(prepared.canonical_sha256)
            or type(prepared.canonical_jsonl_zlib) is not bytes
            or type(prepared.semantic_ids_zlib) is not bytes
            or type(prepared.semantic_record_count) is not int
            or prepared.semantic_record_count < 0
            or type(prepared.semantic_token_count) is not int
            or prepared.semantic_token_count < 3
            or type(prepared.raw_token_count) is not int
            or prepared.raw_token_count < 3
        ):
            raise ValueError("compressed admission result failed lineage validation")
        connection.execute(
            """INSERT INTO documents(
                   document_id, content_sha256, language, relative_path,
                   status, semantic_spec_sha256, parser_runtime_identity_sha256,
                   canonical_sha256,
                   canonical_jsonl_zlib, semantic_ids_zlib,
                   semantic_record_count, semantic_token_count, raw_token_count
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                candidate.document_id,
                candidate.content_sha256,
                candidate.language,
                candidate.relative_path,
                "admitted",
                prepared.semantic_spec_sha256,
                prepared.parser_runtime_identity_sha256,
                prepared.canonical_sha256,
                prepared.canonical_jsonl_zlib,
                prepared.semantic_ids_zlib,
                prepared.semantic_record_count,
                prepared.semantic_token_count,
                prepared.raw_token_count,
            ),
        )

    @staticmethod
    def _insert_rejection(
        connection: sqlite3.Connection,
        candidate: SourceCandidate,
        reason_code: str,
        detail: str | None,
    ) -> None:
        connection.execute(
            """INSERT INTO documents(
                   document_id, content_sha256, language, relative_path,
                   status, reason_code, detail
               ) VALUES(?,?,?,?,?,?,?)""",
            (
                candidate.document_id,
                candidate.content_sha256,
                candidate.language,
                candidate.relative_path,
                "rejected",
                reason_code,
                detail,
            ),
        )
