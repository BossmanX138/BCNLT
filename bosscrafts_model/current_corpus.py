"""Deterministic current-spec corpus admission for the frozen native 65K lineage."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable, cast
from urllib.parse import quote

from .bcnlt_source_pipeline import (
    BCNLTPreprocessingError,
    BCNLTSourcePipeline,
    PARSER_RUNTIME_IDENTITY_SHA256,
)


RECORD_SCHEMA = "bosscrafts-current-corpus-record-v2"
MANIFEST_SCHEMA = "bosscrafts-current-corpus-manifest-v2"
APPROVED_LICENSES = frozenset(
    (
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC0-1.0",
        "ISC",
        "MIT",
        "Unlicense",
    )
)
MAX_SOURCE_BYTES = 200 * 1024


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceCandidate:
    document_id: str
    language: str
    source: str
    relative_path: str
    content_sha256: str
    license_spdx: str
    attribution: str
    ownership: str
    source_provider: str
    source_name: str
    source_revision: str
    repository_identity: str
    upstream_url: str
    dialect: str | None = None


@dataclass(frozen=True)
class CurrentCorpusResult:
    output_root: Path
    admitted_count: int
    rejected_count: int
    records_sha256: str
    rejections_sha256: str
    manifest_sha256: str
    tokenizer_sha256: str
    binding_sha256: str
    semantic_spec_sha256: dict[str, str]
    parser_runtime_identity_sha256: str
    training_ready: bool


class SQLiteSourceCatalog:
    """Read the preserved source-document catalog without mutating it."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        if not self.database_path.is_file():
            raise ValueError("source catalog database does not exist")
        if expected_sha256 is not None:
            if (
                type(expected_sha256) is not str
                or len(expected_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha256)
            ):
                raise ValueError("expected source catalog SHA-256 is invalid")
            if _sha256_file(self.database_path) != expected_sha256:
                raise ValueError("source catalog SHA-256 mismatch")

    @staticmethod
    def _read_only_uri(database_path: Path) -> str:
        """Return a read-only SQLite URI, including valid Windows UNC paths."""
        posix_path = database_path.as_posix()
        if posix_path.startswith("//"):
            return f"file://{quote(posix_path, safe='/:')}?mode=ro"
        return f"{database_path.as_uri()}?mode=ro"

    def iter_candidates(self, *, limit_per_language: int | None = None):
        if limit_per_language is not None and (
            type(limit_per_language) is not int or limit_per_language <= 0
        ):
            raise ValueError("limit_per_language must be a positive integer or None")
        uri = self._read_only_uri(self.database_path)
        connection = sqlite3.connect(uri, uri=True)
        try:
            if limit_per_language is None:
                rows = connection.execute(
                    """SELECT document_id, canonical_language, content, relative_path,
                              content_sha256, license_spdx, attribution, ownership,
                              source_provider, source_name, source_revision,
                              repository_identity, upstream_url, dialect
                       FROM documents
                       ORDER BY canonical_language, document_id"""
                )
            else:
                rows = connection.execute(
                    """WITH ranked AS (
                           SELECT document_id, canonical_language, content, relative_path,
                                  content_sha256, license_spdx, attribution, ownership,
                                  source_provider, source_name, source_revision,
                                  repository_identity, upstream_url, dialect,
                                  ROW_NUMBER() OVER (
                                      PARTITION BY canonical_language ORDER BY document_id
                                  ) AS language_rank
                           FROM documents
                       )
                       SELECT document_id, canonical_language, content, relative_path,
                              content_sha256, license_spdx, attribution, ownership,
                              source_provider, source_name, source_revision,
                              repository_identity, upstream_url, dialect
                       FROM ranked
                       WHERE language_rank <= ?
                       ORDER BY canonical_language, document_id""",
                    (limit_per_language,),
                )
            for row in rows:
                yield SourceCandidate(
                    document_id=str(row[0]),
                    language=str(row[1]),
                    source=str(row[2]),
                    relative_path=str(row[3]),
                    content_sha256=str(row[4]),
                    license_spdx=str(row[5]),
                    attribution=str(row[6]),
                    ownership=str(row[7]),
                    source_provider=str(row[8]),
                    source_name=str(row[9]),
                    source_revision=str(row[10]),
                    repository_identity=str(row[11]),
                    upstream_url=str(row[12]),
                    dialect=str(row[13]) if row[13] else None,
                )
        finally:
            connection.close()


class CurrentCorpusAdmission:
    """Admit source only after live current-spec parse and exact 65K encoding."""

    def __init__(
        self,
        pipeline: BCNLTSourcePipeline,
        *,
        source_catalog_sha256: str | None = None,
    ) -> None:
        if type(pipeline) is not BCNLTSourcePipeline:
            raise TypeError("pipeline must be a BCNLTSourcePipeline")
        if source_catalog_sha256 is not None and (
            type(source_catalog_sha256) is not str
            or len(source_catalog_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_catalog_sha256
            )
        ):
            raise ValueError("source_catalog_sha256 is invalid")
        self._pipeline = pipeline
        self._source_catalog_sha256 = source_catalog_sha256

    def _validate_candidate(self, candidate: SourceCandidate) -> str | None:
        if type(candidate) is not SourceCandidate:
            raise TypeError("candidates must be SourceCandidate values")
        required_strings = (
            candidate.document_id,
            candidate.language,
            candidate.relative_path,
            candidate.license_spdx,
            candidate.attribution,
            candidate.ownership,
            candidate.source_provider,
            candidate.source_name,
            candidate.source_revision,
            candidate.repository_identity,
            candidate.upstream_url,
        )
        if any(type(value) is not str or not value for value in required_strings):
            return "invalid_candidate_metadata"
        if type(candidate.source) is not str or not candidate.source:
            return "empty_source"
        if candidate.language not in self._pipeline.supported_languages:
            return "unsupported_language"
        if candidate.license_spdx not in APPROVED_LICENSES:
            return "license_not_approved"
        try:
            source_bytes = candidate.source.encode("utf-8")
        except UnicodeEncodeError:
            return "invalid_source_unicode"
        if len(source_bytes) > MAX_SOURCE_BYTES:
            return "source_too_large"
        if "\x00" in candidate.source:
            return "nul_byte_in_source"
        if candidate.content_sha256 != _sha256_bytes(source_bytes):
            return "content_hash_mismatch"
        return None

    def materialize(
        self,
        candidates: Iterable[SourceCandidate],
        output_root: str | Path,
    ) -> CurrentCorpusResult:
        indexed = list(enumerate(candidates))
        ordered = [item for _, item in sorted(indexed, key=lambda pair: (pair[1].document_id, pair[0]))]
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        rejections: list[dict[str, object]] = []
        seen_document_ids: set[str] = set()
        seen_content_hashes: set[str] = set()

        for candidate in ordered:
            reason = self._validate_candidate(candidate)
            if reason is None and candidate.document_id in seen_document_ids:
                reason = "duplicate_document_id"
            if reason is None and candidate.content_sha256 in seen_content_hashes:
                reason = "duplicate_content_sha256"
            if reason is not None:
                rejections.append(self._rejection(candidate, reason))
                continue
            seen_document_ids.add(candidate.document_id)
            seen_content_hashes.add(candidate.content_sha256)
            try:
                semantic = self._pipeline.prepare_semantic(
                    candidate.source,
                    language=candidate.language,
                    path=candidate.relative_path,
                    dialect=candidate.dialect,
                    allow_raw_fallback=False,
                )
            except (BCNLTPreprocessingError, ValueError) as error:
                detail = " ".join(str(error).split())[:512]
                rejections.append(
                    self._rejection(candidate, "semantic_parse_failure", detail)
                )
                continue
            raw = self._pipeline.prepare_raw(
                candidate.source,
                language=candidate.language,
                path=candidate.relative_path,
            )
            records.append(
                {
                    "schema_version": RECORD_SCHEMA,
                    "document_id": candidate.document_id,
                    "language": candidate.language,
                    "dialect": candidate.dialect,
                    "relative_path": candidate.relative_path,
                    "source": candidate.source,
                    "source_sha256": candidate.content_sha256,
                    "license_spdx": candidate.license_spdx,
                    "attribution": candidate.attribution,
                    "ownership": candidate.ownership,
                    "source_provider": candidate.source_provider,
                    "source_name": candidate.source_name,
                    "source_revision": candidate.source_revision,
                    "repository_identity": candidate.repository_identity,
                    "upstream_url": candidate.upstream_url,
                    "binding_sha256": semantic.binding_sha256,
                    "tokenizer_sha256": semantic.tokenizer_sha256,
                    "semantic_spec_sha256": semantic.semantic_spec_sha256,
                    "parser_runtime_identity_sha256": (
                        semantic.parser_runtime_identity_sha256
                    ),
                    "canonical_sha256": semantic.canonical_sha256,
                    "canonical_jsonl": semantic.canonical_jsonl,
                    "semantic_record_count": semantic.record_count,
                    "semantic_token_ids": list(semantic.token_ids),
                    "raw_token_ids": list(raw.token_ids),
                }
            )

        records_text = "".join(f"{_canonical_json(record)}\n" for record in records)
        rejections_text = "".join(f"{_canonical_json(record)}\n" for record in rejections)
        records_sha256 = _sha256_text(records_text)
        rejections_sha256 = _sha256_text(rejections_text)
        language_ids = sorted({str(record["language"]) for record in records})
        spec_hashes = {
            language: next(
                str(record["semantic_spec_sha256"])
                for record in records
                if record["language"] == language
            )
            for language in language_ids
        }
        per_language = {
            language: {
                "admitted": sum(record["language"] == language for record in records),
                "rejected": sum(record["language"] == language for record in rejections),
            }
            for language in sorted(
                {str(record["language"]) for record in (*records, *rejections)}
            )
        }
        binding_sha256 = records[0]["binding_sha256"] if records else ""
        tokenizer_sha256 = records[0]["tokenizer_sha256"] if records else ""
        raw_lengths = [
            len(cast(list[int], record["raw_token_ids"])) for record in records
        ]
        semantic_lengths = [
            len(cast(list[int], record["semantic_token_ids"])) for record in records
        ]
        token_statistics = {
            "raw_token_count": sum(raw_lengths),
            "semantic_token_count": sum(semantic_lengths),
            "maximum_raw_document_tokens": max(raw_lengths, default=0),
            "maximum_semantic_document_tokens": max(semantic_lengths, default=0),
            "raw_documents_over_2048": sum(length > 2048 for length in raw_lengths),
            "semantic_documents_over_2048": sum(
                length > 2048 for length in semantic_lengths
            ),
        }
        unsigned_manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "phase": "current_spec_semantic_admission",
            "admitted_count": len(records),
            "rejected_count": len(rejections),
            "records_file": "admitted.jsonl",
            "records_sha256": records_sha256,
            "rejections_file": "rejected.jsonl",
            "rejections_sha256": rejections_sha256,
            "binding_sha256": binding_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "vocabulary_size": 65_536,
            "semantic_spec_sha256": spec_hashes,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "sidecar_timeout_seconds": format(
                self._pipeline.sidecar_timeout_seconds,
                ".17g",
            ),
            "per_language": per_language,
            "token_statistics": token_statistics,
            "training_ready": False,
            "training_blockers": [
                "task-family examples are not packed",
                "overlength document streams require record-boundary task packing",
                "loss-mask admission is not evaluated",
                "held-out semantic accuracy is not evaluated",
            ],
        }
        if self._source_catalog_sha256 is not None:
            unsigned_manifest["source_catalog_sha256"] = self._source_catalog_sha256
        manifest_sha256 = _sha256_text(_canonical_json(unsigned_manifest))
        manifest = {**unsigned_manifest, "manifest_sha256": manifest_sha256}
        self._replace_text(root / "admitted.jsonl", records_text)
        self._replace_text(root / "rejected.jsonl", rejections_text)
        self._replace_text(root / "manifest.json", f"{_canonical_json(manifest)}\n")
        return CurrentCorpusResult(
            output_root=root,
            admitted_count=len(records),
            rejected_count=len(rejections),
            records_sha256=records_sha256,
            rejections_sha256=rejections_sha256,
            manifest_sha256=manifest_sha256,
            tokenizer_sha256=str(tokenizer_sha256),
            binding_sha256=str(binding_sha256),
            semantic_spec_sha256=spec_hashes,
            parser_runtime_identity_sha256=PARSER_RUNTIME_IDENTITY_SHA256,
            training_ready=False,
        )

    @staticmethod
    def _rejection(
        candidate: SourceCandidate,
        reason_code: str,
        detail: str | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": "bosscrafts-current-corpus-rejection-v1",
            "document_id": candidate.document_id,
            "language": candidate.language,
            "relative_path": candidate.relative_path,
            "content_sha256": candidate.content_sha256,
            "reason_code": reason_code,
            "detail": detail,
        }

    @staticmethod
    def _replace_text(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
