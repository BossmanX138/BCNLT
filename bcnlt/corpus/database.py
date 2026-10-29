from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


DATABASE_SCHEMA_VERSION = "bcnlt-corpus-database-v2"


@dataclass(frozen=True)
class SourceDocumentRecord:
    document_id: str
    source_provider: str
    source_name: str
    source_revision: str
    canonical_language: str
    repository_identity: str
    relative_path: str
    content_sha256: str
    content: str
    license_spdx: str
    ownership: str
    attribution: str
    upstream_url: str


def stable_document_id(
    *,
    provider: str,
    revision: str,
    repository_identity: str,
    relative_path: str,
    content_sha256: str,
) -> str:
    payload = {
        "provider": provider,
        "revision": revision,
        "repository_identity": repository_identity,
        "relative_path": relative_path.replace("\\", "/"),
        "content_sha256": content_sha256,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CorpusDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._validate_existing_schema()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _validate_existing_schema(self) -> None:
        tables = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "documents" not in tables:
            return
        metadata_version = None
        if "metadata" in tables:
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            metadata_version = str(row[0]) if row is not None else None
        if metadata_version == DATABASE_SCHEMA_VERSION:
            return
        document_count = int(
            self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        if document_count:
            self.connection.close()
            raise ValueError("legacy corpus database requires a new v2 output path")
        self.connection.executescript(
            """
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS rejections;
            DROP TABLE IF EXISTS metadata;
            """
        )

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_provider TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                canonical_language TEXT NOT NULL,
                repository_identity TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                content TEXT NOT NULL,
                license_spdx TEXT NOT NULL,
                ownership TEXT NOT NULL,
                attribution TEXT NOT NULL,
                upstream_url TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS documents_content_sha256
                ON documents(content_sha256);
            CREATE TABLE IF NOT EXISTS rejections (
                rejection_id TEXT PRIMARY KEY,
                source_provider TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                canonical_language TEXT NOT NULL,
                repository_identity TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                detail TEXT NOT NULL,
                upstream_url TEXT NOT NULL
            );
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
            (DATABASE_SCHEMA_VERSION,),
        )

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def admit(self, record: SourceDocumentRecord) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO documents (
                document_id, source_provider, source_name, source_revision,
                canonical_language, repository_identity, relative_path, content_sha256, content,
                license_spdx, ownership, attribution, upstream_url
            ) VALUES (
                :document_id, :source_provider, :source_name, :source_revision,
                :canonical_language, :repository_identity, :relative_path, :content_sha256, :content,
                :license_spdx, :ownership, :attribution, :upstream_url
            )
            """,
            asdict(record),
        )
        return cursor.rowcount == 1

    def reject(
        self,
        *,
        source_provider: str,
        source_name: str,
        source_revision: str,
        canonical_language: str,
        repository_identity: str,
        relative_path: str,
        content_sha256: str,
        reason_code: str,
        detail: str,
        upstream_url: str,
    ) -> bool:
        identity = {
            "source_provider": source_provider,
            "source_name": source_name,
            "source_revision": source_revision,
            "canonical_language": canonical_language,
            "repository_identity": repository_identity,
            "relative_path": relative_path,
            "content_sha256": content_sha256,
            "reason_code": reason_code,
        }
        rejection_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO rejections (
                rejection_id, source_provider, source_name, source_revision,
                canonical_language, repository_identity, relative_path, content_sha256,
                reason_code, detail, upstream_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rejection_id,
                source_provider,
                source_name,
                source_revision,
                canonical_language,
                repository_identity,
                relative_path,
                content_sha256,
                reason_code,
                detail,
                upstream_url,
            ),
        )
        return cursor.rowcount == 1

    def documents(self) -> list[SourceDocumentRecord]:
        rows = self.connection.execute(
            "SELECT * FROM documents ORDER BY document_id"
        ).fetchall()
        return [SourceDocumentRecord(**dict(row)) for row in rows]

    def rejections(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM rejections ORDER BY rejection_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def document_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def clear(self) -> None:
        self.connection.execute("DELETE FROM documents")
        self.connection.execute("DELETE FROM rejections")

    def close(self) -> None:
        self.connection.close()
