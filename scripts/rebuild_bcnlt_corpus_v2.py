from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from bcnlt.corpus.database import CorpusDatabase, SourceDocumentRecord
from bcnlt.languages import CANONICAL_LANGUAGE_IDS


@dataclass(frozen=True)
class RebuildReport:
    source_sha256: str
    destination_sha256: str
    document_count: int
    language_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_language_map(source_registry: Path) -> dict[tuple[str, str], str]:
    payload = json.loads(source_registry.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "bcnlt-corpus-sources-v1":
        raise ValueError("unsupported corpus source registry schema")
    candidates: dict[tuple[str, str], set[str]] = {}
    for raw in payload.get("sources", []):
        if not isinstance(raw, dict):
            raise ValueError("corpus source registry entries must be objects")
        try:
            key = (str(raw["name"]), str(raw["revision"]))
            language = str(raw["canonical_language"])
        except KeyError as exc:
            raise ValueError("corpus source registry entry is missing provenance") from exc
        if language not in CANONICAL_LANGUAGE_IDS:
            raise ValueError(f"noncanonical language for source {key[0]}@{key[1]}")
        candidates.setdefault(key, set()).add(language)
    ambiguous = sorted(key for key, values in candidates.items() if len(values) != 1)
    if ambiguous:
        rendered = ", ".join(f"{name}@{revision}" for name, revision in ambiguous)
        raise ValueError(f"ambiguous canonical language mapping: {rendered}")
    return {key: next(iter(values)) for key, values in candidates.items()}


def _read_legacy_rows(source: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    uri = f"file:{source.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "documents" not in tables:
            raise ValueError("legacy corpus database has no documents table")
        documents = [
            dict(row)
            for row in connection.execute("SELECT * FROM documents ORDER BY document_id")
        ]
        rejections = (
            [dict(row) for row in connection.execute("SELECT * FROM rejections ORDER BY rejection_id")]
            if "rejections" in tables
            else []
        )
    return documents, rejections


def rebuild_corpus_v2(
    source: Path, destination: Path, source_registry: Path
) -> RebuildReport:
    source = Path(source)
    destination = Path(destination)
    source_registry = Path(source_registry)
    if not source.is_file():
        raise FileNotFoundError(f"legacy corpus database does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    source_hash = _sha256(source)
    language_map = _load_language_map(source_registry)
    documents, rejections = _read_legacy_rows(source)
    source_keys = {
        (str(row["source_name"]), str(row["source_revision"]))
        for row in (*documents, *rejections)
    }
    unmapped = sorted(source_keys.difference(language_map))
    if unmapped:
        rendered = ", ".join(f"{name}@{revision}" for name, revision in unmapped)
        raise ValueError(f"unmapped legacy corpus source: {rendered}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    database: CorpusDatabase | None = None
    try:
        database = CorpusDatabase(temporary)
        with database.transaction():
            for row in documents:
                key = (str(row["source_name"]), str(row["source_revision"]))
                database.admit(
                    SourceDocumentRecord(
                        **row,
                        canonical_language=language_map[key],
                    )
                )
            for row in rejections:
                key = (str(row["source_name"]), str(row["source_revision"]))
                database.reject(
                    source_provider=str(row["source_provider"]),
                    source_name=key[0],
                    source_revision=key[1],
                    canonical_language=language_map[key],
                    repository_identity=str(row["repository_identity"]),
                    relative_path=str(row["relative_path"]),
                    content_sha256=str(row["content_sha256"]),
                    reason_code=str(row["reason_code"]),
                    detail=str(row["detail"]),
                    upstream_url=str(row["upstream_url"]),
                )
        database.close()
        database = None
        if _sha256(source) != source_hash:
            raise RuntimeError("legacy corpus database changed during read-only rebuild")
        os.link(temporary, destination)
        destination_hash = _sha256(destination)
    finally:
        if database is not None:
            database.close()
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}-wal").unlink(missing_ok=True)
        Path(f"{temporary}-shm").unlink(missing_ok=True)

    language_counts = Counter(
        language_map[(str(row["source_name"]), str(row["source_revision"]))]
        for row in documents
    )
    return RebuildReport(
        source_sha256=source_hash,
        destination_sha256=destination_hash,
        document_count=len(documents),
        language_counts=dict(sorted(language_counts.items())),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild a legacy BCNLT corpus at a new v2 path")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.report.exists():
        parser.error(f"report already exists: {args.report}")
    report = rebuild_corpus_v2(args.source, args.destination, args.source_registry)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
