"""Content-addressed, leakage-resistant source freeze for V10 training admission."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from .current_corpus import APPROVED_LICENSES, SQLiteSourceCatalog
from .jsonl import split_jsonl_records

FREEZE_MANIFEST_SCHEMA = "bosscrafts-training-source-freeze-v1"
FREEZE_DOCUMENT_SCHEMA = "bosscrafts-training-source-document-v1"
CATALOG_SCHEMA = "bcnlt-corpus-database-v3"
HELDOUT_MANIFEST_SCHEMA = "bosscrafts-current-corpus-manifest-v2"
QUALIFICATION_SCHEMA = "bosscrafts-combined-semantic-audit-qualification-v1"
REPOSITORY_GROUPED_SOURCE = "bigcode/the-stack-dedup"
DOCUMENT_GROUPED_SOURCE = "nvidia/OpenCodeInstruct"


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


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def assign_source_split(split_group: str) -> str:
    """Assign an immutable 80/10/10 split from a repository or document group."""
    if type(split_group) is not str or not split_group:
        raise ValueError("split_group must be a non-empty string")
    bucket = int(hashlib.sha256(split_group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


@dataclass(frozen=True, slots=True)
class TrainingCorpusFreezeResult:
    output_root: Path
    source_document_count: int
    training_document_count: int
    reserved_heldout_document_count: int
    documents_sha256: str
    manifest_sha256: str
    training_ready: bool


class VerifiedTrainingSourceFreeze:
    """Read an immutable source freeze and expose only its verified train split."""

    _CANDIDATE_FIELDS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "language": "language",
            "dialect": "dialect",
            "relative_path": "relative_path",
            "content_sha256": "content_sha256",
            "license_spdx": "license_spdx",
            "attribution": "attribution",
            "ownership": "ownership",
            "source_provider": "source_provider",
            "source_name": "source_name",
            "source_revision": "source_revision",
            "repository_identity": "repository_identity",
            "upstream_url": "upstream_url",
        }
    )

    def __init__(
        self,
        freeze_root: str | Path,
        catalog_path: str | Path,
        *,
        expected_manifest_sha256: str,
    ) -> None:
        if not _is_sha256(expected_manifest_sha256):
            raise ValueError("expected freeze manifest SHA-256 is invalid")
        self.freeze_root = Path(freeze_root).resolve()
        self.catalog_path = Path(catalog_path).resolve()
        manifest_path = self.freeze_root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("training source freeze manifest is invalid") from exc
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if (
            manifest.get("schema_version") != FREEZE_MANIFEST_SCHEMA
            or claimed != _sha256_bytes(_canonical(unsigned).encode("utf-8"))
            or claimed != expected_manifest_sha256
            or manifest.get("source_freeze_complete") is not True
            or manifest.get("training_ready") is not False
            or manifest.get("repository_cross_split_count") != 0
        ):
            raise ValueError("training source freeze manifest lineage mismatch")
        catalog_sha256 = manifest.get("source_catalog_sha256")
        if not _is_sha256(catalog_sha256) or _sha256_file(self.catalog_path) != catalog_sha256:
            raise ValueError("frozen source catalog SHA-256 mismatch")

        documents_path = self.freeze_root / str(manifest.get("documents_file"))
        try:
            documents_bytes = documents_path.read_bytes()
        except OSError as exc:
            raise ValueError("frozen document inventory cannot be read") from exc
        if _sha256_bytes(documents_bytes) != manifest.get("documents_sha256"):
            raise ValueError("frozen document inventory SHA-256 mismatch")
        rows: dict[str, dict[str, object]] = {}
        split_counts: Counter[str] = Counter()
        for line in split_jsonl_records(documents_bytes.decode("utf-8")):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("frozen document inventory JSON is invalid") from exc
            document_id = row.get("document_id") if type(row) is dict else None
            split = row.get("split") if type(row) is dict else None
            if (
                type(row) is not dict
                or line != _canonical(row)
                or row.get("schema_version") != FREEZE_DOCUMENT_SCHEMA
                or type(document_id) is not str
                or not document_id
                or document_id in rows
                or split not in {"train", "validation", "test", "reserved_heldout"}
            ):
                raise ValueError("frozen document inventory row is invalid")
            rows[document_id] = row
            split_counts[str(split)] += 1
        if (
            len(rows) != manifest.get("source_document_count")
            or split_counts.get("train", 0) != manifest.get("training_document_count")
            or dict(sorted(split_counts.items())) != manifest.get("split_counts")
        ):
            raise ValueError("frozen document inventory counts mismatch")

        connection = sqlite3.connect(
            f"{self.catalog_path.as_uri()}?mode=ro", uri=True, timeout=30
        )
        connection.execute("PRAGMA query_only=ON")
        try:
            catalog_ids = {
                str(row[0]) for row in connection.execute("SELECT document_id FROM documents")
            }
        except sqlite3.Error as exc:
            raise ValueError("frozen source catalog database is invalid") from exc
        finally:
            connection.close()
        if catalog_ids != set(rows):
            raise ValueError("frozen document inventory and source catalog differ")

        self._manifest = manifest
        self._records = rows
        self._training_ids = {
            document_id
            for document_id, row in rows.items()
            if row["split"] == "train"
        }

    @property
    def manifest_sha256(self) -> str:
        return str(self._manifest["manifest_sha256"])

    @property
    def source_catalog_sha256(self) -> str:
        return str(self._manifest["source_catalog_sha256"])

    @property
    def qualification_file_sha256(self) -> str:
        return str(self._manifest["semantic_audit_qualification_file_sha256"])

    @property
    def training_document_count(self) -> int:
        return int(self._manifest["training_document_count"])

    def iter_training_candidates(self):
        """Yield source candidates only after the complete inventory is verified."""
        seen: set[str] = set()
        catalog = SQLiteSourceCatalog(
            self.catalog_path, expected_sha256=self.source_catalog_sha256
        )
        for candidate in catalog.iter_candidates(limit_per_language=None):
            if candidate.document_id not in self._training_ids:
                continue
            record = self._records[candidate.document_id]
            for record_field, candidate_field in self._CANDIDATE_FIELDS.items():
                if record[record_field] != getattr(candidate, candidate_field):
                    raise ValueError(
                        f"frozen training candidate metadata mismatch: {candidate.document_id}"
                    )
            seen.add(candidate.document_id)
            yield candidate
        if seen != self._training_ids:
            raise ValueError("frozen training candidate inventory is incomplete")


class TrainingCorpusFreezer:
    """Verify source lineage and freeze groups without authorizing training."""

    _CATALOG_COLUMNS = (
        "document_id",
        "source_provider",
        "source_name",
        "source_revision",
        "canonical_language",
        "repository_identity",
        "relative_path",
        "content_sha256",
        "content",
        "license_spdx",
        "ownership",
        "attribution",
        "upstream_url",
        "dialect",
    )

    def materialize(
        self,
        catalog_path: str | Path,
        heldout_admission_root: str | Path,
        qualification_path: str | Path,
        output_root: str | Path,
        *,
        expected_catalog_sha256: str,
        expected_heldout_manifest_file_sha256: str,
        expected_qualification_file_sha256: str,
    ) -> TrainingCorpusFreezeResult:
        for name, value in (
            ("expected catalog", expected_catalog_sha256),
            ("expected heldout manifest file", expected_heldout_manifest_file_sha256),
            ("expected qualification file", expected_qualification_file_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} SHA-256 is invalid")

        catalog = Path(catalog_path).resolve()
        heldout_root = Path(heldout_admission_root).resolve()
        qualification = Path(qualification_path).resolve()
        output = Path(output_root).resolve()
        if not catalog.is_file():
            raise ValueError("source catalog does not exist")
        if _sha256_file(catalog) != expected_catalog_sha256:
            raise ValueError("source catalog SHA-256 mismatch")
        if output.exists() and any(output.iterdir()):
            raise ValueError("training source freeze output already exists and is not empty")

        rows = self._load_catalog(catalog)
        by_id = {str(row["document_id"]): row for row in rows}
        heldout_manifest, heldout_ids = self._load_heldout(
            heldout_root,
            by_id,
            expected_catalog_sha256,
            expected_heldout_manifest_file_sha256,
        )
        qualification_report = self._load_qualification(
            qualification, expected_qualification_file_sha256
        )

        heldout_groups = {
            self._split_group(by_id[document_id])[1] for document_id in heldout_ids
        }
        records: list[dict[str, object]] = []
        split_counts: Counter[str] = Counter()
        language_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
        license_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        repository_splits: dict[str, set[str]] = defaultdict(set)
        total_source_bytes = 0

        for row in sorted(rows, key=lambda item: str(item["document_id"])):
            group_kind, group = self._split_group(row)
            split = "reserved_heldout" if group in heldout_groups else assign_source_split(group)
            document_id = str(row["document_id"])
            language = str(row["canonical_language"])
            source_name = str(row["source_name"])
            license_spdx = str(row["license_spdx"])
            split_counts[split] += 1
            language_split_counts[language][split] += 1
            license_counts[license_spdx] += 1
            source_counts[source_name] += 1
            total_source_bytes += len(str(row["content"]).encode("utf-8"))
            if group_kind == "repository":
                repository_splits[str(row["repository_identity"])].add(split)
            records.append(
                {
                    "schema_version": FREEZE_DOCUMENT_SCHEMA,
                    "document_id": document_id,
                    "language": language,
                    "dialect": row["dialect"],
                    "relative_path": row["relative_path"],
                    "content_sha256": row["content_sha256"],
                    "license_spdx": license_spdx,
                    "attribution": row["attribution"],
                    "ownership": row["ownership"],
                    "source_provider": row["source_provider"],
                    "source_name": source_name,
                    "source_revision": row["source_revision"],
                    "repository_identity": row["repository_identity"],
                    "upstream_url": row["upstream_url"],
                    "split_group_kind": group_kind,
                    "split_group_sha256": _sha256_bytes(group.encode("utf-8")),
                    "split": split,
                    "heldout_reviewed": document_id in heldout_ids,
                }
            )

        repository_cross_split_count = sum(
            len(splits) > 1 for splits in repository_splits.values()
        )
        if repository_cross_split_count:
            raise ValueError("repository-grouped source crosses frozen splits")
        documents_text = "".join(f"{_canonical(record)}\n" for record in records)
        documents_sha256 = _sha256_bytes(documents_text.encode("utf-8"))
        unsigned: dict[str, object] = {
            "schema_version": FREEZE_MANIFEST_SCHEMA,
            "source_freeze_complete": True,
            "source_catalog_schema_version": CATALOG_SCHEMA,
            "source_catalog_file": catalog.name,
            "source_catalog_sha256": expected_catalog_sha256,
            "source_document_count": len(records),
            "source_bytes": total_source_bytes,
            "documents_file": "documents.jsonl",
            "documents_sha256": documents_sha256,
            "split_policy": {
                "hash": "sha256-first-32-bits-mod-100",
                "ratios": {"train": 80, "validation": 10, "test": 10},
                "repository_grouped_sources": [REPOSITORY_GROUPED_SOURCE],
                "document_grouped_sources": [DOCUMENT_GROUPED_SOURCE],
                "reviewed_heldout_policy": "reserve every reviewed document and its split group",
            },
            "split_counts": dict(sorted(split_counts.items())),
            "training_document_count": split_counts["train"],
            "reserved_heldout_document_count": split_counts["reserved_heldout"],
            "reviewed_heldout_document_count": len(heldout_ids),
            "repository_cross_split_count": repository_cross_split_count,
            "per_language": {
                language: dict(sorted(counts.items()))
                for language, counts in sorted(language_split_counts.items())
            },
            "per_license": dict(sorted(license_counts.items())),
            "per_source": dict(sorted(source_counts.items())),
            "heldout_admission_manifest_sha256": heldout_manifest["manifest_sha256"],
            "heldout_admission_manifest_file_sha256": (
                expected_heldout_manifest_file_sha256
            ),
            "binding_sha256": heldout_manifest["binding_sha256"],
            "tokenizer_sha256": heldout_manifest["tokenizer_sha256"],
            "parser_runtime_identity_sha256": heldout_manifest[
                "parser_runtime_identity_sha256"
            ],
            "semantic_audit_qualification_report_sha256": qualification_report[
                "report_sha256"
            ],
            "semantic_audit_qualification_file_sha256": (
                expected_qualification_file_sha256
            ),
            "semantic_audit_qualification_passed": True,
            "training_ready": False,
            "training_blockers": [
                "full current-runtime semantic admission is incomplete",
                "qualified semantic audits have not run over every frozen training record",
                "record-boundary task packing and loss-mask dry-run are incomplete",
                "model training remains explicitly held",
            ],
        }
        manifest_sha256 = _sha256_bytes(_canonical(unsigned).encode("utf-8"))
        manifest = {**unsigned, "manifest_sha256": manifest_sha256}
        _atomic_write(output / "documents.jsonl", documents_text.encode("utf-8"))
        _atomic_write(
            output / "manifest.json", f"{_canonical(manifest)}\n".encode()
        )
        return TrainingCorpusFreezeResult(
            output_root=output,
            source_document_count=len(records),
            training_document_count=split_counts["train"],
            reserved_heldout_document_count=split_counts["reserved_heldout"],
            documents_sha256=documents_sha256,
            manifest_sha256=manifest_sha256,
            training_ready=False,
        )

    def _load_catalog(self, catalog: Path) -> list[dict[str, Any]]:
        connection = sqlite3.connect(f"{catalog.as_uri()}?mode=ro", uri=True, timeout=30)
        connection.execute("PRAGMA query_only=ON")
        try:
            if str(connection.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
                raise ValueError("source catalog integrity check failed")
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            if metadata.get("schema_version") != CATALOG_SCHEMA:
                raise ValueError("source catalog schema mismatch")
            query = (
                "SELECT document_id,source_provider,source_name,source_revision,"
                "canonical_language,repository_identity,relative_path,content_sha256,"
                "content,license_spdx,ownership,attribution,upstream_url,dialect "
                "FROM documents ORDER BY document_id"
            )
            raw_rows = connection.execute(query).fetchall()
        except sqlite3.Error as exc:
            raise ValueError("source catalog database is invalid") from exc
        finally:
            connection.close()
        if not raw_rows:
            raise ValueError("source catalog has no documents")
        rows: list[dict[str, Any]] = []
        document_ids: set[str] = set()
        content_hashes: set[str] = set()
        for raw in raw_rows:
            row = dict(zip(self._CATALOG_COLUMNS, raw, strict=True))
            required = [
                row[column]
                for column in self._CATALOG_COLUMNS
                if column not in {"dialect"}
            ]
            if any(type(value) is not str or not value for value in required):
                raise ValueError("source catalog contains invalid document metadata")
            if row["source_name"] not in {
                REPOSITORY_GROUPED_SOURCE,
                DOCUMENT_GROUPED_SOURCE,
            }:
                raise ValueError("source catalog contains an unclassified split source")
            if row["license_spdx"] not in APPROVED_LICENSES:
                raise ValueError("source catalog contains an unapproved license")
            document_id = str(row["document_id"])
            content_hash = str(row["content_sha256"])
            if not _is_sha256(content_hash):
                raise ValueError("source catalog content SHA-256 is invalid")
            if _sha256_bytes(str(row["content"]).encode("utf-8")) != content_hash:
                raise ValueError("catalog content SHA-256 mismatch")
            if document_id in document_ids:
                raise ValueError("source catalog contains duplicate document IDs")
            if content_hash in content_hashes:
                raise ValueError("source catalog contains duplicate content hashes")
            document_ids.add(document_id)
            content_hashes.add(content_hash)
            rows.append(row)
        return rows

    @staticmethod
    def _split_group(row: dict[str, Any]) -> tuple[str, str]:
        if row["source_name"] == REPOSITORY_GROUPED_SOURCE:
            return "repository", str(row["repository_identity"])
        if row["source_name"] == DOCUMENT_GROUPED_SOURCE:
            return "document", str(row["document_id"])
        raise ValueError("source catalog contains an unclassified split source")

    def _load_heldout(
        self,
        root: Path,
        catalog_by_id: dict[str, dict[str, Any]],
        catalog_sha256: str,
        expected_file_sha256: str,
    ) -> tuple[dict[str, Any], set[str]]:
        manifest_path = root / "manifest.json"
        if _sha256_file(manifest_path) != expected_file_sha256:
            raise ValueError("heldout manifest file SHA-256 mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("heldout manifest is invalid") from exc
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if (
            manifest.get("schema_version") != HELDOUT_MANIFEST_SCHEMA
            or claimed != _sha256_bytes(_canonical(unsigned).encode("utf-8"))
            or manifest.get("source_catalog_sha256") != catalog_sha256
        ):
            raise ValueError("heldout admission lineage mismatch")
        heldout_ids: set[str] = set()
        count = 0
        for file_key, hash_key, expected_count, hash_field in (
            ("records_file", "records_sha256", "admitted_count", "source_sha256"),
            ("rejections_file", "rejections_sha256", "rejected_count", "content_sha256"),
        ):
            path = root / str(manifest[file_key])
            data = path.read_bytes()
            if _sha256_bytes(data) != manifest[hash_key]:
                raise ValueError("heldout admission payload SHA-256 mismatch")
            rows = [
                json.loads(line)
                for line in split_jsonl_records(data.decode("utf-8"))
            ]
            if len(rows) != manifest[expected_count]:
                raise ValueError("heldout admission payload count mismatch")
            for row in rows:
                document_id = row.get("document_id")
                if type(document_id) is not str or document_id in heldout_ids:
                    raise ValueError("heldout admission document inventory is invalid")
                catalog_row = catalog_by_id.get(document_id)
                if (
                    catalog_row is None
                    or row.get(hash_field) != catalog_row["content_sha256"]
                    or assign_source_split(document_id) == "train"
                ):
                    raise ValueError("heldout admission document lineage mismatch")
                heldout_ids.add(document_id)
                count += 1
        if count != int(manifest["admitted_count"]) + int(manifest["rejected_count"]):
            raise ValueError("heldout admission document count mismatch")
        return manifest, heldout_ids

    @staticmethod
    def _load_qualification(path: Path, expected_file_sha256: str) -> dict[str, Any]:
        if _sha256_file(path) != expected_file_sha256:
            raise ValueError("qualification file SHA-256 mismatch")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("semantic audit qualification is invalid") from exc
        unsigned = dict(report)
        claimed = unsigned.pop("report_sha256", None)
        metrics = report.get("defect_metrics")
        if (
            report.get("schema_version") != QUALIFICATION_SCHEMA
            or claimed != _sha256_bytes(_canonical(unsigned).encode("utf-8"))
            or report.get("audit_qualification_passed") is not True
            or report.get("positive_false_flag_count") != 0
            or report.get("negative_detected_count")
            != report.get("negative_control_count")
            or type(metrics) is not dict
            or not metrics
            or any(
                type(metric) is not dict
                or metric.get("total", 0) <= 0
                or metric.get("detected") != metric.get("total")
                for metric in metrics.values()
            )
        ):
            raise ValueError("semantic audit qualification did not pass exact gates")
        return report
