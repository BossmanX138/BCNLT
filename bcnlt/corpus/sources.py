from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .database import CorpusDatabase, SourceDocumentRecord, stable_document_id
from .filters import FilterDecision, evaluate_python_document


@dataclass(frozen=True)
class SourceSpec:
    provider: str
    name: str
    revision: str
    upstream_url: str

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.name.strip() or not self.upstream_url.strip():
            raise ValueError("Source provider, name, and upstream URL are required")
        if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", self.revision):
            raise ValueError("Source revision must be an immutable 40- or 64-character hash")


def ingest_local_source(
    database: CorpusDatabase,
    spec: SourceSpec,
    root: str | Path,
    *,
    repository_identity: str,
    ownership: str,
    license_spdx: str,
    holdout_hashes: Iterable[str] = (),
) -> list[SourceDocumentRecord]:
    source_root = Path(root)
    if not source_root.is_dir():
        raise ValueError(f"Local source root is unavailable: {source_root}")
    admitted: list[SourceDocumentRecord] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
            _reject(
                database,
                spec,
                repository_identity,
                relative,
                content_hash,
                "decode_failure",
                str(exc),
            )
            continue
        decision = evaluate_python_document(
            relative_path=relative,
            content=content,
            license_value=license_spdx,
            holdout_hashes=holdout_hashes,
            allow_owner_provided=ownership == "owner_provided",
        )
        record = _admit_decision(
            database,
            spec,
            repository_identity,
            relative,
            content,
            decision,
            ownership=ownership,
            attribution=repository_identity,
        )
        if record is not None:
            admitted.append(record)
    return admitted


def ingest_stack_stream(
    database: CorpusDatabase,
    spec: SourceSpec,
    rows: Iterable[Mapping[str, object]],
    *,
    holdout_hashes: Iterable[str] = (),
) -> list[SourceDocumentRecord]:
    admitted: list[SourceDocumentRecord] = []
    for index, row in enumerate(rows):
        content = str(row.get("content") or "")
        relative = str(row.get("path") or f"row-{index}.py").replace("\\", "/")
        repository = str(
            row.get("repository")
            or row.get("max_stars_repo_name")
            or row.get("repo_name")
            or ""
        ).strip()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not repository:
            _reject(
                database,
                spec,
                "missing",
                relative,
                content_hash,
                "missing_repository_identity",
                "The Stack row has no repository identity",
            )
            continue
        license_value = row.get("license", row.get("max_stars_repo_licenses"))
        decision = evaluate_python_document(
            relative_path=relative,
            content=content,
            license_value=license_value,
            holdout_hashes=holdout_hashes,
        )
        record = _admit_decision(
            database,
            spec,
            repository,
            relative,
            content,
            decision,
            ownership="third_party",
            attribution=repository,
        )
        if record is not None:
            admitted.append(record)
    return admitted


def ingest_opencodeinstruct_stream(
    database: CorpusDatabase,
    spec: SourceSpec,
    rows: Iterable[Mapping[str, object]],
    *,
    holdout_hashes: Iterable[str] = (),
) -> list[SourceDocumentRecord]:
    admitted: list[SourceDocumentRecord] = []
    repository = f"{spec.name}@{spec.revision}"
    for index, row in enumerate(rows):
        row_id = str(row.get("id") or row.get("uuid") or index)
        relative = f"{_safe_identifier(row_id)}.py"
        content = _extract_python(row)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        try:
            score = float(row.get("average_test_score", -1))
        except (TypeError, ValueError):
            score = -1
        if score != 1.0:
            _reject(
                database,
                spec,
                repository,
                relative,
                content_hash,
                "instruction_tests_not_perfect",
                f"average_test_score={score}",
            )
            continue
        statuses = row.get("test_execution_status", ())
        if isinstance(statuses, str):
            statuses = [statuses]
        status_values = [str(item).strip().lower() for item in statuses] if statuses else []
        if not status_values or any(value not in {"pass", "passed", "success"} for value in status_values):
            _reject(
                database,
                spec,
                repository,
                relative,
                content_hash,
                "instruction_test_failure",
                f"statuses={status_values}",
            )
            continue
        decision = evaluate_python_document(
            relative_path=relative,
            content=content,
            license_value="MIT",
            holdout_hashes=holdout_hashes,
        )
        if decision.accepted:
            decision = FilterDecision(True, None, "CC-BY-4.0", decision.content_sha256)
        record = _admit_decision(
            database,
            spec,
            repository,
            relative,
            content,
            decision,
            ownership="third_party",
            attribution="NVIDIA OpenCodeInstruct",
        )
        if record is not None:
            admitted.append(record)
    return admitted


def _admit_decision(
    database: CorpusDatabase,
    spec: SourceSpec,
    repository_identity: str,
    relative_path: str,
    content: str,
    decision: FilterDecision,
    *,
    ownership: str,
    attribution: str,
) -> SourceDocumentRecord | None:
    if not decision.accepted or decision.license_spdx is None:
        _reject(
            database,
            spec,
            repository_identity,
            relative_path,
            decision.content_sha256,
            decision.reason_code or "rejected",
            decision.reason_code or "rejected",
        )
        return None
    record = SourceDocumentRecord(
        document_id=stable_document_id(
            provider=spec.provider,
            revision=spec.revision,
            repository_identity=repository_identity,
            relative_path=relative_path,
            content_sha256=decision.content_sha256,
        ),
        source_provider=spec.provider,
        source_name=spec.name,
        source_revision=spec.revision,
        repository_identity=repository_identity,
        relative_path=relative_path,
        content_sha256=decision.content_sha256,
        content=content,
        license_spdx=decision.license_spdx,
        ownership=ownership,
        attribution=attribution,
        upstream_url=spec.upstream_url,
    )
    database.admit(record)
    return record


def _reject(
    database: CorpusDatabase,
    spec: SourceSpec,
    repository_identity: str,
    relative_path: str,
    content_sha256: str,
    reason_code: str,
    detail: str,
) -> None:
    database.reject(
        source_provider=spec.provider,
        source_name=spec.name,
        source_revision=spec.revision,
        repository_identity=repository_identity,
        relative_path=relative_path,
        content_sha256=content_sha256,
        reason_code=reason_code,
        detail=detail,
        upstream_url=spec.upstream_url,
    )


def _extract_python(row: Mapping[str, object]) -> str:
    raw = next(
        (str(row[key]) for key in ("output", "solution", "response", "generated_solution") if row.get(key)),
        "",
    )
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() + "\n" if match else raw


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned or hashlib.sha256(value.encode()).hexdigest()[:16]
