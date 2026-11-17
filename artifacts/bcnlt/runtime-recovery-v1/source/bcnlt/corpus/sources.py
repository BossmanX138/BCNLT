from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .database import CorpusDatabase, SourceDocumentRecord, stable_document_id
from .filters import FilterDecision, evaluate_python_document, evaluate_source_document
from bcnlt.languages import LANGUAGE_BY_ID, resolve_language_id


EXTRACTION_RULE_VERSION = "bcnlt-multilingual-extraction-v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PARSER_BUNDLE_PATH = _PROJECT_ROOT / "configs" / "bcnlt_parser_bundle_v1.json"


def _parser_bundle_identity() -> tuple[str, str]:
    raw = _PARSER_BUNDLE_PATH.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    return hashlib.sha256(raw).hexdigest(), str(payload["release"])


def canonical_registry_entry_sha256(payload: Mapping[str, object]) -> str:
    excluded = {"registry_entry_sha256", "enabled", "note", "root", "ownership"}
    canonical_payload = {key: payload[key] for key in sorted(payload) if key not in excluded}
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceSpec:
    provider: str
    name: str
    revision: str
    upstream_url: str
    canonical_language: str
    data_dir: str | None = None
    dialect: str | None = None
    registry_entry_sha256: str = ""
    parser_bundle_sha256: str = ""
    grammar_revision: str = ""
    semantic_spec_sha256: str = ""
    extraction_rule_version: str = EXTRACTION_RULE_VERSION

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.name.strip() or not self.upstream_url.strip():
            raise ValueError("Source provider, name, and upstream URL are required")
        if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", self.revision):
            raise ValueError("Source revision must be an immutable 40- or 64-character hash")
        try:
            resolved_language = resolve_language_id(self.canonical_language)
        except ValueError as exc:
            raise ValueError("Source canonical language must use an exact registered id") from exc
        if resolved_language != self.canonical_language:
            raise ValueError("Source canonical language must use an exact registered id")
        definition = LANGUAGE_BY_ID[self.canonical_language]
        if self.dialect is not None and self.dialect not in definition.dialects:
            raise ValueError("Source dialect is not registered for its canonical language")
        parser_bundle_sha256, parser_release = _parser_bundle_identity()
        from bcnlt.common_core_registry import build_semantic_spec

        semantic_spec_sha256 = build_semantic_spec(self.canonical_language).canonical_hash
        grammar_revision = f"tree-sitter-language-pack@{parser_release}/{definition.grammar_id}"
        for field_name, derived in (
            ("parser_bundle_sha256", parser_bundle_sha256),
            ("grammar_revision", grammar_revision),
            ("semantic_spec_sha256", semantic_spec_sha256),
        ):
            supplied = getattr(self, field_name)
            if supplied and supplied != derived:
                raise ValueError(f"Source {field_name} does not match the pinned BCNLT registry")
            object.__setattr__(self, field_name, derived)
        entry_payload = {
            "provider": self.provider,
            "name": self.name,
            "revision": self.revision,
            "upstream_url": self.upstream_url,
            "canonical_language": self.canonical_language,
            "data_dir": self.data_dir,
            "dialect": self.dialect,
            "parser_bundle_sha256": self.parser_bundle_sha256,
            "grammar_revision": self.grammar_revision,
            "semantic_spec_sha256": self.semantic_spec_sha256,
            "extraction_rule_version": self.extraction_rule_version,
        }
        derived_entry_hash = canonical_registry_entry_sha256(entry_payload)
        if self.registry_entry_sha256 and self.registry_entry_sha256 != derived_entry_hash:
            raise ValueError("Source registry_entry_sha256 does not match its canonical entry")
        object.__setattr__(self, "registry_entry_sha256", derived_entry_hash)


def ingest_local_source(
    database: CorpusDatabase,
    spec: SourceSpec,
    root: str | Path,
    *,
    repository_identity: str,
    ownership: str,
    license_spdx: str,
    holdout_hashes: Iterable[str] = (),
    adapter: object | None = None,
) -> list[SourceDocumentRecord]:
    source_root = Path(root)
    if not source_root.is_dir():
        raise ValueError(f"Local source root is unavailable: {source_root}")
    admitted: list[SourceDocumentRecord] = []
    extensions = LANGUAGE_BY_ID[spec.canonical_language].accepted_extensions
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    for path in paths:
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
        decision = evaluate_source_document(
            relative_path=relative,
            content=content,
            license_value=license_spdx,
            language=spec.canonical_language,
            dialect=spec.dialect,
            holdout_hashes=holdout_hashes,
            allow_owner_provided=ownership == "owner_provided",
            adapter=adapter,
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
    adapter: object | None = None,
) -> list[SourceDocumentRecord]:
    admitted: list[SourceDocumentRecord] = []
    for index, row in enumerate(rows):
        content = str(row.get("content") or "")
        default_extension = LANGUAGE_BY_ID[spec.canonical_language].accepted_extensions[0]
        relative = str(row.get("path") or f"row-{index}{default_extension}").replace("\\", "/")
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
        decision = evaluate_source_document(
            relative_path=relative,
            content=content,
            license_value=license_value,
            language=spec.canonical_language,
            dialect=spec.dialect,
            holdout_hashes=holdout_hashes,
            adapter=adapter,
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
    adapter: object | None = None,
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
        statuses = row.get(
            "tests_execution_status",
            row.get("test_execution_status", ()),
        )
        if isinstance(statuses, str):
            try:
                decoded_statuses = json.loads(statuses)
            except json.JSONDecodeError:
                statuses = [statuses]
            else:
                statuses = (
                    decoded_statuses
                    if isinstance(decoded_statuses, list)
                    else [decoded_statuses]
                )
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
        decision = (
            evaluate_source_document(
                relative_path=relative,
                content=content,
                license_value="MIT",
                language="python",
                dialect=spec.dialect,
                holdout_hashes=holdout_hashes,
                adapter=adapter,
            )
            if adapter is not None
            else evaluate_python_document(
                relative_path=relative,
                content=content,
                license_value="MIT",
                holdout_hashes=holdout_hashes,
            )
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
            language=spec.canonical_language,
            registry_entry_sha256=spec.registry_entry_sha256,
        ),
        source_provider=spec.provider,
        source_name=spec.name,
        source_revision=spec.revision,
        canonical_language=spec.canonical_language,
        repository_identity=repository_identity,
        relative_path=relative_path,
        content_sha256=decision.content_sha256,
        content=content,
        license_spdx=decision.license_spdx,
        ownership=ownership,
        attribution=attribution,
        upstream_url=spec.upstream_url,
        dialect=spec.dialect,
        registry_entry_sha256=spec.registry_entry_sha256,
        parser_bundle_sha256=spec.parser_bundle_sha256,
        grammar_revision=spec.grammar_revision,
        semantic_spec_sha256=spec.semantic_spec_sha256,
        extraction_rule_version=spec.extraction_rule_version,
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
        canonical_language=spec.canonical_language,
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
