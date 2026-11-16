"""Freeze balanced packed-task acceptance cases from one audited test database."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .category_case_sources import _build_record
from .immutable_artifacts import publish_new_files
from .model_acceptance import (
    EXPECTED_LANGUAGES,
    MAXIMUM_CATEGORY_EXAMPLES,
    MINIMUM_CATEGORY_EXAMPLES,
)
from .resumable_task_packing import audit_task_database
from .tokenizer import NativeTokenizerBinding

PACKED_CATEGORY_AUDIT_SCHEMA = "bosscrafts-packed-category-case-freeze-audit-v1"
PACKED_CATEGORIES = (
    "tokenizer_fidelity",
    "semantic_fidelity",
    "raw_code",
)
_FAMILY_BY_CATEGORY = {
    "tokenizer_fidelity": None,
    "semantic_fidelity": "code_to_semantic",
    "raw_code": "semantic_to_code",
}
_CASE_KIND_BY_CATEGORY = {
    "tokenizer_fidelity": "packed_tokenizer_roundtrip",
    "semantic_fidelity": "packed_model_task",
    "raw_code": "packed_model_task",
}


@dataclass(frozen=True, slots=True)
class FrozenPackedCategoryCases:
    outputs: dict[str, Path]
    audit_output: Path
    task_database_identity_sha256: str
    record_counts: dict[str, int]
    audit_sha256: str


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _balanced_language_quotas(total: int) -> dict[str, int]:
    languages = tuple(sorted(EXPECTED_LANGUAGES))
    if type(total) is not int or total < len(languages):
        raise ValueError("packed category count cannot cover every language")
    quotient, remainder = divmod(total, len(languages))
    return {
        language: quotient + int(index < remainder)
        for index, language in enumerate(languages)
    }


def _select_rows(
    connection: sqlite3.Connection,
    *,
    family: str | None,
    quotas: dict[str, int],
    excluded_ids: set[str],
) -> list[tuple[str, str, str]]:
    selected: list[tuple[str, str, str]] = []
    counts: Counter[str] = Counter()
    parameters: tuple[str, ...]
    query = "SELECT example_id,language,family FROM tasks WHERE split='test'"
    if family is not None:
        query += " AND family=?"
        parameters = (family,)
    else:
        parameters = ()
    query += " ORDER BY example_id"
    for example_id, language, observed_family in connection.execute(query, parameters):
        identity = str(example_id)
        name = str(language)
        if identity in excluded_ids or name not in quotas or counts[name] >= quotas[name]:
            continue
        selected.append((identity, name, str(observed_family)))
        counts[name] += 1
        if all(counts[item] == quota for item, quota in quotas.items()):
            break
    missing = {
        language: quota - counts[language]
        for language, quota in quotas.items()
        if counts[language] != quota
    }
    if missing:
        raise ValueError(f"test task database cannot fill packed category quotas: {missing}")
    return selected


def freeze_packed_category_cases(
    binding: NativeTokenizerBinding,
    task_database: str | Path,
    outputs: Mapping[str, str | Path],
    audit_output: str | Path,
) -> FrozenPackedCategoryCases:
    """Audit a completed test database and freeze disjoint balanced case catalogs."""

    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if not isinstance(outputs, Mapping) or set(outputs) != set(PACKED_CATEGORIES):
        raise ValueError("packed category output paths are incomplete")
    destinations = {category: Path(outputs[category]).resolve() for category in outputs}
    audit_destination = Path(audit_output).resolve()
    if len({*destinations.values(), audit_destination}) != len(destinations) + 1:
        raise ValueError("packed category output paths must be distinct")

    audit = audit_task_database(binding, task_database)
    if not audit.audit_passed or set(audit.split_counts) != {"test"}:
        raise ValueError("packed category source must be an audited test-only database")
    counts = {
        category: MINIMUM_CATEGORY_EXAMPLES[category]
        for category in PACKED_CATEGORIES
    }
    if any(
        count < len(EXPECTED_LANGUAGES) or count > MAXIMUM_CATEGORY_EXAMPLES
        for count in counts.values()
    ):
        raise ValueError("packed category acceptance counts are invalid")

    database_path = Path(task_database).resolve(strict=True)
    connection = sqlite3.connect(
        f"{database_path.as_uri()}?mode=ro", uri=True, timeout=30
    )
    connection.execute("PRAGMA query_only=ON")
    selected: dict[str, list[tuple[str, str, str]]] = {}
    used_ids: set[str] = set()
    try:
        # Family-specific cases are selected first so tokenizer roundtrips cannot
        # consume a scarce model-task row. All three catalogs remain disjoint.
        for category in ("semantic_fidelity", "raw_code", "tokenizer_fidelity"):
            rows = _select_rows(
                connection,
                family=_FAMILY_BY_CATEGORY[category],
                quotas=_balanced_language_quotas(counts[category]),
                excluded_ids=used_ids,
            )
            selected[category] = rows
            used_ids.update(row[0] for row in rows)
    finally:
        connection.close()

    provenance: dict[str, object] = {
        "source": (
            "bosscrafts://audited-test-task-database/"
            f"{audit.database_identity_sha256}"
        ),
        "revision": audit.database_identity_sha256,
        "source_sha256": audit.content_sha256,
        "license": "mixed-source-license; see frozen source catalog",
        "attribution": "BossCrafts audited current-runtime test task database",
    }
    rendered: dict[str, bytes] = {}
    case_file_sha256: dict[str, str] = {}
    selected_task_ids_sha256: dict[str, str] = {}
    language_counts: dict[str, dict[str, int]] = {}
    for category in PACKED_CATEGORIES:
        records: list[dict[str, object]] = []
        observed: Counter[str] = Counter()
        selected_ids: list[str] = []
        for task_id, language, family in selected[category]:
            case: dict[str, object] = {
                "task_database_identity_sha256": audit.database_identity_sha256,
                "task_example_id": task_id,
                "split": "test",
                "language": language,
            }
            expected_family = _FAMILY_BY_CATEGORY[category]
            if expected_family is not None:
                if family != expected_family:
                    raise AssertionError("selected packed task family changed")
                case["family"] = expected_family
            records.append(
                _build_record(
                    category=category,
                    case_kind=_CASE_KIND_BY_CATEGORY[category],
                    case=case,
                    provenance=provenance,
                )
            )
            observed[language] += 1
            selected_ids.append(task_id)
        records.sort(key=lambda row: str(row["example_id"]))
        content = b"".join(_canonical(record) + b"\n" for record in records)
        rendered[category] = content
        case_file_sha256[category] = _sha256(content)
        selected_task_ids_sha256[category] = _sha256(_canonical(sorted(selected_ids)))
        language_counts[category] = dict(sorted(observed.items()))

    unsigned_audit: dict[str, object] = {
        "schema_version": PACKED_CATEGORY_AUDIT_SCHEMA,
        "task_database_identity_sha256": audit.database_identity_sha256,
        "task_database_content_sha256": audit.content_sha256,
        "task_database_example_count": audit.example_count,
        "selection_policy": (
            "minimum acceptance count; balanced deterministic per-language quotas; "
            "family-specific catalogs first; globally disjoint task IDs; ascending example ID"
        ),
        "record_counts": counts,
        "language_counts": language_counts,
        "selected_task_ids_sha256": selected_task_ids_sha256,
        "case_file_sha256": case_file_sha256,
        "training_ready": False,
    }
    audit_sha256 = _sha256(_canonical(unsigned_audit))
    audit_payload = {**unsigned_audit, "audit_sha256": audit_sha256}
    audit_content = (
        json.dumps(
            audit_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    publication = tuple(
        (destinations[category], rendered[category])
        for category in PACKED_CATEGORIES
    ) + ((audit_destination, audit_content),)
    publish_new_files(publication, artifact_label="packed category cases and audit")
    return FrozenPackedCategoryCases(
        outputs=destinations,
        audit_output=audit_destination,
        task_database_identity_sha256=audit.database_identity_sha256,
        record_counts=counts,
        audit_sha256=audit_sha256,
    )
