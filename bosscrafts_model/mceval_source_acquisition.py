"""Acquire a pinned McEval generation snapshot for evaluation-only use."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import quote

from .category_case_sources import (
    MCEVAL_LANGUAGE_FILES,
    MCEVAL_SOURCE_MANIFEST_SCHEMA,
)
from .immutable_artifacts import publish_new_files

MCEVAL_DATASET_ID = "Multilingual-Multimodal-NLP/McEval"
MCEVAL_DATASET_URI = f"hf://datasets/{MCEVAL_DATASET_ID}"
MCEVAL_LICENSE = "CC-BY-SA-4.0"
MCEVAL_ATTRIBUTION = "McEval; Chai et al., 2024"
ACQUISITION_SCHEMA = "bosscrafts-mceval-source-acquisition-v1"

_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_FIELDS = {
    "task_id",
    "prompt",
    "canonical_solution",
    "test",
    "entry_point",
    "signature",
    "docstring",
    "instruction",
    "level",
}
_MAX_FILE_BYTES = 16 * 1024 * 1024


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate source key: {key}")
        result[key] = value
    return result


def _fetch_url(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BossCrafts-BCNLT-evaluation-source-acquirer/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > _MAX_FILE_BYTES:
                raise ValueError("McEval source file exceeds the acquisition bound")
            content = cast(bytes, response.read(_MAX_FILE_BYTES + 1))
    except (OSError, ValueError) as exc:
        raise ValueError(f"McEval source download failed: {url}") from exc
    if not content or len(content) > _MAX_FILE_BYTES:
        raise ValueError("McEval source file size is invalid")
    return content


def _validate_jsonl(content: bytes, *, language: str) -> int:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"McEval {language} source is not UTF-8") from exc
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError(f"McEval {language} source must be non-empty JSONL")
    task_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"McEval {language} line {line_number} is invalid JSON"
            ) from exc
        if type(row) is not dict or set(row) != _EXPECTED_FIELDS:
            raise ValueError(f"McEval {language} line {line_number} fields are invalid")
        task_id = row.get("task_id")
        required = (
            task_id,
            row.get("prompt"),
            row.get("canonical_solution"),
            row.get("test"),
        )
        if (
            any(type(value) is not str or not value.strip() for value in required)
            or task_id in task_ids
        ):
            raise ValueError(f"McEval {language} line {line_number} task is invalid")
        task_ids.add(str(task_id))
    return len(lines)


@dataclass(frozen=True, slots=True)
class McEvalAcquisitionResult:
    output_root: Path
    revision: str
    source_manifest_sha256: str
    acquisition_receipt_sha256: str
    source_file_count: int
    source_row_count: int


def acquire_mceval_generation_sources(
    output_root: str | Path,
    *,
    revision: str,
    fetch: Callable[[str], bytes] = _fetch_url,
) -> McEvalAcquisitionResult:
    """Download and freeze the exact 14-language generation source snapshot.

    The resulting files are evaluation-only.  In particular, this function does
    not add CC-BY-SA material to the training-source license allowlist.
    """

    if _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError("McEval revision must be an exact 40-character Git SHA")
    if not callable(fetch):
        raise TypeError("fetch must be callable")
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("McEval acquisition output already exists and is not empty")

    files: list[tuple[Path, bytes]] = []
    language_manifest: dict[str, object] = {}
    acquisition_files: dict[str, object] = {}
    total_rows = 0
    base_url = f"https://huggingface.co/datasets/{MCEVAL_DATASET_ID}/resolve/{revision}"
    for language, filename in sorted(MCEVAL_LANGUAGE_FILES.items()):
        url = f"{base_url}/generation/{quote(filename, safe='')}"
        content = fetch(url)
        if type(content) is not bytes:
            raise TypeError("McEval fetcher must return bytes")
        if not content or len(content) > _MAX_FILE_BYTES:
            raise ValueError(f"McEval {language} source file size is invalid")
        row_count = _validate_jsonl(content, language=language)
        digest = _sha256(content)
        total_rows += row_count
        language_manifest[language] = {"file": filename, "sha256": digest}
        acquisition_files[language] = {
            "bytes": len(content),
            "file": filename,
            "row_count": row_count,
            "sha256": digest,
            "url": url,
        }
        files.append((output / filename, content))

    manifest = {
        "schema_version": MCEVAL_SOURCE_MANIFEST_SCHEMA,
        "source": MCEVAL_DATASET_URI,
        "revision": revision,
        "license": MCEVAL_LICENSE,
        "attribution": MCEVAL_ATTRIBUTION,
        "languages": language_manifest,
    }
    manifest_bytes = _canonical(manifest)
    manifest_sha256 = _sha256(manifest_bytes)
    receipt_unsigned = {
        "schema_version": ACQUISITION_SCHEMA,
        "source": MCEVAL_DATASET_URI,
        "revision": revision,
        "license": MCEVAL_LICENSE,
        "attribution": MCEVAL_ATTRIBUTION,
        "selection": "generation files for the 14 BCNLT languages",
        "source_file_count": len(language_manifest),
        "source_row_count": total_rows,
        "files": acquisition_files,
        "source_manifest_file": "source-manifest.json",
        "source_manifest_sha256": manifest_sha256,
        "evaluation_only": True,
        "training_ready": False,
        "training_blockers": [
            "CC-BY-SA-4.0 benchmark material is isolated from training sources",
            "acquisition alone is not semantic calibration evidence",
        ],
    }
    receipt_sha256 = _sha256(_canonical(receipt_unsigned)[:-1])
    receipt = {**receipt_unsigned, "receipt_sha256": receipt_sha256}
    files.extend(
        (
            (output / "source-manifest.json", manifest_bytes),
            (output / "acquisition-receipt.json", _canonical(receipt)),
        )
    )
    publish_new_files(tuple(files), artifact_label="pinned McEval evaluation sources")
    return McEvalAcquisitionResult(
        output_root=output,
        revision=revision,
        source_manifest_sha256=manifest_sha256,
        acquisition_receipt_sha256=receipt_sha256,
        source_file_count=len(language_manifest),
        source_row_count=total_rows,
    )
