"""Deterministic accepted-only Project CodeNet pilot selection."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol, TypedDict

from .immutable_artifacts import publish_new_files

PILOT_SCHEMA = "bosscrafts-project-codenet-pilot-v1"
DATASET_LICENSE = "CDLA-Permissive-2.0"
_HEX64 = re.compile(r"[0-9a-f]{64}")


class OverlapFilter(Protocol):
    sha256: str

    def overlaps(self, prompt: str, code: str) -> bool: ...


class PilotSelection(TypedDict):
    languages: list[str]
    per_language_limit: int
    seed: str
    minimum_code_bytes: int
    maximum_code_bytes: int
    one_solution_per_problem_language_pair: bool
    split_unit: str


class PilotReport(TypedDict):
    schema_version: str
    dataset: str
    dataset_license: str
    archive_sha256: str
    tooling_commit: str
    selection: PilotSelection
    benchmark_exclusion_sha256: str | None
    record_count: int
    language_counts: dict[str, int]
    split_counts: dict[str, int]
    rejection_counts: dict[str, int]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag.lower() in {
            "br",
            "p",
            "div",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "pre",
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag.lower() in {
            "p",
            "div",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "pre",
        }:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line).strip()


@dataclass(frozen=True)
class Candidate:
    rank: str
    problem_id: str
    submission_id: str
    language: str
    filename_ext: str
    source_path: Path


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rank(seed: str, problem_id: str, language: str, submission_id: str) -> str:
    return _sha256_text("\0".join((seed, problem_id, language, submission_id)))


def _split(problem_id: str) -> str:
    bucket = int(_sha256_text(problem_id)[:8], 16) % 100
    if bucket < 96:
        return "train"
    if bucket < 98:
        return "validation"
    return "test"


def _normalize_code(code: str) -> str:
    lines = code.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(line.rstrip() for line in lines).strip() + "\n"


def _description_text(path: Path) -> str:
    extractor = _TextExtractor()
    extractor.feed(path.read_text(encoding="utf-8", errors="replace"))
    return extractor.text()


def select_candidates(
    dataset_root: Path,
    *,
    languages: set[str],
    per_language_limit: int,
    seed: str,
    minimum_code_bytes: int,
    maximum_code_bytes: int,
) -> tuple[list[Candidate], Counter[str]]:
    if not languages:
        raise ValueError("at least one language is required")
    if per_language_limit < 1:
        raise ValueError("per-language limit must be positive")
    best_by_problem_language: dict[tuple[str, str], Candidate] = {}
    rejections: Counter[str] = Counter()
    metadata_root = dataset_root / "metadata"
    for metadata_path in sorted(metadata_root.glob("p[0-9]*.csv")):
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("status", "").strip() != "Accepted":
                    rejections["not_accepted"] += 1
                    continue
                language = row.get("language", "").strip()
                if language not in languages:
                    rejections["language_not_requested"] += 1
                    continue
                try:
                    code_size = int(row.get("code_size", ""))
                except ValueError:
                    rejections["invalid_code_size"] += 1
                    continue
                if not minimum_code_bytes <= code_size <= maximum_code_bytes:
                    rejections["code_size_out_of_range"] += 1
                    continue
                problem_id = row.get("problem_id", "").strip()
                submission_id = row.get("submission_id", "").strip()
                filename_ext = row.get("filename_ext", "").strip().lstrip(".")
                if not problem_id or not submission_id or not filename_ext:
                    rejections["missing_identity_field"] += 1
                    continue
                source_path = (
                    dataset_root
                    / "data"
                    / problem_id
                    / language
                    / f"{submission_id}.{filename_ext}"
                )
                if not source_path.is_file():
                    rejections["missing_source_file"] += 1
                    continue
                candidate = Candidate(
                    rank=_rank(seed, problem_id, language, submission_id),
                    problem_id=problem_id,
                    submission_id=submission_id,
                    language=language,
                    filename_ext=filename_ext,
                    source_path=source_path,
                )
                key = (problem_id, language)
                current = best_by_problem_language.get(key)
                if current is None or candidate.rank < current.rank:
                    best_by_problem_language[key] = candidate

    by_language: dict[str, list[Candidate]] = {language: [] for language in languages}
    for candidate in best_by_problem_language.values():
        by_language[candidate.language].append(candidate)
    selected: list[Candidate] = []
    for language in sorted(languages):
        ranked = sorted(by_language[language], key=lambda item: item.rank)
        selected.extend(ranked[:per_language_limit])
        rejections["per_language_limit"] += max(0, len(ranked) - per_language_limit)
    return sorted(selected, key=lambda item: (item.rank, item.problem_id, item.language)), rejections


def build_pilot(
    dataset_root: Path,
    *,
    languages: set[str],
    per_language_limit: int,
    seed: str,
    minimum_code_bytes: int,
    maximum_code_bytes: int,
    archive_sha256: str,
    tooling_commit: str,
    overlap_filter: OverlapFilter | None,
) -> tuple[list[dict[str, object]], PilotReport]:
    if not _HEX64.fullmatch(archive_sha256):
        raise ValueError("archive SHA-256 is invalid")
    candidates, rejections = select_candidates(
        dataset_root,
        languages=languages,
        per_language_limit=per_language_limit,
        seed=seed,
        minimum_code_bytes=minimum_code_bytes,
        maximum_code_bytes=maximum_code_bytes,
    )
    records: list[dict[str, object]] = []
    seen_code: set[str] = set()
    language_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for candidate in candidates:
        description_path = dataset_root / "problem_descriptions" / f"{candidate.problem_id}.html"
        if not description_path.is_file():
            rejections["missing_problem_description"] += 1
            continue
        description = _description_text(description_path)
        if len(description) < 80:
            rejections["description_too_short"] += 1
            continue
        code = _normalize_code(
            candidate.source_path.read_text(encoding="utf-8", errors="replace")
        )
        code_sha256 = _sha256_text(code)
        if code_sha256 in seen_code:
            rejections["exact_duplicate_code"] += 1
            continue
        prompt = (
            f"Write a complete {candidate.language} program that solves the following "
            f"problem. Return only the complete source code.\n\n{description}"
        )
        if overlap_filter is not None and overlap_filter.overlaps(prompt, code):
            rejections["benchmark_overlap"] += 1
            continue
        seen_code.add(code_sha256)
        split = _split(candidate.problem_id)
        relative_source = candidate.source_path.relative_to(dataset_root).as_posix()
        record: dict[str, object] = {
            "schema_version": PILOT_SCHEMA,
            "dataset": "IBM/Project_CodeNet",
            "dataset_license": DATASET_LICENSE,
            "archive_sha256": archive_sha256,
            "tooling_commit": tooling_commit,
            "problem_id": candidate.problem_id,
            "submission_id": candidate.submission_id,
            "language": candidate.language,
            "split": split,
            "prompt": prompt,
            "code": code,
            "description_sha256": _sha256_text(description),
            "code_sha256": code_sha256,
            "source_path": relative_source,
            "selection_rank": candidate.rank,
        }
        records.append(record)
        language_counts[candidate.language] += 1
        split_counts[split] += 1
    report: PilotReport = {
        "schema_version": PILOT_SCHEMA,
        "dataset": "IBM/Project_CodeNet",
        "dataset_license": DATASET_LICENSE,
        "archive_sha256": archive_sha256,
        "tooling_commit": tooling_commit,
        "selection": {
            "languages": sorted(languages),
            "per_language_limit": per_language_limit,
            "seed": seed,
            "minimum_code_bytes": minimum_code_bytes,
            "maximum_code_bytes": maximum_code_bytes,
            "one_solution_per_problem_language_pair": True,
            "split_unit": "problem_id",
        },
        "benchmark_exclusion_sha256": (
            overlap_filter.sha256 if overlap_filter is not None else None
        ),
        "record_count": len(records),
        "language_counts": dict(language_counts),
        "split_counts": dict(split_counts),
        "rejection_counts": dict(rejections),
    }
    return records, report


def publish_pilot_artifacts(
    output: Path,
    report_path: Path,
    records: list[dict[str, object]],
    report: PilotReport,
) -> None:
    """Publish the corpus and its report as one immutable generation."""
    if not records:
        raise ValueError("Project CodeNet pilot requires at least one record")
    if report.get("record_count") != len(records):
        raise ValueError("Project CodeNet pilot report count mismatch")
    try:
        records_content = "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for record in records
        ).encode("utf-8")
        report_content = (
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Project CodeNet pilot artifacts are not JSON safe") from error
    publish_new_files(
        (
            (output.resolve(), records_content),
            (report_path.resolve(), report_content),
        ),
        artifact_label="Project CodeNet pilot",
    )
