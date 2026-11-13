"""Audit an extracted IBM Project CodeNet source tree.

The audit is intentionally read-only and fail-closed.  It counts every metadata
row and, when requested, verifies that every accepted submission resolves to the
source file implied by the official Project CodeNet directory convention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "bosscrafts-project-codenet-source-audit-v1"
REQUIRED_COLUMNS = {
    "submission_id",
    "problem_id",
    "language",
    "filename_ext",
    "status",
    "code_size",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_root(path: Path) -> Path:
    path = path.resolve()
    candidates = (path, path / "Project_CodeNet", path / "Project CodeNet")
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in ("data", "metadata")):
            return candidate
    raise ValueError(f"not a Project CodeNet dataset root: {path}")


def _source_path(dataset_root: Path, row: dict[str, str]) -> Path:
    extension = row["filename_ext"].strip().lstrip(".")
    filename = row["submission_id"].strip()
    if extension:
        filename = f"{filename}.{extension}"
    return (
        dataset_root
        / "data"
        / row["problem_id"].strip()
        / row["language"].strip()
        / filename
    )


def audit_dataset(
    dataset_root: Path,
    *,
    verify_accepted_files: bool,
    archive_path: Path | None = None,
    tooling_commit: str | None = None,
) -> dict[str, Any]:
    root = resolve_dataset_root(dataset_root)
    metadata_root = root / "metadata"
    description_root = root / "problem_descriptions"
    problem_list = metadata_root / "problem_list.csv"
    if not problem_list.is_file():
        raise ValueError(f"missing problem list: {problem_list}")

    with problem_list.open("r", encoding="utf-8-sig", newline="") as handle:
        problem_rows = list(csv.DictReader(handle))
    problem_ids = {
        row.get("id", "").strip() for row in problem_rows if row.get("id", "").strip()
    }
    if not problem_ids:
        raise ValueError("problem_list.csv contains no problem IDs")

    metadata_files = sorted(metadata_root.glob("p*.csv"))
    metadata_files = [path for path in metadata_files if path.name != "problem_list.csv"]
    status_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    accepted_language_counts: Counter[str] = Counter()
    accepted_problem_counts: Counter[str] = Counter()
    accepted_problem_language_pairs: set[tuple[str, str]] = set()
    missing_accepted_files: list[str] = []
    malformed_metadata_files: list[str] = []
    metadata_problem_ids: set[str] = set()
    total_rows = 0
    accepted_rows = 0
    accepted_code_bytes = 0

    for metadata_path in metadata_files:
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            if not REQUIRED_COLUMNS.issubset(columns):
                malformed_metadata_files.append(str(metadata_path))
                continue
            for row in reader:
                total_rows += 1
                problem_id = row["problem_id"].strip()
                language = row["language"].strip() or "<empty>"
                status = row["status"].strip() or "<empty>"
                metadata_problem_ids.add(problem_id)
                status_counts[status] += 1
                language_counts[language] += 1
                if status != "Accepted":
                    continue
                accepted_rows += 1
                accepted_language_counts[language] += 1
                accepted_problem_counts[problem_id] += 1
                accepted_problem_language_pairs.add((problem_id, language))
                try:
                    accepted_code_bytes += max(0, int(row["code_size"]))
                except (TypeError, ValueError):
                    pass
                if verify_accepted_files:
                    source = _source_path(root, row)
                    if not source.is_file() and len(missing_accepted_files) < 1000:
                        missing_accepted_files.append(str(source))

    description_ids = (
        {path.stem for path in description_root.glob("p*.html")}
        if description_root.is_dir()
        else set()
    )
    missing_problem_metadata = sorted(problem_ids - metadata_problem_ids)
    unexpected_problem_metadata = sorted(metadata_problem_ids - problem_ids)

    archive: dict[str, Any] | None = None
    if archive_path is not None:
        archive_path = archive_path.resolve()
        if not archive_path.is_file():
            raise ValueError(f"archive does not exist: {archive_path}")
        archive = {
            "path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        }

    integrity_passed = (
        not malformed_metadata_files
        and not missing_problem_metadata
        and not unexpected_problem_metadata
        and (not verify_accepted_files or not missing_accepted_files)
        and accepted_rows > 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "IBM/Project_CodeNet",
        "dataset_license": "CDLA-Permissive-2.0",
        "dataset_root": str(root),
        "archive": archive,
        "tooling_commit": tooling_commit,
        "verification": {
            "accepted_files_checked": verify_accepted_files,
            "integrity_passed": integrity_passed,
        },
        "counts": {
            "problem_list_rows": len(problem_rows),
            "problem_ids": len(problem_ids),
            "metadata_files": len(metadata_files),
            "metadata_rows": total_rows,
            "accepted_rows": accepted_rows,
            "accepted_code_bytes_from_metadata": accepted_code_bytes,
            "accepted_problem_ids": len(accepted_problem_counts),
            "accepted_problem_language_pairs": len(accepted_problem_language_pairs),
            "problem_descriptions": len(description_ids),
        },
        "status_counts": dict(status_counts.most_common()),
        "language_counts": dict(language_counts.most_common()),
        "accepted_language_counts": dict(accepted_language_counts.most_common()),
        "integrity_findings": {
            "malformed_metadata_files": malformed_metadata_files,
            "missing_problem_metadata": missing_problem_metadata,
            "unexpected_problem_metadata": unexpected_problem_metadata,
            "missing_accepted_files_first_1000": missing_accepted_files,
            "problems_without_descriptions": sorted(problem_ids - description_ids),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--tooling-commit")
    parser.add_argument("--verify-accepted-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_dataset(
        args.dataset_root,
        verify_accepted_files=args.verify_accepted_files,
        archive_path=args.archive,
        tooling_commit=args.tooling_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], sort_keys=True))
    if not report["verification"]["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
