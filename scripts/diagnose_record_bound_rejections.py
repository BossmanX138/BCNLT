"""Count semantic records for sources rejected by the sidecar safety ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT / "site-packages"))
sys.path.insert(0, str(RUNTIME_ROOT / "source"))
os.environ["BCNLT_PARSER_CACHE"] = str(RUNTIME_ROOT / "grammars")

from bcnlt.adapters import BCNLTMultilingualAdapter, LanguageCarousel  # noqa: E402
from bcnlt.adapters.tree_sitter import (  # noqa: E402
    TreeSitterSemanticEngine,
    _default_parser_provider,
)
from bcnlt.canonical import canonical_record_bytes  # noqa: E402
from bcnlt.semantic_record import records_from_assessments  # noqa: E402

from bosscrafts_model.category_case_sources import (  # noqa: E402
    MCEVAL_LANGUAGE_FILES,
    _mceval_harness,
    _read_mceval_manifest,
)
from bosscrafts_model.immutable_artifacts import publish_new_files  # noqa: E402

DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "artifacts"
    / "source-catalogs"
    / "legacy-current-source-v1"
    / "corpus.sqlite"
)


def _parse_diagnostics(source: str, language: str) -> list[dict[str, object]]:
    tree = _default_parser_provider(language).parse(source.encode("utf-8"))
    diagnostics: list[dict[str, object]] = []
    stack = [tree.root_node]
    source_bytes = source.encode("utf-8")
    while stack:
        node = stack.pop()
        node_type = str(getattr(node, "type", ""))
        is_missing = bool(getattr(node, "is_missing", False))
        if node_type in {"ERROR", "MISSING"} or is_missing:
            start_row, start_column = node.start_point
            end_row, end_column = node.end_point
            start = max(0, int(node.start_byte) - 80)
            end = min(len(source_bytes), int(node.end_byte) + 80)
            diagnostics.append(
                {
                    "node_type": node_type,
                    "is_missing": is_missing,
                    "span": [
                        start_row + 1,
                        start_column,
                        end_row + 1,
                        end_column,
                    ],
                    "snippet": source_bytes[start:end].decode(
                        "utf-8", errors="replace"
                    ),
                }
            )
        stack.extend(reversed(tuple(getattr(node, "children", ()))))
    return diagnostics[:20]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("relative_paths", nargs="*")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--probe-observations", type=Path)
    parser.add_argument("--mceval-source-root", type=Path)
    parser.add_argument("--mceval-source-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    relative_paths = list(arguments.relative_paths)
    if arguments.admission is not None:
        admission = sqlite3.connect(arguments.admission)
        try:
            relative_paths.extend(
                str(row[0])
                for row in admission.execute(
                    "SELECT relative_path FROM documents WHERE status='rejected' "
                    "ORDER BY language, relative_path"
                )
            )
        finally:
            admission.close()
    relative_paths = list(dict.fromkeys(relative_paths))
    mceval_rows: list[tuple[str, str, str, None]] = []
    if arguments.probe_observations is not None:
        if (
            arguments.mceval_source_root is None
            or arguments.mceval_source_manifest is None
        ):
            parser.error(
                "--probe-observations requires --mceval-source-root and "
                "--mceval-source-manifest"
            )
        rejected_ids: set[str] = set()
        for line in arguments.probe_observations.read_text(
            encoding="utf-8"
        ).splitlines():
            observation = json.loads(line)
            if observation.get("status") == "preprocessing_rejected":
                rejected_ids.add(str(observation["task_id"]))
        manifest, _ = _read_mceval_manifest(arguments.mceval_source_manifest)
        language_manifest = manifest["languages"]
        if type(language_manifest) is not dict:
            raise AssertionError("validated McEval manifest is invalid")
        for language, filename in MCEVAL_LANGUAGE_FILES.items():
            source_file = arguments.mceval_source_root / filename
            entry = language_manifest[language]
            if type(entry) is not dict:
                raise AssertionError("validated McEval language entry is invalid")
            content = source_file.read_bytes()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"McEval {language} source SHA-256 mismatch")
            for line in content.decode("utf-8").splitlines():
                row = json.loads(line)
                task_id = str(row["task_id"])
                if task_id not in rejected_ids:
                    continue
                prefix, suffix = _mceval_harness(
                    language,
                    str(row["prompt"]),
                    str(row["test"]),
                )
                mceval_rows.append(
                    (
                        f"mceval/{language}/{task_id}",
                        language,
                        prefix + str(row["canonical_solution"]) + suffix,
                        None,
                    )
                )
        if {"/".join(path.split("/")[-2:]) for path, *_ in mceval_rows} != rejected_ids:
            raise ValueError("McEval rejected-task inventory is incomplete")
    if not relative_paths and not mceval_rows:
        parser.error("provide relative paths, --admission, or --probe-observations")

    rows: list[tuple[str, str, str, object]] = []
    if relative_paths:
        connection = sqlite3.connect(arguments.catalog)
        try:
            for relative_path in relative_paths:
                row = connection.execute(
                    "SELECT canonical_language, content, dialect FROM documents "
                    "WHERE relative_path=?",
                    (relative_path,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"catalog path not found: {relative_path}")
                rows.append((relative_path, str(row[0]), str(row[1]), row[2]))
        finally:
            connection.close()
    rows.extend(mceval_rows)

    carousel = LanguageCarousel.default()
    adapter = BCNLTMultilingualAdapter(
        engine=TreeSitterSemanticEngine(),
        carousel=carousel,
    )
    results = []
    for relative_path, language, source, dialect in rows:
        started = time.perf_counter()
        try:
            document = adapter.parse(
                source,
                language=language,
                path=relative_path,
                dialect=str(dialect) if dialect is not None else None,
            )
            extraction = adapter.extract(document)
            records = records_from_assessments(extraction.assessments)
        except ValueError as error:
            results.append(
                {
                    "relative_path": relative_path,
                    "language": language,
                    "classification": "parser_rejection",
                    "error_type": type(error).__name__,
                    "detail": " ".join(str(error).split())[:512],
                    "parse_diagnostics": _parse_diagnostics(source, language),
                    "seconds": time.perf_counter() - started,
                }
            )
            continue
        canonical_bytes = sum(len(canonical_record_bytes(record)) for record in records)
        invalid_history_items = []
        maximum_history_chars = 0
        for record_index, record in enumerate(records):
            for history_index, item in enumerate(record.recent_history):
                maximum_history_chars = max(maximum_history_chars, len(item))
                if (
                    len(item) > 512
                    or "\r" in item
                    or unicodedata.normalize("NFC", item) != item
                ):
                    invalid_history_items.append(
                        {
                            "record_index": record_index,
                            "history_index": history_index,
                            "characters": len(item),
                            "has_carriage_return": "\r" in item,
                            "is_nfc": unicodedata.normalize("NFC", item) == item,
                        }
                    )
        if invalid_history_items:
            classification = "history_bound_defect"
        elif len(records) > 16_384 or canonical_bytes > 16 * 1024 * 1024:
            classification = "safety_bound"
        else:
            classification = "admissible"
        results.append(
            {
                "relative_path": relative_path,
                "language": language,
                "semantic_record_count": len(records),
                "canonical_bytes": canonical_bytes,
                "exceeds_record_limit": len(records) > 16_384,
                "exceeds_canonical_byte_limit": canonical_bytes > 16 * 1024 * 1024,
                "maximum_history_chars": maximum_history_chars,
                "invalid_history_item_count": len(invalid_history_items),
                "invalid_history_examples": invalid_history_items[:10],
                "classification": classification,
                "seconds": time.perf_counter() - started,
            }
        )
    output = json.dumps(results, indent=2, sort_keys=True)
    if arguments.output is not None:
        publish_new_files(
            ((arguments.output.resolve(), (output + "\n").encode("utf-8")),),
            artifact_label="record-bound rejection diagnostics",
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
