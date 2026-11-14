"""Count semantic records for sources rejected by the sidecar safety ceiling."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "vendor" / "bcnlt_runtime_v6_candidate"
sys.path.insert(0, str(RUNTIME_ROOT / "site-packages"))
sys.path.insert(0, str(RUNTIME_ROOT / "source"))
os.environ["BCNLT_PARSER_CACHE"] = str(RUNTIME_ROOT / "grammars")

from bcnlt.adapters import BCNLTMultilingualAdapter, LanguageCarousel  # noqa: E402
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine  # noqa: E402
from bcnlt.canonical import canonical_record_bytes  # noqa: E402
from bcnlt.semantic_record import records_from_assessments  # noqa: E402

DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "artifacts"
    / "source-catalogs"
    / "legacy-current-source-v1"
    / "corpus.sqlite"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("relative_paths", nargs="*")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--admission", type=Path)
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
    if not relative_paths:
        parser.error("provide relative paths or --admission")

    connection = sqlite3.connect(arguments.catalog)
    try:
        rows = []
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

    carousel = LanguageCarousel.default()
    adapter = BCNLTMultilingualAdapter(
        engine=TreeSitterSemanticEngine(),
        carousel=carousel,
    )
    results = []
    for relative_path, language, source, dialect in rows:
        started = time.perf_counter()
        document = adapter.parse(
            source,
            language=language,
            path=relative_path,
            dialect=str(dialect) if dialect is not None else None,
        )
        extraction = adapter.extract(document)
        records = records_from_assessments(extraction.assessments)
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
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
