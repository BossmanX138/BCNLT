"""Inspect empty semantic surfaces in catalog documents without executing source."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT_ROOT / "vendor" / "bcnlt_runtime_v6_candidate"
sys.path.insert(0, str(RUNTIME / "site-packages"))
sys.path.insert(0, str(RUNTIME / "source"))
os.environ["BCNLT_PARSER_CACHE"] = str(RUNTIME / "grammars")

from bcnlt.adapters import BCNLTMultilingualAdapter, LanguageCarousel  # noqa: E402
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine  # noqa: E402
from bcnlt.canonical import canonical_record_bytes  # noqa: E402
from bcnlt.semantic_record import records_from_assessments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("relative_paths", nargs="+")
    arguments = parser.parse_args()
    connection = sqlite3.connect(arguments.database)
    carousel = LanguageCarousel.default()
    adapter = BCNLTMultilingualAdapter(
        engine=TreeSitterSemanticEngine(),
        carousel=carousel,
    )
    results: list[dict[str, object]] = []
    try:
        for relative_path in arguments.relative_paths:
            row = connection.execute(
                "SELECT canonical_language, content, dialect FROM documents "
                "WHERE relative_path=?",
                (relative_path,),
            ).fetchone()
            if row is None:
                raise ValueError(f"catalog path not found: {relative_path}")
            language, source, dialect = row
            document = adapter.parse(
                str(source),
                language=str(language),
                path=relative_path,
                dialect=str(dialect) if dialect is not None else None,
            )
            extraction = adapter.extract(document)
            failures = []
            source_lines = str(source).splitlines()
            for assessment in extraction.assessments:
                unit = assessment.unit
                if unit.surface:
                    continue
                start_line = unit.span[0]
                failures.append(
                    {
                        "ast_relation": unit.ast_relation,
                        "event": str(unit.event),
                        "ir_kind": unit.ir.kind.value if unit.ir is not None else None,
                        "ir_name": unit.ir.name if unit.ir is not None else None,
                        "metadata": unit.metadata,
                        "scope": unit.scope,
                        "span": unit.span,
                        "source_line": (
                            source_lines[start_line - 1]
                            if 0 < start_line <= len(source_lines)
                            else None
                        ),
                    }
                )
            conversion_error = None
            canonical_bytes = 0
            record_count = 0
            try:
                records = records_from_assessments(extraction.assessments)
                record_count = len(records)
                canonical_bytes = sum(len(canonical_record_bytes(item)) for item in records)
            except Exception as error:
                conversion_error = f"{type(error).__name__}: {error}"
            results.append(
                {
                    "language": language,
                    "relative_path": relative_path,
                    "empty_surface_count": len(failures),
                    "record_count": record_count,
                    "canonical_bytes": canonical_bytes,
                    "conversion_error": conversion_error,
                    "failures": failures,
                }
            )
    finally:
        connection.close()
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
