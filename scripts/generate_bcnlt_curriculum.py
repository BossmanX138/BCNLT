from __future__ import annotations

import argparse
import json
from pathlib import Path

from bcnlt.adapters.carousel import BCNLTMultilingualAdapter
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine
from bcnlt.corpus.curriculum import generate_document_curriculum
from bcnlt.corpus.database import CorpusDatabase
from bcnlt.semantic_record import records_from_assessments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic BCNLT curriculum rows")
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="unassigned")
    parser.add_argument("--wildcard-memory")
    args = parser.parse_args(argv)

    database = CorpusDatabase(args.database)
    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = TreeSitterSemanticEngine(
        wildcard_memory_root=args.wildcard_memory,
        wildcard_memory_storage="sqlite" if args.wildcard_memory else "jsonl",
    )
    adapter = BCNLTMultilingualAdapter(engine=engine)
    count = 0
    documents = database.documents()
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            for document_index, document in enumerate(documents, start=1):
                parsed = adapter.parse(
                    document.content,
                    language=document.canonical_language,
                    path=document.relative_path,
                    dialect=document.dialect,
                )
                extraction = adapter.extract(parsed)
                units = extraction.units
                records = records_from_assessments(extraction.assessments)
                for item in generate_document_curriculum(
                    document,
                    units,
                    records,
                    split=args.split,
                    assessments=extraction.assessments,
                ):
                    handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
                if document_index % 1_000 == 0:
                    print(
                        json.dumps(
                            {
                                "documents_processed": document_index,
                                "documents_total": len(documents),
                                "records_written": count,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        engine.close()
        database.close()
    print(json.dumps({"records_written": count, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
