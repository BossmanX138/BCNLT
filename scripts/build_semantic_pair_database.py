from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bcnlt.semantic_db import build_database, export_training_jsonl
from bcnlt.languages import CANONICAL_LANGUAGE_IDS


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the versioned BCNLT semantic-pair database.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--database", type=Path, default=Path("data/bcnlt_semantic_pairs.sqlite3"))
    parser.add_argument("--export", type=Path, default=Path("data/exports/bcnlt_semantic_pairs.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.json"))
    parser.add_argument("--holdout-list", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--language", choices=CANONICAL_LANGUAGE_IDS, required=True)
    parser.add_argument("--dialect")
    args = parser.parse_args()

    root = args.root.resolve()
    holdouts: list[Path] = []
    if args.holdout_list and args.holdout_list.exists():
        holdouts = [
            Path(line.strip()).resolve()
            for line in args.holdout_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    stats = build_database(
        root, args.database.resolve(), holdouts, args.observations,
        language=args.language, dialect=args.dialect,
    )
    exported = export_training_jsonl(args.database.resolve(), args.export.resolve())
    database_sha = hashlib.sha256(args.database.resolve().read_bytes()).hexdigest()
    with sqlite3.connect(args.database.resolve()) as connection:
        holdout_pairs = connection.execute(
            "SELECT COUNT(*) FROM semantic_pairs WHERE split='evaluation_holdout'"
        ).fetchone()[0]
        eligible_pairs = connection.execute(
            "SELECT COUNT(*) FROM semantic_pairs WHERE training_eligible=1"
        ).fetchone()[0]
        lineage = connection.execute(
            "SELECT parser_bundle_sha256, semantic_spec_sha256 FROM documents "
            "WHERE language = ? AND dialect = ? LIMIT 1",
            (args.language, args.dialect or ""),
        ).fetchone()
    manifest = {
        "schema_version": 2,
        "language": args.language,
        "dialect": args.dialect,
        "parser_bundle_sha256": lineage[0] if lineage else None,
        "semantic_spec_sha256": lineage[1] if lineage else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": stats.run_id,
        "database": str(args.database.resolve()),
        "database_sha256": database_sha,
        "export": str(args.export.resolve()),
        "documents": stats.documents,
        "semantic_units": stats.semantic_units,
        "semantic_pairs": stats.semantic_pairs,
        "exported_pairs": exported,
        "evaluation_holdout_pairs": holdout_pairs,
        "training_eligible_pairs": eligible_pairs,
        "model_observations": stats.observations,
        "admission_policy": "Pairs remain unreviewed and ineligible until an explicit review marks them eligible.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
