"""Build a current-spec, frozen-65K semantic admission corpus from source SQLite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.current_corpus import CurrentCorpusAdmission, SQLiteSourceCatalog
from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "source-catalogs"
    / "legacy-current-source-v1"
    / "corpus.sqlite"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-current-spec-65536-pilot-v1"
)
DEFAULT_DATABASE_SHA256 = "286d92f79f84b75049b845be0270e4308f32056421c8dea92d82d41b4e4003e2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--database-sha256",
        default=DEFAULT_DATABASE_SHA256,
        help="Required SHA-256 identity of the source catalog.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-per-language", type=int, default=3)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process all source documents; never implied by the default pilot run.",
    )
    arguments = parser.parse_args()
    if arguments.full:
        parser.error(
            "--full is not supported by the bounded pilot builder; "
            "use scripts/run_full_resumable_admission.py"
        )
    if not 1 <= arguments.limit_per_language <= 3:
        parser.error("--limit-per-language must be between 1 and 3 for the bounded pilot")
    limit = arguments.limit_per_language
    catalog = SQLiteSourceCatalog(
        arguments.database,
        expected_sha256=arguments.database_sha256,
    )
    binding = NativeTokenizerBinding.load()
    pipeline = BCNLTSourcePipeline(binding)
    result = CurrentCorpusAdmission(
        pipeline, source_catalog_sha256=arguments.database_sha256
    ).materialize(
        catalog.iter_candidates(limit_per_language=limit),
        arguments.output,
    )
    print(
        json.dumps(
            {
                "admitted_count": result.admitted_count,
                "binding_sha256": result.binding_sha256,
                "manifest_sha256": result.manifest_sha256,
                "output_root": str(result.output_root),
                "records_sha256": result.records_sha256,
                "rejected_count": result.rejected_count,
                "semantic_spec_sha256": result.semantic_spec_sha256,
                "tokenizer_sha256": result.tokenizer_sha256,
                "training_ready": result.training_ready,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
