"""Build a bounded current-spec admission corpus using held-out documents only."""
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
from bosscrafts_model.heldout_source_admission import select_heldout_candidates
from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_DATABASE = PROJECT_ROOT / "artifacts" / "source-catalogs" / "legacy-current-source-v1" / "corpus.sqlite"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-v6-65536-heldout-admission-v2"
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
    parser.add_argument("--candidates-per-language", type=int, default=12)
    parser.add_argument(
        "--persistent-sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    arguments = parser.parse_args()
    catalog = SQLiteSourceCatalog(
        arguments.database, expected_sha256=arguments.database_sha256
    )
    binding = NativeTokenizerBinding.load()
    pipeline = BCNLTSourcePipeline(
        binding,
        persistent_sidecars=arguments.persistent_sidecars,
    )
    try:
        candidates = select_heldout_candidates(
            catalog.iter_candidates(limit_per_language=None),
            languages=set(pipeline.supported_languages),
            per_language=arguments.candidates_per_language,
        )
        result = CurrentCorpusAdmission(
            pipeline, source_catalog_sha256=arguments.database_sha256
        ).materialize(candidates, arguments.output)
    finally:
        pipeline.close()
    print(
        json.dumps(
            {
                "admitted_count": result.admitted_count,
                "manifest_sha256": result.manifest_sha256,
                "output_root": str(result.output_root),
                "records_sha256": result.records_sha256,
                "rejected_count": result.rejected_count,
                "training_ready": result.training_ready,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
