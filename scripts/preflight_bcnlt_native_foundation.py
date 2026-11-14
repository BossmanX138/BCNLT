from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bcnlt.native_foundation.preflight import run_source_preflight


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and bind existing BCNLT multilingual and Wizard sources"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--multilingual-report", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--accepted-corpus", required=True)
    parser.add_argument("--english-dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run_source_preflight(
        config_path=args.config,
        multilingual_report_path=args.multilingual_report,
        dataset_manifest_path=args.dataset_manifest,
        accepted_corpus_path=args.accepted_corpus,
        english_dataset_path=args.english_dataset,
        output_path=args.output,
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
