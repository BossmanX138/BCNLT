"""Build a deterministic, accepted-only multilingual Project CodeNet pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bosscrafts_model.codex_code_curriculum import HumanEvalExclusion
from bosscrafts_model.project_codenet_pilot import (
    build_pilot,
    publish_pilot_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--tooling-commit", required=True)
    parser.add_argument("--humaneval", required=True, type=Path)
    parser.add_argument("--language", action="append", required=True)
    parser.add_argument("--per-language-limit", type=int, default=1000)
    parser.add_argument("--minimum-code-bytes", type=int, default=64)
    parser.add_argument("--maximum-code-bytes", type=int, default=16384)
    parser.add_argument("--seed", default="bosscrafts-project-codenet-pilot-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclusion = HumanEvalExclusion(args.humaneval)
    records, report = build_pilot(
        args.dataset_root.resolve(),
        languages=set(args.language),
        per_language_limit=args.per_language_limit,
        seed=args.seed,
        minimum_code_bytes=args.minimum_code_bytes,
        maximum_code_bytes=args.maximum_code_bytes,
        archive_sha256=args.archive_sha256.lower(),
        tooling_commit=args.tooling_commit,
        overlap_filter=exclusion,
    )
    publish_pilot_artifacts(
        args.output.resolve(),
        args.report.resolve(),
        records,
        report,
    )
    print(json.dumps({"records": len(records), **report["language_counts"]}))


if __name__ == "__main__":
    main()
