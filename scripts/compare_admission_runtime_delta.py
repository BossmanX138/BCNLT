"""Publish an immutable per-document admission runtime delta receipt."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.admission_runtime_delta import (  # noqa: E402
    compare_admission_runtime_delta,
    publish_delta_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-database", type=Path, required=True)
    parser.add_argument("--candidate-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    result = compare_admission_runtime_delta(
        arguments.baseline_database,
        arguments.candidate_database,
    )
    manifest_sha256 = publish_delta_receipt(result, arguments.output)
    summary = asdict(result)
    summary.pop("documents")
    print(
        json.dumps(
            {**summary, "manifest_sha256": manifest_sha256},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
