"""Acquire an immutable McEval generation snapshot for evaluation-only use."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.mceval_source_acquisition import (  # noqa: E402
    acquire_mceval_generation_sources,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = acquire_mceval_generation_sources(
        arguments.output,
        revision=arguments.revision,
    )
    print(
        json.dumps(
            {**asdict(result), "output_root": str(result.output_root)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
