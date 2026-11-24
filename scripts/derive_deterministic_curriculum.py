"""Build the clean, balanced BossCrafts curriculum from a full candidate pool."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.deterministic_curriculum import (  # noqa: E402
    derive_deterministic_curriculum,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402

DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-balanced-oversampled-compact-v2-v1"
    / "tasks.sqlite"
)
DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-deterministic-clean-v1"
    / "tasks.sqlite"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--train-per-cell", type=int, default=2048)
    parser.add_argument("--validation-per-cell", type=int, default=128)
    parser.add_argument("--test-per-cell", type=int, default=128)
    arguments = parser.parse_args()

    status = derive_deterministic_curriculum(
        NativeTokenizerBinding.load(),
        arguments.source,
        arguments.database,
        quotas={
            "train": arguments.train_per_cell,
            "validation": arguments.validation_per_cell,
            "test": arguments.test_per_cell,
        },
    )
    payload = asdict(status)
    payload["database_path"] = str(status.database_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
