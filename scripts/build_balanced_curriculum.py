"""Build the deterministic all-language BossCrafts training curriculum."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.balanced_curriculum import build_balanced_curriculum  # noqa: E402
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402


DEFAULT_ADMISSION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-full-resumable-v1"
    / "admission.sqlite"
)
DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "artifacts"
    / "source-catalogs"
    / "legacy-current-source-v1"
    / "corpus.sqlite"
)
DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-balanced-oversampled-compact-v2-v1"
    / "tasks.sqlite"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--max-sequence", type=int, default=2048)
    parser.add_argument(
        "--semantic-wire",
        choices=("compact_v2", "full_v3"),
        default="compact_v2",
    )
    parser.add_argument("--train-per-cell", type=int, default=4096)
    parser.add_argument("--validation-per-cell", type=int, default=256)
    parser.add_argument("--test-per-cell", type=int, default=256)
    arguments = parser.parse_args()
    binding = NativeTokenizerBinding.load()
    status = build_balanced_curriculum(
        binding,
        arguments.admission,
        arguments.catalog,
        arguments.database,
        max_sequence=arguments.max_sequence,
        semantic_wire=arguments.semantic_wire,
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
