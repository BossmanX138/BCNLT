"""Build a tiny current-lineage database for a full-model GPU training gate."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.tokenizer import NativeTokenizerBinding
from bosscrafts_model.training_gate_database import build_training_gate_database

DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-pilot-tasks-v1"
    / "manifest.json"
)
DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-full-model-gate-v1"
    / "tasks.sqlite"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--max-examples", type=int, default=6)
    arguments = parser.parse_args()
    binding = NativeTokenizerBinding.load()
    status = build_training_gate_database(
        binding,
        arguments.source_manifest,
        arguments.database,
        max_examples=arguments.max_examples,
    )
    print(json.dumps(asdict(status), default=str, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
