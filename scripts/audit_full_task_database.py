"""Audit a completed packed task database and write its content-bound manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PINNED_TOKENIZERS_SITE = PROJECT_ROOT / "vendor" / "tokenizers_runtime_0_23"
if not (PINNED_TOKENIZERS_SITE / "tokenizers" / "tokenizers.pyd").is_file():
    raise RuntimeError("project-contained tokenizers 0.23 runtime is missing")
if "tokenizers" in sys.modules:
    raise RuntimeError("tokenizers was imported before the pinned runtime was selected")
sys.path.insert(0, str(PINNED_TOKENIZERS_SITE))

import tokenizers

if tokenizers.__version__ != "0.23.2":
    raise RuntimeError("project-contained tokenizers runtime version mismatch")

from bosscrafts_model.task_database_audit_manifest import (
    materialize_task_database_audit_manifest,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-full-resumable-tasks-v1"
    / "tasks.sqlite"
)
DEFAULT_OUTPUT = DEFAULT_DATABASE.with_name("audit-manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    materialize_task_database_audit_manifest(
        NativeTokenizerBinding.load(), arguments.database, arguments.output
    )
    payload = json.loads(arguments.output.resolve().read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
