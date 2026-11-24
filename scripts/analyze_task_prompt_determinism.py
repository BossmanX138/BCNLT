"""Audit global prompt determinism and held-out isolation for every task family."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.task_determinism import (  # noqa: E402
    audit_task_prompt_determinism,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-deterministic-clean-v1"
    / "tasks.sqlite"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluations"
    / "task-prompt-determinism-v7.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-limit", type=int, default=20)
    arguments = parser.parse_args()

    binding = NativeTokenizerBinding.load()
    payload = audit_task_prompt_determinism(
        binding,
        arguments.database,
        sample_limit=arguments.sample_limit,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    destination = arguments.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
