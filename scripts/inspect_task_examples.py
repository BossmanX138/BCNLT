"""Decode selected lineage-verified task examples without changing artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


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

from bosscrafts_model.resumable_task_packing import (
    ResumableTaskDataset,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-novel-train-v2"
    / "tasks.sqlite"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--example-id", action="append", required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    requested = set(arguments.example_id)
    if any(
        len(example_id) != 64
        or any(character not in "0123456789abcdef" for character in example_id)
        for example_id in requested
    ):
        parser.error("every example-id must be a lowercase SHA-256")

    binding = NativeTokenizerBinding.load()
    audit = audit_task_database(binding, arguments.database)
    found: dict[str, dict[str, object]] = {}
    dataset = ResumableTaskDataset(binding, arguments.database)
    for example in dataset.iter_examples():
        example_id = str(example["example_id"])
        if example_id not in requested:
            continue
        ids = list(example["input_ids"])
        loss_start = int(example["loss_start_index"])
        target_ids = ids[loss_start:]
        found[example_id] = {
            "example_id": example_id,
            "document_id": example["document_id"],
            "language": example["language"],
            "family": example["family"],
            "record_index": example["record_index"],
            "split": example["split"],
            "prompt_token_count": loss_start,
            "target_token_count": len(target_ids),
            "prompt_text": binding.decode(ids[:loss_start]),
            "target_text": binding.decode(target_ids),
            "target_token_ids": target_ids,
            "source_sha256": example["source_sha256"],
        }
    missing = sorted(requested - set(found))
    if missing:
        raise ValueError("requested task examples were not found: " + ",".join(missing))
    payload = {
        "schema_version": "bosscrafts-task-example-inspection-v1",
        "database": str(arguments.database.resolve()),
        "task_database_identity_sha256": audit.database_identity_sha256,
        "examples": [found[example_id] for example_id in arguments.example_id],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if arguments.output is not None:
        destination = arguments.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(destination)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
