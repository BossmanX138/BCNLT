"""List deterministic held-out cases reachable by the bounded inference selector."""
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

from bosscrafts_model.inference_protocol import (
    INFERENCE_TASKS,
    select_heldout_inference_case,
)
from bosscrafts_model.resumable_task_packing import (
    ResumableTaskDataset,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config


DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-balanced-curriculum-v1"
    / "tasks.sqlite"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--maximum-cap", type=int, default=256)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.maximum_cap <= 256:
        parser.error("maximum-cap must be between 1 and 256")

    binding = NativeTokenizerBinding.load()
    audit = audit_task_database(binding, arguments.database)
    dataset = ResumableTaskDataset(binding, arguments.database)
    config = bound_native_65k_config(binding)
    examples = tuple(dataset.iter_examples(split=arguments.split))

    candidates: dict[str, list[dict[str, object]]] = {}
    for family in INFERENCE_TASKS[1:]:
        family_candidates: list[dict[str, object]] = []
        previous_example_id: str | None = None
        for cap in range(1, arguments.maximum_cap + 1):
            try:
                case = select_heldout_inference_case(
                    examples,
                    family=family,
                    max_new_tokens=cap,
                    maximum_sequence=config.max_sequence,
                )
            except ValueError as error:
                if str(error) == "no eligible held-out example fits the inference bounds":
                    continue
                raise
            if case.example_id == previous_example_id:
                continue
            previous_example_id = case.example_id
            family_candidates.append(
                {
                    "example_id": case.example_id,
                    "expected_token_count": len(case.expected_ids),
                    "minimum_max_new_tokens": cap,
                    "prompt_token_count": len(case.prompt_ids),
                }
            )
        candidates[family] = family_candidates

    payload = {
        "database": str(arguments.database.resolve()),
        "task_database_identity_sha256": audit.database_identity_sha256,
        "split": arguments.split,
        "maximum_cap": arguments.maximum_cap,
        "model_maximum_sequence": config.max_sequence,
        "candidates": candidates,
    }
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
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
