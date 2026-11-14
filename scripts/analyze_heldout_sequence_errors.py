"""Rank held-out teacher-forced sequence failures without changing model state."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

import torch

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

from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.resumable_task_packing import (
    ResumableTaskDataset,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config
from bosscrafts_model.training_checkpoint import load_model_checkpoint_for_inference

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-novel-train-v2"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v2"
    / "checkpoint.pt"
)


class _RankedSequenceError(TypedDict):
    example_id: str
    language: str
    prompt_token_count: int
    target_token_count: int
    mismatch_count: int
    token_accuracy: float
    first_mismatch_offset: int | None
    expected_token_id_at_first_mismatch: int | None
    predicted_token_id_at_first_mismatch: int | None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-per-family", type=int, default=25)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not 1 <= arguments.batch_size <= 32:
        parser.error("batch-size must be between 1 and 32")
    if not 1 <= arguments.top_per_family <= 100:
        parser.error("top-per-family must be between 1 and 100")
    if arguments.max_examples is not None and arguments.max_examples < 1:
        parser.error("max-examples must be positive")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if (
        arguments.dtype == "bfloat16"
        and arguments.device == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        parser.error("CUDA bfloat16 was requested but is unsupported")

    binding = NativeTokenizerBinding.load()
    audit = audit_task_database(binding, arguments.database)
    dataset = ResumableTaskDataset(binding, arguments.database)
    config = bound_native_65k_config(binding)
    examples = list(dataset.iter_examples(split=arguments.split))
    if arguments.max_examples is not None:
        examples = examples[: arguments.max_examples]
    examples.sort(key=lambda example: (len(example["input_ids"]), example["example_id"]))

    device = torch.device(arguments.device)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model = EngineeringModel(config).to(device=device, dtype=dtype).eval()
    state = load_model_checkpoint_for_inference(
        arguments.checkpoint,
        model,
        expected_task_database_identity_sha256=audit.database_identity_sha256,
        expected_binding_sha256=binding.binding_sha256,
        expected_model_fingerprint=config.fingerprint(),
    )

    candidates: dict[str, list[_RankedSequenceError]] = defaultdict(list)
    mismatch_histograms: dict[str, Counter[int]] = defaultdict(Counter)
    scanned_by_family: Counter[str] = Counter()
    exact_by_family: Counter[str] = Counter()
    target_tokens_by_family: Counter[str] = Counter()
    correct_tokens_by_family: Counter[str] = Counter()
    started = time.monotonic()
    processed = 0
    with torch.inference_mode():
        for offset in range(0, len(examples), arguments.batch_size):
            batch = examples[offset : offset + arguments.batch_size]
            maximum_length = max(len(example["input_ids"]) for example in batch)
            ids = torch.zeros(
                (len(batch), maximum_length), dtype=torch.long, device=device
            )
            for row, example in enumerate(batch):
                input_ids = example["input_ids"]
                ids[row, : len(input_ids)] = torch.tensor(
                    input_ids, dtype=torch.long, device=device
                )
            logits, _ = model(ids)
            for row, example in enumerate(batch):
                input_ids = example["input_ids"]
                loss_start = example["loss_start_index"]
                length = len(input_ids)
                predictions = logits[row, loss_start - 1 : length - 1].argmax(dim=-1)
                expected = ids[row, loss_start:length]
                mismatch_mask = predictions.ne(expected)
                mismatches = int(mismatch_mask.sum().item())
                target_count = int(expected.numel())
                family = example["family"]
                scanned_by_family[family] += 1
                mismatch_histograms[family][mismatches] += 1
                target_tokens_by_family[family] += target_count
                correct_tokens_by_family[family] += target_count - mismatches
                if mismatches == 0:
                    exact_by_family[family] += 1
                    first_mismatch_offset = None
                    expected_token_id = None
                    predicted_token_id = None
                else:
                    first_mismatch_offset = int(
                        mismatch_mask.nonzero(as_tuple=False)[0].item()
                    )
                    expected_token_id = int(expected[first_mismatch_offset].item())
                    predicted_token_id = int(predictions[first_mismatch_offset].item())
                item: _RankedSequenceError = {
                    "example_id": example["example_id"],
                    "language": example["language"],
                    "prompt_token_count": loss_start,
                    "target_token_count": target_count,
                    "mismatch_count": mismatches,
                    "token_accuracy": (target_count - mismatches) / target_count,
                    "first_mismatch_offset": first_mismatch_offset,
                    "expected_token_id_at_first_mismatch": expected_token_id,
                    "predicted_token_id_at_first_mismatch": predicted_token_id,
                }
                ranked = candidates[family]
                ranked.append(item)
                ranked.sort(
                    key=lambda value: (
                        value["mismatch_count"],
                        -value["token_accuracy"],
                        value["target_token_count"],
                        value["example_id"],
                    )
                )
                del ranked[arguments.top_per_family :]
            processed += len(batch)
            if processed == len(examples) or processed % 512 < len(batch):
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "total": len(examples),
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    payload = {
        "schema_version": "bosscrafts-heldout-sequence-error-analysis-v2",
        "analysis_complete": processed == len(examples),
        "split": arguments.split,
        "example_count": processed,
        "batch_size": arguments.batch_size,
        "device": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "checkpoint_lineage_verified": True,
        "checkpoint_sha256": state.checkpoint_sha256,
        "checkpoint_training_complete": (
            state.epoch_index >= state.target_epochs
            and state.cursor_after_example_id is None
        ),
        "checkpoint_step": state.step,
        "model_fingerprint": config.fingerprint(),
        "task_database_identity_sha256": audit.database_identity_sha256,
        "scanned_by_family": dict(sorted(scanned_by_family.items())),
        "exact_by_family": dict(sorted(exact_by_family.items())),
        "token_accuracy_by_family": {
            family: correct_tokens_by_family[family] / target_tokens_by_family[family]
            for family in sorted(scanned_by_family)
        },
        "minimum_mismatches_by_family": {
            family: min(mismatch_histograms[family])
            for family in sorted(scanned_by_family)
        },
        "mismatch_histograms_by_family": {
            family: {
                str(count): occurrences
                for count, occurrences in sorted(mismatch_histograms[family].items())
            }
            for family in sorted(scanned_by_family)
        },
        "closest_examples_by_family": {
            family: candidates[family] for family in sorted(candidates)
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(arguments.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
