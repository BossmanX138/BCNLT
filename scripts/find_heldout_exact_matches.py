"""Find held-out examples whose complete target is greedy-exact under teacher forcing."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

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
    / "bcnlt-current-spec-65536-balanced-curriculum-v1"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v1"
    / "checkpoint.pt"
)


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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not 1 <= arguments.batch_size <= 32:
        parser.error("batch-size must be between 1 and 32")
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

    exact_by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    scanned_by_family: Counter[str] = Counter()
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
                family = example["family"]
                scanned_by_family[family] += 1
                if torch.equal(predictions, expected):
                    exact_by_family[family].append(
                        {
                            "example_id": example["example_id"],
                            "language": example["language"],
                            "prompt_token_count": loss_start,
                            "expected_token_count": length - loss_start,
                        }
                    )
            processed += len(batch)
            if processed == len(examples) or processed % 128 < len(batch):
                elapsed = time.monotonic() - started
                counts = {key: len(value) for key, value in sorted(exact_by_family.items())}
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "total": len(examples),
                            "elapsed_seconds": elapsed,
                            "exact_by_family": counts,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    payload = {
        "schema_version": "bosscrafts-heldout-exact-match-scan-v1",
        "scan_complete": processed == len(examples),
        "split": arguments.split,
        "example_count": processed,
        "batch_size": arguments.batch_size,
        "device": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "checkpoint_lineage_verified": True,
        "checkpoint_training_complete": (
            state.epoch_index >= state.target_epochs
            and state.cursor_after_example_id is None
        ),
        "checkpoint_step": state.step,
        "model_fingerprint": config.fingerprint(),
        "task_database_identity_sha256": audit.database_identity_sha256,
        "scanned_by_family": dict(sorted(scanned_by_family.items())),
        "exact_match_count": sum(len(value) for value in exact_by_family.values()),
        "exact_matches_by_family": {
            family: matches for family, matches in sorted(exact_by_family.items())
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(arguments.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
