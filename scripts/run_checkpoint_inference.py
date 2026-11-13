"""Run bounded greedy inference from a lineage-verified BossCrafts checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
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

from bosscrafts_model.artifact_schemas import CHECKPOINT_INFERENCE_SCHEMA
from bosscrafts_model.config import ModelConfig
from bosscrafts_model.inference_protocol import (
    INFERENCE_TASKS,
    build_inference_prompt,
    response_payload_token_ids,
    response_protocol_complete,
    select_heldout_inference_case,
    task_end_token_id,
)
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
    / "bcnlt-current-spec-65536-full-resumable-tasks-v1"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bounded-training-trial-v1"
    / "checkpoint.pt"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--checkpoint-database",
        type=Path,
        help="training database recorded by the checkpoint; defaults to --database",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--full-model", action="store_true")
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt")
    prompt_source.add_argument("--heldout-family", choices=INFERENCE_TASKS[1:])
    parser.add_argument("--heldout-split", choices=("validation", "test"), default="test")
    parser.add_argument("--task", choices=INFERENCE_TASKS)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.max_new_tokens <= 256:
        parser.error("max-new-tokens must be between 1 and 256")
    if arguments.prompt is not None and not arguments.prompt:
        parser.error("prompt must not be empty")
    if (
        arguments.heldout_family is not None
        and arguments.task is not None
        and arguments.task != arguments.heldout_family
    ):
        parser.error("held-out family and task must match")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if arguments.dtype == "bfloat16" and arguments.device == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("CUDA bfloat16 was requested but is unsupported")

    binding = NativeTokenizerBinding.load()
    task_audit = audit_task_database(binding, arguments.database)
    checkpoint_task_audit = (
        audit_task_database(binding, arguments.checkpoint_database)
        if arguments.checkpoint_database is not None
        else task_audit
    )
    dataset = ResumableTaskDataset(binding, arguments.database)
    if arguments.full_model:
        config = bound_native_65k_config(binding)
    else:
        config = ModelConfig.tiny(
            vocab_size=binding.vocab_size,
            tokenizer_manifest=binding.binding_sha256,
        )
    device = torch.device(arguments.device)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model = EngineeringModel(config).to(device=device, dtype=dtype).eval()
    state = load_model_checkpoint_for_inference(
        arguments.checkpoint,
        model,
        expected_task_database_identity_sha256=(
            checkpoint_task_audit.database_identity_sha256
        ),
        expected_binding_sha256=binding.binding_sha256,
        expected_model_fingerprint=config.fingerprint(),
    )
    task = arguments.heldout_family or arguments.task or "raw"
    heldout_case = None
    if arguments.heldout_family is not None:
        heldout_case = select_heldout_inference_case(
            dataset.iter_examples(split=arguments.heldout_split),
            family=arguments.heldout_family,
            max_new_tokens=arguments.max_new_tokens,
            maximum_sequence=config.max_sequence,
        )
        prompt_ids = list(heldout_case.prompt_ids)
    else:
        if arguments.prompt is None:
            raise RuntimeError("validated manual inference prompt is missing")
        prompt_ids = build_inference_prompt(
            binding,
            task=task,
            prompt=arguments.prompt,
        )
    if len(prompt_ids) + arguments.max_new_tokens > config.max_sequence:
        parser.error("prompt plus generated tokens exceeds the model context")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        ids,
        max_new_tokens=arguments.max_new_tokens,
        eos_token_id=task_end_token_id(binding, task),
    )
    full_ids = generated[0].tolist()
    new_ids = full_ids[len(prompt_ids) :]
    response_ids = response_payload_token_ids(binding, task, new_ids)
    expected_ids = list(heldout_case.expected_ids) if heldout_case is not None else None
    heldout_accuracy = None
    if expected_ids is not None:
        heldout_correct = sum(
            actual == expected
            for actual, expected in zip(new_ids, expected_ids, strict=False)
        )
        heldout_accuracy = heldout_correct / len(expected_ids)
    payload = {
        "schema_version": CHECKPOINT_INFERENCE_SCHEMA,
        "checkpoint_lineage_verified": True,
        "checkpoint_task_database_identity_sha256": (
            checkpoint_task_audit.database_identity_sha256
        ),
        "checkpoint_epoch_index": state.epoch_index,
        "checkpoint_step": state.step,
        "checkpoint_target_epochs": state.target_epochs,
        "checkpoint_training_complete": (
            state.epoch_index >= state.target_epochs
            and state.cursor_after_example_id is None
        ),
        "device": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "full_model": arguments.full_model,
        "full_text": binding.decode(full_ids),
        "generated_protocol_text": binding.decode(new_ids),
        "generated_text": binding.decode(response_ids),
        "heldout_exact_match": (
            new_ids == expected_ids if expected_ids is not None else None
        ),
        "heldout_example_id": (
            heldout_case.example_id if heldout_case is not None else None
        ),
        "heldout_expected_target_token_count": (
            len(expected_ids) if expected_ids is not None else None
        ),
        "heldout_expected_token_ids": expected_ids,
        "heldout_split": (
            arguments.heldout_split if heldout_case is not None else None
        ),
        "heldout_target_token_accuracy": (
            heldout_accuracy
        ),
        "model_fingerprint": config.fingerprint(),
        "new_token_count": len(new_ids),
        "new_token_ids": new_ids,
        "prompt_token_count": len(prompt_ids),
        "response_protocol_complete": response_protocol_complete(
            binding, task=task, generated_ids=new_ids
        ),
        "prompt_source": "heldout_database" if heldout_case is not None else "manual",
        "task": task,
        "evaluation_task_database_identity_sha256": (
            task_audit.database_identity_sha256
        ),
        "task_database_identity_sha256": task_audit.database_identity_sha256,
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
