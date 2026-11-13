"""Generate lineage-recorded OpenAI HumanEval samples from a BossCrafts checkpoint.

This command only generates samples.  It never executes model-produced code.  The
resulting ``samples.jsonl`` is intended for a separately isolated evaluator.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypedDict

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

from bosscrafts_model.generation_artifacts import (
    HUMANEVAL_GENERATION_SCHEMA,
    HumanEvalGenerationRecord,
    clean_python_completion,
    load_humaneval_generation_records,
    validate_humaneval_generation_records,
)
from bosscrafts_model.inference_protocol import (
    build_inference_prompt,
    response_payload_token_ids,
    response_protocol_complete,
    task_end_token_id,
)
from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.resumable_task_packing import audit_task_database
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config
from bosscrafts_model.training_checkpoint import load_model_checkpoint_for_inference

SCHEMA = HUMANEVAL_GENERATION_SCHEMA
OPENAI_HUMANEVAL_SHA256 = (
    "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
)
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "artifacts"
    / "benchmarks"
    / "openai-humaneval-v1"
    / "source"
    / "HumanEval.jsonl.gz"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v8-consolidation"
    / "checkpoint.pt"
)
DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-balanced-curriculum-clean-v5"
    / "tasks.sqlite"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "benchmarks"
    / "openai-humaneval-v1"
    / "v8-consolidation-greedy"
)


class _HumanEvalTask(TypedDict):
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_dataset(path: Path) -> list[_HumanEvalTask]:
    tasks: list[_HumanEvalTask] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            required = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}
            if type(task) is not dict or not required.issubset(task):
                raise ValueError(f"invalid HumanEval row {line_number}")
            if any(type(task[name]) is not str for name in required):
                raise ValueError(f"non-string HumanEval field on row {line_number}")
            tasks.append(
                _HumanEvalTask(
                    task_id=task["task_id"],
                    prompt=task["prompt"],
                    canonical_solution=task["canonical_solution"],
                    test=task["test"],
                    entry_point=task["entry_point"],
                )
            )
    expected_ids = [f"HumanEval/{index}" for index in range(164)]
    if [task["task_id"] for task in tasks] != expected_ids:
        raise ValueError("HumanEval task IDs do not match the official 164-task order")
    return tasks


def _write_samples(
    path: Path,
    records: list[HumanEvalGenerationRecord],
    field: Literal["raw_completion", "clean_completion"],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {"task_id": record["task_id"], "completion": record[field]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=164)
    parser.add_argument(
        "--prompt-protocol",
        choices=("raw", "instruction_to_code"),
        default="raw",
        help=(
            "raw feeds the official prompt without BCNLT framing; "
            "instruction_to_code uses the trained mixed-instruction protocol"
        ),
    )
    arguments = parser.parse_args()

    if not 1 <= arguments.max_new_tokens <= 512:
        parser.error("max-new-tokens must be between 1 and 512")
    if not 1 <= arguments.limit <= 164:
        parser.error("limit must be between 1 and 164")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if (
        arguments.device == "cuda"
        and arguments.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        parser.error("CUDA bfloat16 was requested but is unsupported")

    dataset = arguments.dataset.resolve()
    checkpoint = arguments.checkpoint.resolve()
    database = arguments.database.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = _sha256(dataset)
    if dataset_sha256 != OPENAI_HUMANEVAL_SHA256:
        raise ValueError("HumanEval dataset SHA-256 does not match the pinned official file")
    tasks = _read_dataset(dataset)[: arguments.limit]

    binding = NativeTokenizerBinding.load()
    database_audit = audit_task_database(binding, database)
    config = bound_native_65k_config(binding)
    checkpoint_sha256 = _sha256(checkpoint)
    run_spec = {
        "schema_version": SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "database_identity_sha256": database_audit.database_identity_sha256,
        "device": arguments.device,
        "dtype": arguments.dtype,
        "limit": arguments.limit,
        "max_new_tokens": arguments.max_new_tokens,
        "model_fingerprint": config.fingerprint(),
        "tokenizer_sha256": binding.tokenizer_sha256,
        "binding_sha256": binding.binding_sha256,
        "generation": "greedy_argmax",
        "prompt_protocol": arguments.prompt_protocol,
    }
    run_identity = hashlib.sha256(
        json.dumps(run_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_spec["run_identity_sha256"] = run_identity
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != run_spec:
            raise ValueError("output directory already belongs to another benchmark run")
    else:
        _atomic_json(manifest_path, run_spec)

    records_path = output / "generation-records.jsonl"
    records = load_humaneval_generation_records(records_path, run_identity)
    validate_humaneval_generation_records(
        binding,
        tasks,
        records,
        prompt_protocol=arguments.prompt_protocol,
        maximum_new_tokens=arguments.max_new_tokens,
        maximum_sequence=config.max_sequence,
    )

    device = torch.device(arguments.device)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model = EngineeringModel(config).to(device=device, dtype=dtype).eval()
    state = load_model_checkpoint_for_inference(
        checkpoint,
        model,
        expected_task_database_identity_sha256=database_audit.database_identity_sha256,
        expected_binding_sha256=binding.binding_sha256,
        expected_model_fingerprint=config.fingerprint(),
    )

    for index, task in enumerate(tasks):
        task_id = str(task["task_id"])
        if task_id in records:
            continue
        prompt = str(task["prompt"])
        eos_token_id: int | None
        if arguments.prompt_protocol == "raw":
            prompt_ids = binding.encode_payload(prompt)
            eos_token_id = 2
        else:
            prompt_ids = build_inference_prompt(
                binding,
                task="instruction_to_code",
                prompt=prompt,
            )
            eos_token_id = task_end_token_id(binding, "instruction_to_code")
            if eos_token_id is None:
                raise AssertionError("instruction protocol must have a task-end token")
        available = config.max_sequence - len(prompt_ids)
        if available < 1:
            raise ValueError(f"{task_id} prompt exceeds the model context")
        max_new_tokens = min(arguments.max_new_tokens, available)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        generated = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        generated_ids = generated[0, len(prompt_ids) :].tolist()
        if arguments.prompt_protocol == "raw":
            protocol_complete = True
            completion_ids = list(generated_ids)
            if completion_ids[-1:] == [2]:
                completion_ids = completion_ids[:-1]
        else:
            protocol_complete = response_protocol_complete(
                binding,
                task="instruction_to_code",
                generated_ids=generated_ids,
            )
            completion_ids = response_payload_token_ids(
                binding,
                "instruction_to_code",
                generated_ids,
            )
        raw_model_response = binding.decode(generated_ids)
        raw_completion = binding.decode(completion_ids)
        record = HumanEvalGenerationRecord(
            schema_version=SCHEMA,
            run_identity_sha256=run_identity,
            task_index=index,
            task_id=task_id,
            entry_point=task["entry_point"],
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            prompt_token_count=len(prompt_ids),
            generated_token_count=len(generated_ids),
            completion_token_count=len(completion_ids),
            generated_ids=generated_ids,
            completion_ids=completion_ids,
            generation_seconds=elapsed,
            tokens_per_second=len(generated_ids) / elapsed,
            response_protocol_complete=protocol_complete,
            raw_model_response=raw_model_response,
            raw_completion=raw_completion,
            clean_completion=clean_python_completion(raw_completion),
        )
        _append_jsonl(records_path, record)
        records[task_id] = record
        _atomic_json(
            output / "progress.json",
            {
                **run_spec,
                "checkpoint_epoch_index": state.epoch_index,
                "checkpoint_step": state.step,
                "completed_tasks": len(records),
                "planned_tasks": len(tasks),
                "last_task_id": task_id,
                "total_generation_seconds": sum(
                    float(item["generation_seconds"]) for item in records.values()
                ),
                "generation_complete": len(records) == len(tasks),
            },
        )
        print(
            json.dumps(
                {
                    "completed_tasks": len(records),
                    "planned_tasks": len(tasks),
                    "task_id": task_id,
                    "generated_tokens": len(generated_ids),
                    "seconds": elapsed,
                    "tokens_per_second": len(generated_ids) / elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ordered = [records[str(task["task_id"])] for task in tasks]
    _write_samples(output / "samples-raw.jsonl", ordered, "raw_completion")
    _write_samples(output / "samples.jsonl", ordered, "clean_completion")
    summary = {
        **run_spec,
        "checkpoint_epoch_index": state.epoch_index,
        "checkpoint_step": state.step,
        "completed_tasks": len(ordered),
        "planned_tasks": len(tasks),
        "generation_complete": len(ordered) == len(tasks),
        "total_generated_tokens": sum(int(item["generated_token_count"]) for item in ordered),
        "total_generation_seconds": sum(float(item["generation_seconds"]) for item in ordered),
        "mean_tokens_per_second": (
            sum(int(item["generated_token_count"]) for item in ordered)
            / sum(float(item["generation_seconds"]) for item in ordered)
        ),
    }
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
