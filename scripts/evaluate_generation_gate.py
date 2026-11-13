"""Run a resumable strict-generation gate on deterministic held-out task rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
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

from bosscrafts_model.generation_artifacts import (
    GENERATION_GATE_SCHEMA,
    GenerationGateRecord,
    load_generation_gate_records,
)
from bosscrafts_model.immutable_artifacts import publish_new_files, replace_mutable_file
from bosscrafts_model.inference_protocol import (
    INFERENCE_TASKS,
    response_payload_token_ids,
    response_protocol_complete,
    task_end_token_id,
)
from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.resumable_task_packing import (
    ResumableTaskDataset,
    _canonical_json,
    _sha256_file,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config
from bosscrafts_model.training_checkpoint import load_model_checkpoint_for_inference

SCHEMA = GENERATION_GATE_SCHEMA
DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-deterministic-clean-v5"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bosscrafts-engineering-v8-consolidation"
    / "checkpoint.pt"
)


class _GenerationCase(TypedDict):
    example_id: str
    family: str
    prompt_ids: list[int]
    expected_ids: list[int]


class _GenerationSummary(TypedDict):
    case_count: int
    protocol_complete_count: int
    protocol_complete_rate: float
    exact_match_count: int
    exact_match_rate: float
    mean_target_token_accuracy: float
    total_generation_seconds: float


def _render_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _replace_progress(path: Path, payload: Mapping[str, object]) -> None:
    replace_mutable_file(path, _render_json(payload))


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _select_cases(
    dataset: ResumableTaskDataset,
    *,
    split: str,
    families: tuple[str, ...],
    limit_per_family: int,
    maximum_new_tokens: int,
    maximum_sequence: int,
) -> list[_GenerationCase]:
    selected: dict[str, list[_GenerationCase]] = {
        family: [] for family in families
    }
    for row in dataset.iter_examples(split=split):
        family = str(row["family"])
        if family not in selected or len(selected[family]) >= limit_per_family:
            continue
        input_ids = list(row["input_ids"])
        loss_start = int(row["loss_start_index"])
        expected_ids = input_ids[loss_start:]
        prompt_ids = input_ids[:loss_start]
        if (
            len(expected_ids) > maximum_new_tokens
            or len(prompt_ids) + maximum_new_tokens > maximum_sequence
        ):
            continue
        selected[family].append(
            _GenerationCase(
                example_id=str(row["example_id"]),
                family=family,
                prompt_ids=prompt_ids,
                expected_ids=expected_ids,
            )
        )
    missing = {
        family: len(rows)
        for family, rows in selected.items()
        if len(rows) != limit_per_family
    }
    if missing:
        raise ValueError(
            "insufficient bounded held-out generation cases: "
            + json.dumps(missing, sort_keys=True)
        )
    return [row for family in families for row in selected[family]]


def _verify_resumed_records(
    binding: NativeTokenizerBinding,
    cases: list[_GenerationCase],
    records: Mapping[str, GenerationGateRecord],
    *,
    maximum_new_tokens: int,
) -> None:
    """Rebind persisted records to their exact held-out rows and tokenizer."""
    selected = {str(case["example_id"]): case for case in cases}
    if not set(records).issubset(selected):
        raise ValueError("generation gate records contain an unselected example")
    for example_id, record in records.items():
        case = selected[example_id]
        family = str(case["family"])
        prompt_ids = list(case["prompt_ids"])
        expected_ids = list(case["expected_ids"])
        generated_ids = list(record["generated_ids"])
        if record["example_id"] != example_id:
            raise ValueError("generation gate record key is inconsistent")
        if record["family"] != family:
            raise ValueError("generation gate record family does not match its case")
        if record["prompt_token_count"] != len(prompt_ids):
            raise ValueError("generation gate prompt count does not match its case")
        if (
            record["expected_token_count"] != len(expected_ids)
            or record["expected_ids"] != expected_ids
        ):
            raise ValueError("generation gate expected tokens do not match their case")
        if len(generated_ids) > maximum_new_tokens:
            raise ValueError("generation gate record exceeds the generation bound")
        if any(token_id >= binding.vocab_size for token_id in generated_ids):
            raise ValueError("generation gate record contains an out-of-vocabulary token")
        if record["expected_protocol_text"] != binding.decode(expected_ids):
            raise ValueError("generation gate expected text is inconsistent")
        if record["generated_protocol_text"] != binding.decode(generated_ids):
            raise ValueError("generation gate generated text is inconsistent")
        payload_ids = response_payload_token_ids(binding, family, generated_ids)
        if record["generated_payload_text"] != binding.decode(payload_ids):
            raise ValueError("generation gate payload text is inconsistent")
        protocol_complete = response_protocol_complete(
            binding,
            task=family,
            generated_ids=generated_ids,
        )
        if record["protocol_complete"] is not protocol_complete:
            raise ValueError("generation gate protocol result is inconsistent")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--checkpoint-database",
        type=Path,
        help="training database recorded by the checkpoint; defaults to --database",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=INFERENCE_TASKS[1:],
        default=None,
    )
    parser.add_argument("--limit-per-family", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--allow-incomplete-probe",
        action="store_true",
        help=(
            "allow a diagnostic-only run against an incomplete checkpoint; "
            "the resulting report can never pass the strict gate"
        ),
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.limit_per_family <= 128:
        parser.error("limit-per-family must be between 1 and 128")
    if not 1 <= arguments.max_new_tokens <= 256:
        parser.error("max-new-tokens must be between 1 and 256")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if (
        arguments.device == "cuda"
        and arguments.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        parser.error("CUDA bfloat16 was requested but is unsupported")

    database = arguments.database.resolve()
    checkpoint_database = (
        arguments.checkpoint_database.resolve()
        if arguments.checkpoint_database is not None
        else database
    )
    checkpoint = arguments.checkpoint.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binding = NativeTokenizerBinding.load()
    evaluation_audit = audit_task_database(binding, database)
    checkpoint_audit = audit_task_database(binding, checkpoint_database)
    dataset = ResumableTaskDataset(binding, database)
    available_families = tuple(
        family
        for family in INFERENCE_TASKS[1:]
        if family in evaluation_audit.family_counts
    )
    families = tuple(arguments.families or available_families)
    if len(set(families)) != len(families) or not families:
        parser.error("families must be unique and nonempty")
    unavailable = sorted(set(families) - set(available_families))
    if unavailable:
        parser.error("database lacks requested families: " + ",".join(unavailable))

    config = bound_native_65k_config(binding)
    cases = _select_cases(
        dataset,
        split=arguments.split,
        families=families,
        limit_per_family=arguments.limit_per_family,
        maximum_new_tokens=arguments.max_new_tokens,
        maximum_sequence=config.max_sequence,
    )
    run_spec: dict[str, object] = {
        "schema_version": SCHEMA,
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "model_fingerprint": config.fingerprint(),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_task_database_identity_sha256": (
            checkpoint_audit.database_identity_sha256
        ),
        "evaluation_task_database_identity_sha256": (
            evaluation_audit.database_identity_sha256
        ),
        "device": arguments.device,
        "dtype": arguments.dtype,
        "split": arguments.split,
        "families": list(families),
        "limit_per_family": arguments.limit_per_family,
        "max_new_tokens": arguments.max_new_tokens,
        "generation": "greedy_argmax",
        "evaluation_mode": (
            "incomplete_checkpoint_probe"
            if arguments.allow_incomplete_probe
            else "promotion_gate"
        ),
        "selected_example_ids": [str(row["example_id"]) for row in cases],
    }
    run_identity = hashlib.sha256(
        _canonical_json(run_spec).encode("utf-8")
    ).hexdigest()
    run_spec["run_identity_sha256"] = run_identity
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != run_spec:
            raise ValueError("generation gate output belongs to another run")
    else:
        publish_new_files(
            ((manifest_path, _render_json(run_spec)),),
            artifact_label="generation gate manifest",
        )

    records_path = output / "generation-records.jsonl"
    records = load_generation_gate_records(records_path, run_identity)
    _verify_resumed_records(
        binding,
        cases,
        records,
        maximum_new_tokens=arguments.max_new_tokens,
    )

    device = torch.device(arguments.device)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model = EngineeringModel(config).to(device=device, dtype=dtype).eval()
    state = load_model_checkpoint_for_inference(
        checkpoint,
        model,
        expected_task_database_identity_sha256=(
            checkpoint_audit.database_identity_sha256
        ),
        expected_binding_sha256=binding.binding_sha256,
        expected_model_fingerprint=config.fingerprint(),
    )
    checkpoint_training_complete = (
        state.epoch_index >= state.target_epochs
        and state.cursor_after_example_id is None
    )
    if not checkpoint_training_complete and not arguments.allow_incomplete_probe:
        raise ValueError("strict generation requires a completed checkpoint")

    for case in cases:
        example_id = str(case["example_id"])
        if example_id in records:
            continue
        family = str(case["family"])
        prompt_ids = list(case["prompt_ids"])
        expected_ids = list(case["expected_ids"])
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        generated = model.generate(
            ids,
            max_new_tokens=arguments.max_new_tokens,
            eos_token_id=task_end_token_id(binding, family),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        generated_ids = generated[0, len(prompt_ids) :].tolist()
        correct = sum(
            actual == expected
            for actual, expected in zip(generated_ids, expected_ids, strict=False)
        )
        payload_ids = response_payload_token_ids(binding, family, generated_ids)
        record = GenerationGateRecord(
            schema_version=SCHEMA,
            run_identity_sha256=run_identity,
            example_id=example_id,
            family=family,
            prompt_token_count=len(prompt_ids),
            expected_token_count=len(expected_ids),
            generated_token_count=len(generated_ids),
            expected_ids=expected_ids,
            generated_ids=generated_ids,
            expected_protocol_text=binding.decode(expected_ids),
            generated_protocol_text=binding.decode(generated_ids),
            generated_payload_text=binding.decode(payload_ids),
            protocol_complete=response_protocol_complete(
                binding,
                task=family,
                generated_ids=generated_ids,
            ),
            exact_match=generated_ids == expected_ids,
            target_token_accuracy=correct / len(expected_ids),
            generation_seconds=elapsed,
            tokens_per_second=len(generated_ids) / elapsed,
        )
        _append_jsonl(records_path, record)
        records[example_id] = record
        _replace_progress(
            output / "progress.json",
            {
                **run_spec,
                "checkpoint_step": state.step,
                "completed_cases": len(records),
                "planned_cases": len(cases),
                "generation_complete": len(records) == len(cases),
                "last_example_id": example_id,
            },
        )
        print(
            json.dumps(
                {
                    "completed_cases": len(records),
                    "planned_cases": len(cases),
                    "family": family,
                    "example_id": example_id,
                    "protocol_complete": record["protocol_complete"],
                    "exact_match": record["exact_match"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ordered = [records[str(case["example_id"])] for case in cases]
    by_family: dict[str, list[GenerationGateRecord]] = defaultdict(list)
    for record in ordered:
        by_family[str(record["family"])].append(record)

    def summarize(items: list[GenerationGateRecord]) -> _GenerationSummary:
        return {
            "case_count": len(items),
            "protocol_complete_count": sum(
                bool(item["protocol_complete"]) for item in items
            ),
            "protocol_complete_rate": sum(
                bool(item["protocol_complete"]) for item in items
            )
            / len(items),
            "exact_match_count": sum(bool(item["exact_match"]) for item in items),
            "exact_match_rate": sum(bool(item["exact_match"]) for item in items)
            / len(items),
            "mean_target_token_accuracy": sum(
                float(item["target_token_accuracy"]) for item in items
            )
            / len(items),
            "total_generation_seconds": sum(
                float(item["generation_seconds"]) for item in items
            ),
        }

    overall = summarize(ordered)
    report = {
        **run_spec,
        "checkpoint_epoch_index": state.epoch_index,
        "checkpoint_step": state.step,
        "checkpoint_training_complete": checkpoint_training_complete,
        "generation_complete": len(ordered) == len(cases),
        "overall": overall,
        "by_family": {
            family: summarize(by_family[family]) for family in families
        },
        "strict_gate_passed": (
            checkpoint_training_complete
            and overall["protocol_complete_rate"] == 1.0
            and overall["exact_match_rate"] == 1.0
        ),
    }
    publish_new_files(
        ((output / "report.json", _render_json(report)),),
        artifact_label="generation gate report",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
