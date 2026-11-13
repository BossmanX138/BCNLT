"""Run at most 100 verified training steps and write an atomic resumable checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from itertools import chain
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.config import ModelConfig  # noqa: E402
from bosscrafts_model.model import EngineeringModel  # noqa: E402
from bosscrafts_model.model_acceptance import (  # noqa: E402
    verify_heldout_suite_manifest,
)
from bosscrafts_model.resumable_task_packing import (  # noqa: E402
    ResumableTaskDataset,
    audit_task_database,
)
from bosscrafts_model.tokenizer import (  # noqa: E402
    EXPECTED_RUNTIME_VERSION,
    NativeTokenizerBinding,
    bound_native_65k_config,
)
from bosscrafts_model.trainer import run_training_steps  # noqa: E402
from bosscrafts_model.training_checkpoint import (  # noqa: E402
    load_model_checkpoint_for_inference,
    load_training_checkpoint,
    save_training_checkpoint,
)
from bosscrafts_model.training_progress import TRAINING_PROGRESS_SCHEMA  # noqa: E402
from bosscrafts_model.training_schedule import cosine_learning_rate  # noqa: E402

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-deterministic-clean-v1"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bcnlt-v7-bounded-training-trial-v1"
    / "checkpoint.pt"
)
MAX_HELDOUT_MANIFEST_BYTES = 16 * 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate held-out manifest key: {key}")
        payload[key] = value
    return payload


def _read_heldout_manifest(path: Path) -> dict[str, object]:
    try:
        content = path.resolve().read_bytes()
        if len(content) > MAX_HELDOUT_MANIFEST_BYTES:
            raise ValueError("held-out suite manifest exceeds size limit")
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("held-out suite manifest is unreadable") from error
    if type(payload) is not dict:
        raise ValueError("held-out suite manifest must be a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Publish a complete UTF-8 JSON document without exposing partial writes."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths_alias(left: Path, right: Path) -> bool:
    """Return whether two paths name the same destination, including hard links."""
    left = left.resolve()
    right = right.resolve()
    if left == right:
        return True
    if not left.exists() or not right.exists():
        return False
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--evaluation-database", type=Path)
    parser.add_argument("--validation-database", type=Path)
    parser.add_argument("--test-database", type=Path)
    parser.add_argument("--heldout-suite-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initial-model-checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint-database", type=Path)
    parser.add_argument("--initialization-checkpoint-sha256")
    parser.add_argument("--initialization-database-identity-sha256")
    arguments = parser.parse_args()
    if not 1 <= arguments.max_steps <= 100:
        parser.error("bounded trial max-steps must be between 1 and 100")
    if arguments.gradient_accumulation < 1:
        parser.error("gradient accumulation must be positive")
    if not 1 <= arguments.epochs <= 100:
        parser.error("epochs must be between 1 and 100")
    if not math.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if not math.isfinite(arguments.warmup_ratio) or not 0 <= arguments.warmup_ratio < 1:
        parser.error("warmup ratio must be in [0, 1)")
    if (
        not math.isfinite(arguments.minimum_learning_rate_ratio)
        or not 0 < arguments.minimum_learning_rate_ratio <= 1
    ):
        parser.error("minimum learning-rate ratio must be in (0, 1]")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if arguments.dtype == "bfloat16" and arguments.device == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("CUDA bfloat16 was requested but is unsupported")
    if (arguments.initial_model_checkpoint is None) != (
        arguments.initial_checkpoint_database is None
    ):
        parser.error(
            "initial-model-checkpoint and initial-checkpoint-database must be supplied together"
        )
    plural_evaluation_supplied = (
        arguments.validation_database is not None
        or arguments.test_database is not None
    )
    if (arguments.validation_database is None) != (arguments.test_database is None):
        parser.error("validation-database and test-database must be supplied together")
    if arguments.evaluation_database is not None and plural_evaluation_supplied:
        parser.error("singular and plural evaluation databases are mutually exclusive")
    evaluation_contract_supplied = (
        arguments.evaluation_database is not None or plural_evaluation_supplied
    )
    if evaluation_contract_supplied != (arguments.heldout_suite_manifest is not None):
        parser.error(
            "evaluation databases and heldout-suite-manifest must be supplied together"
        )
    if arguments.full_model and not evaluation_contract_supplied:
        parser.error("full-model training requires a frozen evaluation contract")
    if (arguments.initialization_checkpoint_sha256 is None) != (
        arguments.initialization_database_identity_sha256 is None
    ):
        parser.error("both initialization SHA-256 identities must be supplied together")
    for name, value in (
        ("initialization-checkpoint-sha256", arguments.initialization_checkpoint_sha256),
        (
            "initialization-database-identity-sha256",
            arguments.initialization_database_identity_sha256,
        ),
    ):
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            parser.error(f"{name} must be a lowercase SHA-256 digest")
    if (
        arguments.initial_model_checkpoint is None
        and arguments.initialization_checkpoint_sha256 is not None
        and not arguments.resume
    ):
        parser.error("fingerprint-only initialization is allowed only when resuming")
    output_paths = {"checkpoint": arguments.checkpoint}
    if arguments.status_output is not None:
        output_paths["status output"] = arguments.status_output
    input_paths = {"training database": arguments.database}
    if arguments.evaluation_database is not None:
        input_paths["evaluation database"] = arguments.evaluation_database
    if arguments.validation_database is not None:
        input_paths["validation database"] = arguments.validation_database
    if arguments.test_database is not None:
        input_paths["test database"] = arguments.test_database
    if arguments.heldout_suite_manifest is not None:
        input_paths["held-out suite manifest"] = arguments.heldout_suite_manifest
    if arguments.initial_model_checkpoint is not None:
        input_paths["initial checkpoint"] = arguments.initial_model_checkpoint
    if arguments.initial_checkpoint_database is not None:
        input_paths["initial checkpoint database"] = arguments.initial_checkpoint_database
    for output_name, output_path in output_paths.items():
        for input_name, input_path in input_paths.items():
            if _paths_alias(output_path, input_path):
                parser.error(f"{output_name} and {input_name} must differ")
    if (
        arguments.status_output is not None
        and _paths_alias(arguments.status_output, arguments.checkpoint)
    ):
        parser.error("status output and checkpoint must differ")
    if not arguments.resume and arguments.checkpoint.exists():
        parser.error("checkpoint already exists; use --resume or a new destination")

    torch.manual_seed(7)
    torch.set_num_threads(2)
    device = torch.device(arguments.device)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    binding = NativeTokenizerBinding.load()
    task_audit = audit_task_database(binding, arguments.database)
    evaluation_database_identity_sha256: str | None = None
    evaluation_database_identities_sha256: dict[str, str] | None = None
    heldout_suite_manifest_sha256: str | None = None
    if arguments.evaluation_database is not None:
        if arguments.heldout_suite_manifest is None:
            raise RuntimeError("validated held-out suite manifest is missing")
        evaluation_audit = audit_task_database(
            binding, arguments.evaluation_database
        )
        heldout_suite = verify_heldout_suite_manifest(
            _read_heldout_manifest(arguments.heldout_suite_manifest),
            task_database_identity_sha256=(
                evaluation_audit.database_identity_sha256
            ),
        )
        evaluation_database_identity_sha256 = (
            heldout_suite.task_database_identity_sha256
        )
        heldout_suite_manifest_sha256 = heldout_suite.manifest_sha256
    elif arguments.validation_database is not None:
        if arguments.test_database is None or arguments.heldout_suite_manifest is None:
            raise RuntimeError("validated plural evaluation contract is incomplete")
        evaluation_audits = {
            "validation": audit_task_database(binding, arguments.validation_database),
            "test": audit_task_database(binding, arguments.test_database),
        }
        heldout_suite = verify_heldout_suite_manifest(
            _read_heldout_manifest(arguments.heldout_suite_manifest),
            task_database_identities_sha256={
                split: audit.database_identity_sha256
                for split, audit in evaluation_audits.items()
            },
        )
        evaluation_database_identities_sha256 = (
            heldout_suite.evaluation_task_database_identities_sha256
        )
        heldout_suite_manifest_sha256 = heldout_suite.manifest_sha256
    dataset = ResumableTaskDataset(binding, arguments.database)
    train_example_count = dataset.example_count(split="train")
    if train_example_count < 1:
        raise ValueError("training database contains no train examples")
    optimizer_steps_per_epoch = math.ceil(
        train_example_count / arguments.gradient_accumulation
    )
    planned_optimizer_steps = optimizer_steps_per_epoch * arguments.epochs
    warmup_steps = min(
        planned_optimizer_steps - 1,
        int(planned_optimizer_steps * arguments.warmup_ratio),
    )
    if arguments.full_model:
        config = bound_native_65k_config(binding)
    else:
        config = ModelConfig.tiny(
            vocab_size=binding.vocab_size,
            tokenizer_manifest=binding.binding_sha256,
        )
    config.check_manifest(binding.binding_sha256)
    initialization: dict[str, object] = {"type": "random_seed_7"}
    initial_task_audit = None
    initial_checkpoint_path = None
    if arguments.initial_model_checkpoint is not None:
        initial_checkpoint_path = arguments.initial_model_checkpoint.resolve()
        if not initial_checkpoint_path.is_file():
            parser.error("initial model checkpoint does not exist")
        if arguments.initial_checkpoint_database is None:
            raise RuntimeError("validated initial checkpoint database is missing")
        initial_task_audit = audit_task_database(
            binding, arguments.initial_checkpoint_database.resolve()
        )
        observed_checkpoint_sha256 = _sha256_file(initial_checkpoint_path)
        if (
            arguments.initialization_checkpoint_sha256 is not None
            and arguments.initialization_checkpoint_sha256
            != observed_checkpoint_sha256
        ):
            raise ValueError("initial checkpoint SHA-256 mismatch")
        if (
            arguments.initialization_database_identity_sha256 is not None
            and arguments.initialization_database_identity_sha256
            != initial_task_audit.database_identity_sha256
        ):
            raise ValueError("initial checkpoint database identity mismatch")
        initialization = {
            "type": "verified_completed_checkpoint",
            "checkpoint_sha256": observed_checkpoint_sha256,
            "task_database_identity_sha256": (
                initial_task_audit.database_identity_sha256
            ),
        }
    elif arguments.initialization_checkpoint_sha256 is not None:
        initialization = {
            "type": "verified_completed_checkpoint",
            "checkpoint_sha256": arguments.initialization_checkpoint_sha256,
            "task_database_identity_sha256": (
                arguments.initialization_database_identity_sha256
            ),
        }
    training_plan: dict[str, object] = {
        "dtype": arguments.dtype,
        "epochs": arguments.epochs,
        "full_model": arguments.full_model,
        "gradient_accumulation": arguments.gradient_accumulation,
        "gradient_checkpointing": arguments.gradient_checkpointing,
        "optimizer": {
            "amsgrad": False,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "learning_rate": arguments.learning_rate,
            "name": "AdamW",
            "weight_decay": 0.01,
        },
        "schedule": {
            "minimum_learning_rate_ratio": arguments.minimum_learning_rate_ratio,
            "name": "linear_warmup_cosine_decay",
            "planned_optimizer_steps": planned_optimizer_steps,
            "warmup_ratio": arguments.warmup_ratio,
            "warmup_steps": warmup_steps,
        },
        "schema_version": (
            "bosscrafts-training-plan-v4"
            if evaluation_database_identities_sha256 is not None
            else (
                "bosscrafts-training-plan-v3"
                if evaluation_database_identity_sha256 is not None
                else "bosscrafts-training-plan-v2"
            )
        ),
        "seed": 7,
        "split": "train",
        "initialization": initialization,
    }
    if evaluation_database_identity_sha256 is not None:
        training_plan["evaluation_contract"] = {
            "evaluation_task_database_identity_sha256": (
                evaluation_database_identity_sha256
            ),
            "heldout_suite_manifest_sha256": heldout_suite_manifest_sha256,
        }
    elif evaluation_database_identities_sha256 is not None:
        training_plan["evaluation_contract"] = {
            "evaluation_task_database_identities_sha256": (
                evaluation_database_identities_sha256
            ),
            "heldout_suite_manifest_sha256": heldout_suite_manifest_sha256,
        }
    training_plan_sha256 = hashlib.sha256(
        json.dumps(training_plan, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    model = EngineeringModel(config).to(device=device, dtype=dtype)
    if arguments.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    def scheduled_learning_rate(step_index: int) -> float:
        return cosine_learning_rate(
            step_index,
            total_steps=planned_optimizer_steps,
            peak_learning_rate=arguments.learning_rate,
            warmup_steps=warmup_steps,
            minimum_ratio=arguments.minimum_learning_rate_ratio,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scheduled_learning_rate(0),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
    )

    initial_steps = 0
    initial_examples = 0
    epoch_index = 0
    cursor: str | None = None
    last_example_id: str | None = None
    if arguments.resume:
        state = load_training_checkpoint(
            arguments.checkpoint,
            model,
            optimizer,
            expected_task_database_identity_sha256=task_audit.database_identity_sha256,
            expected_binding_sha256=binding.binding_sha256,
            expected_model_fingerprint=config.fingerprint(),
            expected_target_epochs=arguments.epochs,
            expected_training_plan_sha256=training_plan_sha256,
            expected_evaluation_task_database_identity_sha256=(
                evaluation_database_identity_sha256
            ),
            expected_evaluation_task_database_identities_sha256=(
                evaluation_database_identities_sha256
            ),
            expected_heldout_suite_manifest_sha256=(
                heldout_suite_manifest_sha256
            ),
        )
        initial_steps = state.step
        initial_examples = state.examples_seen
        epoch_index = state.epoch_index
        cursor = state.cursor_after_example_id
        last_example_id = state.last_example_id
    elif initial_checkpoint_path is not None:
        if initial_task_audit is None:
            raise RuntimeError("initial checkpoint audit was not prepared")
        initial_state = load_model_checkpoint_for_inference(
            initial_checkpoint_path,
            model,
            expected_task_database_identity_sha256=(
                initial_task_audit.database_identity_sha256
            ),
            expected_binding_sha256=binding.binding_sha256,
            expected_model_fingerprint=config.fingerprint(),
        )
        if not (
            initial_state.epoch_index >= initial_state.target_epochs
            and initial_state.cursor_after_example_id is None
        ):
            raise ValueError("initial model checkpoint training is incomplete")
        if initial_state.checkpoint_sha256 != initialization["checkpoint_sha256"]:
            raise ValueError("initial checkpoint changed before loading")

    remaining_steps = arguments.max_steps
    invocation_steps = 0
    invocation_examples = 0
    invocation_sequence_tokens = 0
    invocation_target_tokens = 0
    invocation_weighted_loss = 0.0
    gradients_finite = True
    last_learning_rate = float(optimizer.param_groups[0]["lr"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    while remaining_steps and epoch_index < arguments.epochs:
        examples = dataset.iter_examples(split="train", after_example_id=cursor)
        first = next(examples, None)
        if first is None:
            epoch_index += 1
            cursor = None
            continue
        requested_steps = remaining_steps
        result = run_training_steps(
            model,
            optimizer,
            chain((first,), examples),
            device=device,
            max_steps=requested_steps,
            gradient_accumulation=arguments.gradient_accumulation,
            step_offset=initial_steps + invocation_steps,
            learning_rate_schedule=scheduled_learning_rate,
        )
        invocation_steps += result.optimizer_steps
        invocation_examples += result.examples_seen
        invocation_sequence_tokens += result.sequence_tokens_seen
        invocation_target_tokens += result.target_tokens_seen
        invocation_weighted_loss += result.mean_loss * result.examples_seen
        gradients_finite = gradients_finite and result.gradients_finite
        last_learning_rate = result.last_learning_rate
        remaining_steps -= result.optimizer_steps
        cursor = result.last_example_id
        last_example_id = result.last_example_id
        if result.optimizer_steps < requested_steps or next(examples, None) is None:
            epoch_index += 1
            cursor = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    if last_example_id is None:
        raise ValueError("training database contains no train examples")
    total_steps = initial_steps + invocation_steps
    total_examples = initial_examples + invocation_examples
    mean_loss = (
        invocation_weighted_loss / invocation_examples
        if invocation_examples
        else 0.0
    )
    checkpoint_started = time.perf_counter()
    save_training_checkpoint(
        arguments.checkpoint,
        model,
        optimizer,
        step=total_steps,
        examples_seen=total_examples,
        epoch_index=epoch_index,
        target_epochs=arguments.epochs,
        last_example_id=last_example_id,
        cursor_after_example_id=cursor,
        task_database_identity_sha256=task_audit.database_identity_sha256,
        binding_sha256=binding.binding_sha256,
        model_fingerprint=config.fingerprint(),
        training_plan_sha256=training_plan_sha256,
        evaluation_task_database_identity_sha256=(
            evaluation_database_identity_sha256
        ),
        evaluation_task_database_identities_sha256=(
            evaluation_database_identities_sha256
        ),
        heldout_suite_manifest_sha256=heldout_suite_manifest_sha256,
    )
    checkpoint_write_seconds = time.perf_counter() - checkpoint_started
    checkpoint_sha256 = _sha256_file(arguments.checkpoint.resolve())
    payload = {
        "schema_version": TRAINING_PROGRESS_SCHEMA,
        "binding_sha256": binding.binding_sha256,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": arguments.checkpoint.stat().st_size,
        "checkpoint_written": True,
        "device": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "full_model": arguments.full_model,
        "gradient_checkpointing": model.gradient_checkpointing,
        "epoch_index": epoch_index,
        "target_epochs": arguments.epochs,
        "gradients_finite": gradients_finite,
        "invocation_examples_seen": invocation_examples,
        "invocation_optimizer_steps": invocation_steps,
        "invocation_sequence_tokens_seen": invocation_sequence_tokens,
        "invocation_target_tokens_seen": invocation_target_tokens,
        "last_example_id": last_example_id,
        "learning_rate": last_learning_rate,
        "max_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "mean_loss": mean_loss,
        "model_fingerprint": config.fingerprint(),
        "planned_optimizer_steps": planned_optimizer_steps,
        "optimizer_steps_remaining": max(0, planned_optimizer_steps - total_steps),
        "optimizer_step_fraction": min(1.0, total_steps / planned_optimizer_steps),
        "task_database_identity_sha256": task_audit.database_identity_sha256,
        "heldout_suite_manifest_sha256": heldout_suite_manifest_sha256,
        "training_seconds": training_seconds,
        "sequence_tokens_per_second": (
            invocation_sequence_tokens / training_seconds if training_seconds > 0 else 0.0
        ),
        "target_tokens_per_second": (
            invocation_target_tokens / training_seconds if training_seconds > 0 else 0.0
        ),
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "tokenizers_version": EXPECTED_RUNTIME_VERSION,
        "training_plan_sha256": training_plan_sha256,
        "initialization": initialization,
        "total_examples_seen": total_examples,
        "total_optimizer_steps": total_steps,
        "training_complete": epoch_index >= arguments.epochs,
    }
    if evaluation_database_identity_sha256 is not None:
        payload["evaluation_task_database_identity_sha256"] = (
            evaluation_database_identity_sha256
        )
    if evaluation_database_identities_sha256 is not None:
        payload["evaluation_task_database_identities_sha256"] = (
            evaluation_database_identities_sha256
        )
    if arguments.status_output is not None:
        _write_json_atomic(arguments.status_output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
