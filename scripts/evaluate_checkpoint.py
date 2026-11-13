"""Evaluate a lineage-verified BossCrafts checkpoint on streamed task examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
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

from bosscrafts_model.artifact_schemas import CHECKPOINT_EVALUATION_SCHEMA
from bosscrafts_model.config import ModelConfig
from bosscrafts_model.evaluation import (
    EvaluationBaseline,
    EvaluationResult,
    evaluate_model,
    load_evaluation_baseline,
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


def _publish_new_evaluation(path: Path, rendered: str) -> None:
    """Publish immutable evaluation evidence without clobbering prior results."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError("evaluation output already exists") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--checkpoint-database",
        type=Path,
        help="training database recorded by the checkpoint; defaults to --database",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--baseline-evaluation",
        type=Path,
        help="completed evaluation JSON to use as the frozen baseline",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.max_examples <= 10_000:
        parser.error("max-examples must be between 1 and 10000")
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
    split_example_count = dataset.example_count(split=arguments.split)
    torch.manual_seed(7)
    if arguments.full_model:
        config = bound_native_65k_config(binding)
    else:
        config = ModelConfig.tiny(
            vocab_size=binding.vocab_size,
            tokenizer_manifest=binding.binding_sha256,
        )
    device = torch.device(arguments.device)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model = EngineeringModel(config).to(device=device, dtype=dtype)
    baseline_evaluation_sha256 = None
    baseline: EvaluationResult | EvaluationBaseline
    if arguments.baseline_evaluation is None:
        baseline = evaluate_model(
            model,
            dataset.iter_examples(split=arguments.split),
            device=device,
            max_examples=arguments.max_examples,
        )
        baseline_source = "deterministic_random_initialization"
    else:
        baseline_path = arguments.baseline_evaluation.resolve()
        baseline = load_evaluation_baseline(
            baseline_path,
            expected_split=arguments.split,
            expected_example_count=split_example_count,
            expected_task_database_identity_sha256=(
                task_audit.database_identity_sha256
            ),
            expected_model_fingerprint=config.fingerprint(),
        )
        baseline_source = str(baseline_path)
        baseline_evaluation_sha256 = hashlib.sha256(
            baseline_path.read_bytes()
        ).hexdigest()
    state = load_model_checkpoint_for_inference(
        arguments.checkpoint,
        model,
        expected_task_database_identity_sha256=(
            checkpoint_task_audit.database_identity_sha256
        ),
        expected_binding_sha256=binding.binding_sha256,
        expected_model_fingerprint=config.fingerprint(),
    )
    result = evaluate_model(
        model,
        dataset.iter_examples(split=arguments.split),
        device=device,
        max_examples=arguments.max_examples,
    )
    relative_improvement = (
        baseline.mean_loss - result.mean_loss
    ) / baseline.mean_loss
    language_regressions = sorted(
        language
        for language, loss in result.language_mean_loss.items()
        if language in baseline.language_mean_loss
        and loss > baseline.language_mean_loss[language]
    )
    family_regressions = sorted(
        family
        for family, loss in result.family_mean_loss.items()
        if family in baseline.family_mean_loss
        and loss > baseline.family_mean_loss[family]
    )
    cell_regressions = sorted(
        cell
        for cell, loss in result.cell_mean_loss.items()
        if cell in baseline.cell_mean_loss
        and loss > baseline.cell_mean_loss[cell]
    )
    promotion_threshold = 0.02
    checkpoint_training_complete = (
        state.epoch_index >= state.target_epochs
        and state.cursor_after_example_id is None
    )
    evaluation_complete = result.example_count == split_example_count
    baseline_scope_matches = (
        baseline.target_token_count == result.target_token_count
        and set(baseline.language_mean_loss) == set(result.language_mean_loss)
        and set(baseline.family_mean_loss) == set(result.family_mean_loss)
        and set(baseline.cell_mean_loss) == set(result.cell_mean_loss)
    )
    promotion_blockers: list[str] = []
    if not checkpoint_training_complete:
        promotion_blockers.append("checkpoint training is incomplete")
    if not arguments.full_model:
        promotion_blockers.append("full production model was not evaluated")
    if not evaluation_complete:
        promotion_blockers.append("evaluation does not cover the complete split")
    if not baseline_scope_matches:
        promotion_blockers.append("baseline and candidate evaluation scopes differ")
    if relative_improvement < promotion_threshold:
        promotion_blockers.append("relative loss improvement is below threshold")
    if language_regressions:
        promotion_blockers.append("one or more language losses regressed")
    if family_regressions:
        promotion_blockers.append("one or more task-family losses regressed")
    if cell_regressions:
        promotion_blockers.append("one or more language-family cell losses regressed")
    payload = {
        **asdict(result),
        "schema_version": CHECKPOINT_EVALUATION_SCHEMA,
        "baseline_correct_target_token_count": baseline.correct_target_token_count,
        "baseline_evaluation_sha256": baseline_evaluation_sha256,
        "baseline_source": baseline_source,
        "baseline_cell_mean_loss": baseline.cell_mean_loss,
        "baseline_exact_match_count": baseline.exact_match_count,
        "baseline_exact_match_rate": baseline.exact_match_rate,
        "baseline_family_mean_loss": baseline.family_mean_loss,
        "baseline_language_mean_loss": baseline.language_mean_loss,
        "baseline_mean_loss": baseline.mean_loss,
        "baseline_perplexity": baseline.perplexity,
        "baseline_target_token_accuracy": baseline.target_token_accuracy,
        "baseline_target_token_count": baseline.target_token_count,
        "checkpoint_lineage_verified": True,
        "checkpoint_task_database_identity_sha256": (
            checkpoint_task_audit.database_identity_sha256
        ),
        "checkpoint_epoch_index": state.epoch_index,
        "checkpoint_step": state.step,
        "checkpoint_target_epochs": state.target_epochs,
        "checkpoint_training_complete": checkpoint_training_complete,
        "device": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "full_model": arguments.full_model,
        "model_fingerprint": config.fingerprint(),
        "family_regressions": family_regressions,
        "cell_regressions": cell_regressions,
        "language_regressions": language_regressions,
        "promotion_blockers": promotion_blockers,
        "promotion_gate_passed": not promotion_blockers,
        "promotion_threshold": promotion_threshold,
        "relative_loss_improvement": relative_improvement,
        "split": arguments.split,
        "split_example_count": split_example_count,
        "evaluated_fraction": result.example_count / split_example_count,
        "evaluation_complete": evaluation_complete,
        "target_token_accuracy_improved": (
            result.target_token_accuracy > baseline.target_token_accuracy
        ),
        "evaluation_task_database_identity_sha256": (
            task_audit.database_identity_sha256
        ),
        "task_database_identity_sha256": task_audit.database_identity_sha256,
    }
    rendered = json.dumps(payload, sort_keys=True)
    if arguments.output is not None:
        _publish_new_evaluation(arguments.output, rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
