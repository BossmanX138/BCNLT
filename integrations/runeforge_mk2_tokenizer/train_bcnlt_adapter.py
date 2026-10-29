from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from bcnlt.corpus.packing import PackedBCNLTDataset

from .embedding_extension import ExtendedLMHead, attach_bcnlt_extension


@dataclass(frozen=True)
class TrainingLineage:
    base_model_sha256: str
    tokenizer_manifest_sha256: str
    corpus_manifest_sha256: str
    training_config_sha256: str
    engine: str

    def __post_init__(self) -> None:
        for name in (
            "base_model_sha256",
            "tokenizer_manifest_sha256",
            "corpus_manifest_sha256",
            "training_config_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.engine.strip():
            raise ValueError("training engine must be recorded")

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()

    @classmethod
    def from_artifacts(
        cls,
        *,
        base_model: str | Path,
        tokenizer_manifest: str | Path,
        corpus_manifest: str | Path,
        training_config: str | Path,
        engine: str,
    ) -> "TrainingLineage":
        return cls(
            base_model_sha256=hash_artifact(base_model),
            tokenizer_manifest_sha256=hash_artifact(tokenizer_manifest),
            corpus_manifest_sha256=hash_artifact(corpus_manifest),
            training_config_sha256=hash_artifact(training_config),
            engine=engine,
        )


@dataclass(frozen=True)
class TrainingPreflightReport:
    passed: bool
    extension_trainable_parameters: int
    lora_trainable_parameters: int
    frozen_base_parameters: int
    lineage: TrainingLineage


@dataclass(frozen=True)
class AdapterArtifact:
    root: Path
    weights_path: Path
    lineage_path: Path
    weights_sha256: str


@dataclass(frozen=True)
class QLoRAConfig:
    target_modules: tuple[str, ...]
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    maximum_sequence_tokens: int = 4096
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 8


def load_qlora_model(
    base_model_path: str | Path,
    *,
    manifest: Any,
    definitions: dict[str, str],
    config: QLoRAConfig,
    seed: int,
    engine: str = "auto",
) -> tuple[nn.Module, str]:
    """Load a 4-bit base and attach LoRA plus the sparse BCNLT tables.

    Unsloth is selected when importable. The fallback uses the equivalent
    Transformers, PEFT, and bitsandbytes path. Imports remain lazy so corpus
    tooling does not require the GPU training environment.
    """
    if not config.target_modules:
        raise ValueError("at least one proven Runeforge LoRA target module is required")
    selected = engine
    if selected == "auto":
        try:
            import unsloth  # noqa: F401
        except ImportError:
            selected = "transformers-peft-bitsandbytes"
        else:
            selected = "unsloth"
    if selected == "unsloth":
        from unsloth import FastLanguageModel

        model, _ = FastLanguageModel.from_pretrained(
            model_name=str(base_model_path),
            max_seq_length=config.maximum_sequence_tokens,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.rank,
            target_modules=list(config.target_modules),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )
    elif selected == "transformers-peft-bitsandbytes":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            quantization_config=quantization,
            device_map="auto",
            trust_remote_code=False,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(config.target_modules),
            ),
        )
    else:
        raise ValueError(f"unsupported training engine: {selected}")
    attach_bcnlt_extension(model, manifest, definitions, seed)
    return model, selected


def run_training_steps(
    model: nn.Module,
    dataset: PackedBCNLTDataset,
    config: QLoRAConfig,
    *,
    maximum_optimizer_steps: int,
) -> list[float]:
    """Run a bounded, explicit optimizer loop over packed rows."""
    if maximum_optimizer_steps <= 0:
        raise ValueError("maximum_optimizer_steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Runeforge BCNLT training")
    if not len(dataset):
        raise ValueError("training dataset is empty")
    optimizer = torch.optim.AdamW(
        list(trainable_parameters(model)), lr=config.learning_rate
    )
    model.train()
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    optimizer_steps = 0
    output_head = model.get_output_embeddings()  # type: ignore[attr-defined]
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([sample["labels"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [sample["attention_mask"]], dtype=torch.long, device=device
        )
        if isinstance(output_head, ExtendedLMHead):
            hidden_states = _unsloth_hidden_states(
                model, input_ids=input_ids, attention_mask=attention_mask
            )
            measured_loss = backward_split_causal_loss(
                hidden_states,
                labels,
                output_head,
                gradient_scale=1.0 / config.gradient_accumulation_steps,
            )
        else:
            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                use_cache=False,
            )
            loss = output.loss / config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite BCNLT training loss")
            loss.backward()
            measured_loss = float(loss.detach().cpu()) * config.gradient_accumulation_steps
        losses.append(measured_loss)
        boundary = (sample_index + 1) % config.gradient_accumulation_steps == 0
        final_sample = sample_index + 1 == len(dataset)
        if boundary or final_sample:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if optimizer_steps >= maximum_optimizer_steps:
                break
    return losses


def backward_split_causal_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    output_head: ExtendedLMHead,
    *,
    gradient_scale: float,
    chunk_tokens: int = 128,
) -> float:
    """Backpropagate exact causal loss without a dense full-head gradient buffer."""
    if hidden_states.ndim != 3 or labels.shape != hidden_states.shape[:2]:
        raise ValueError("hidden states and labels have incompatible shapes")
    if gradient_scale <= 0 or chunk_tokens <= 0:
        raise ValueError("gradient scale and chunk size must be positive")
    shifted_hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    valid_tokens = int((shifted_labels != -100).sum().item())
    if valid_tokens <= 0:
        raise ValueError("training example has no supervised target tokens")
    detached_hidden = shifted_hidden.detach().requires_grad_(True)
    total_loss = 0.0
    for start in range(0, detached_hidden.shape[0], chunk_tokens):
        stop = min(start + chunk_tokens, detached_hidden.shape[0])
        logits = output_head(detached_hidden[start:stop])
        chunk_loss = F.cross_entropy(
            logits,
            shifted_labels[start:stop],
            ignore_index=-100,
            reduction="sum",
        )
        if not torch.isfinite(chunk_loss):
            raise FloatingPointError("non-finite BCNLT training loss")
        total_loss += float(chunk_loss.detach().cpu())
        (chunk_loss * (gradient_scale / valid_tokens)).backward()
    if detached_hidden.grad is None:
        raise RuntimeError("split BCNLT loss produced no hidden-state gradient")
    shifted_hidden.backward(detached_hidden.grad)
    return total_loss / valid_tokens


def _unsloth_hidden_states(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    previous = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    finally:
        if previous is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = previous
    hidden_states = output.logits
    if hidden_states.ndim != 3:
        raise RuntimeError("Unsloth did not return causal-model hidden states")
    return hidden_states


def verify_training_preflight(
    model: nn.Module,
    *,
    expected: TrainingLineage,
    observed: TrainingLineage,
    cuda_available: bool,
    resume: TrainingLineage | None = None,
) -> TrainingPreflightReport:
    _compare_lineage("", expected, observed)
    if resume is not None:
        _compare_lineage("resume ", expected, resume)
    if not cuda_available:
        raise RuntimeError("CUDA is required for Runeforge BCNLT training")

    extension = 0
    lora = 0
    frozen_base = 0
    for name, parameter in model.named_parameters():
        if ".base.weight" in name:
            if parameter.requires_grad:
                raise ValueError(f"whole base table is trainable: {name}")
            frozen_base += parameter.numel()
        elif ".extension.weight" in name:
            if parameter.requires_grad:
                extension += parameter.numel()
        elif "lora_" in name.lower() and parameter.requires_grad:
            lora += parameter.numel()
    if extension <= 0:
        raise ValueError("no trainable BCNLT extension parameters")
    if lora <= 0:
        raise ValueError("no trainable LoRA parameters")
    if frozen_base <= 0:
        raise ValueError("no frozen base embedding/output parameters detected")
    return TrainingPreflightReport(True, extension, lora, frozen_base, observed)


def trainable_parameters(model: nn.Module):
    for parameter in model.parameters():
        if parameter.requires_grad:
            yield parameter


def save_bcnlt_adapter(
    model: nn.Module,
    output_path: str | Path,
    tokenizer_artifact: str | Path,
    lineage: TrainingLineage,
) -> AdapterArtifact:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"adapter output already exists: {output}")
    tokenizer_source = Path(tokenizer_artifact)
    if not tokenizer_source.is_dir():
        raise FileNotFoundError(f"tokenizer artifact is missing: {tokenizer_source}")
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "extension" in name or "lora_" in name.lower()
    }
    if not state or not any("extension" in name for name in state):
        raise ValueError("adapter state has no BCNLT extension weights")
    output.mkdir(parents=True)
    weights_path = output / "adapter_model.pt"
    torch.save(state, weights_path)
    lineage_path = output / "lineage.json"
    lineage_path.write_text(
        json.dumps(lineage.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copytree(tokenizer_source, output / "tokenizer")
    digest = _sha256(weights_path)
    (output / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "adapter_weights": weights_path.name,
                "adapter_weights_sha256": digest,
                "lineage": lineage.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return AdapterArtifact(output, weights_path, lineage_path, digest)


def load_bcnlt_adapter_state(
    model: nn.Module,
    adapter_path: str | Path,
    *,
    expected: TrainingLineage | None = None,
) -> TrainingLineage:
    """Verify and restore an exact sparse-extension plus LoRA adapter artifact."""
    root = Path(adapter_path)
    artifact_path = root / "artifact_manifest.json"
    lineage_path = root / "lineage.json"
    if not artifact_path.is_file() or not lineage_path.is_file():
        raise FileNotFoundError("adapter artifact manifest or lineage is missing")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    weights_name = artifact.get("adapter_weights")
    if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
        raise ValueError("adapter weights path is invalid")
    weights_path = root / weights_name
    expected_hash = artifact.get("adapter_weights_sha256")
    if not weights_path.is_file() or _sha256(weights_path) != expected_hash:
        raise ValueError("adapter weights hash does not match the artifact manifest")
    lineage_raw = json.loads(lineage_path.read_text(encoding="utf-8"))
    if artifact.get("lineage") != lineage_raw:
        raise ValueError("adapter manifest lineage does not match lineage.json")
    observed = TrainingLineage(**lineage_raw)
    if expected is not None:
        _compare_lineage("adapter ", expected, observed)

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise ValueError("adapter weights are not a tensor state dictionary")
    model_state = model.state_dict()
    required = {
        name
        for name in model_state
        if "extension" in name or "lora_" in name.lower()
    }
    supplied = set(state)
    if supplied != required:
        missing = sorted(required - supplied)
        unexpected = sorted(supplied - required)
        raise ValueError(
            f"adapter state keys do not match model: missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(state, strict=False)
    return observed


def evaluate_packed_loss(
    model: nn.Module,
    dataset: Sequence[dict[str, list[int]]],
    *,
    maximum_examples: int | None = None,
) -> float:
    """Measure mean causal-LM loss directly on a fixed packed-data split."""
    if not len(dataset):
        raise ValueError("evaluation dataset is empty")
    if maximum_examples is not None and maximum_examples <= 0:
        raise ValueError("maximum_examples must be positive")
    count = len(dataset) if maximum_examples is None else min(len(dataset), maximum_examples)
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise ValueError("evaluation model has no parameters")
    device = first_parameter.device
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        with torch.inference_mode():
            for index in range(count):
                sample = dataset[index]
                input_ids = torch.tensor(
                    [sample["input_ids"]], dtype=torch.long, device=device
                )
                labels = torch.tensor(
                    [sample["labels"]], dtype=torch.long, device=device
                )
                attention_mask = torch.tensor(
                    [sample["attention_mask"]], dtype=torch.long, device=device
                )
                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                loss = output.loss
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError("non-finite BCNLT evaluation loss")
                losses.append(float(loss.detach().cpu()))
    finally:
        if was_training:
            model.train()
    return sum(losses) / len(losses)


def _compare_lineage(prefix: str, expected: TrainingLineage, observed: TrainingLineage) -> None:
    for field in expected.__dataclass_fields__:
        if getattr(expected, field) != getattr(observed, field):
            raise ValueError(f"{prefix}{field} does not match expected lineage")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_artifact(path_value: str | Path) -> str:
    path = Path(path_value)
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(f"lineage artifact is missing: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"lineage directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_digest = bytes.fromhex(_sha256(item))
        digest.update(file_digest)
    return digest.hexdigest()


def _default_definition(token: str) -> str:
    body = token.removeprefix("<|bcnlt:").removesuffix("|>")
    return "BCNLT structural token for " + body.replace(":", " ").replace("_", " ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded Runeforge BCNLT adapter training")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--packed-dataset", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--expected-lineage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--maximum-optimizer-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--engine", default="auto")
    args = parser.parse_args(argv)

    if args.engine in {"auto", "unsloth"}:
        try:
            import unsloth  # noqa: F401
        except ImportError:
            if args.engine == "unsloth":
                raise

    from bcnlt.tokenizer import BCNLTTokenizer

    config_raw = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
    config = QLoRAConfig(
        target_modules=tuple(config_raw["target_modules"]),
        rank=int(config_raw.get("rank", 16)),
        alpha=int(config_raw.get("alpha", 32)),
        dropout=float(config_raw.get("dropout", 0.0)),
        maximum_sequence_tokens=int(config_raw.get("maximum_sequence_tokens", 4096)),
        learning_rate=float(config_raw.get("learning_rate", 2e-4)),
        gradient_accumulation_steps=int(
            config_raw.get("gradient_accumulation_steps", 8)
        ),
    )
    tokenizer_root = Path(args.tokenizer_artifact)
    tokenizer_manifest_path = tokenizer_root / "bcnlt_tokenizer_manifest.json"
    tokenizer = BCNLTTokenizer.from_pretrained(tokenizer_root, args.base_model)
    manifest = tokenizer.manifest
    definitions = {
        item.token: _default_definition(item.token) for item in manifest.added_tokens
    }
    model, engine = load_qlora_model(
        args.base_model,
        manifest=manifest,
        definitions=definitions,
        config=config,
        seed=args.seed,
        engine=args.engine,
    )
    corpus_manifest = Path(args.packed_dataset) / "dataset_manifest.json"
    observed = TrainingLineage.from_artifacts(
        base_model=args.base_model,
        tokenizer_manifest=tokenizer_manifest_path,
        corpus_manifest=corpus_manifest,
        training_config=args.training_config,
        engine=engine,
    )
    expected = TrainingLineage(
        **json.loads(Path(args.expected_lineage).read_text(encoding="utf-8"))
    )
    verify_training_preflight(
        model,
        expected=expected,
        observed=observed,
        cuda_available=torch.cuda.is_available(),
    )
    dataset = PackedBCNLTDataset(
        args.packed_dataset,
        args.split,
        expected_tokenizer_digest=manifest.manifest_sha256,
    )
    losses = run_training_steps(
        model,
        dataset,
        config,
        maximum_optimizer_steps=args.maximum_optimizer_steps,
    )
    artifact = save_bcnlt_adapter(model, args.output, tokenizer_root, observed)
    smoke_manifest = {
        "schema_version": "bcnlt-smoke-manifest-v1",
        "lineage": observed.to_dict(),
        "maximum_optimizer_steps": args.maximum_optimizer_steps,
        "completed_training_examples": len(losses),
        "losses": losses,
        "finite_loss": all(math.isfinite(value) for value in losses),
        "adapter_weights_sha256": artifact.weights_sha256,
    }
    (artifact.root / "smoke_manifest.json").write_text(
        json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
