"""Run exactly one bounded native-65K optimization step without saving weights."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

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

from bosscrafts_model.config import ModelConfig
from bosscrafts_model.model import EngineeringModel, masked_next_token_loss
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config


def _training_example(
    binding: NativeTokenizerBinding, sequence_length: int
) -> tuple[list[int], int]:
    token = binding.structural_token_ids
    fixed_token_count = 9
    raw_token_count = sequence_length - fixed_token_count
    if raw_token_count < 1:
        raise ValueError("sequence length must leave at least one raw token")
    raw_token = binding.encode("x")[0]
    ids = [
        token["<|bcnlt:task:code_to_semantic|>"],
        token["<|bcnlt:stream:begin|>"],
        token["<|bcnlt:mode:raw_code|>"],
        *([raw_token] * raw_token_count),
        token["<|bcnlt:stream:end|>"],
    ]
    loss_start = len(ids)
    ids.extend(
        [
            token["<|bcnlt:stream:begin|>"],
            token["<|bcnlt:mode:semantic_code|>"],
            token["<|bcnlt:event:assign|>"],
            token["<|bcnlt:stream:end|>"],
            token["<|bcnlt:task:end|>"],
        ]
    )
    return ids, loss_start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--sequence-length", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if arguments.dtype == "bfloat16" and arguments.device == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("CUDA bfloat16 was requested but is unsupported")
    if not math.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")

    torch.manual_seed(7)
    torch.set_num_threads(2)
    binding = NativeTokenizerBinding.load()
    if arguments.full_model:
        config = bound_native_65k_config(binding)
    else:
        config = ModelConfig.tiny(
            vocab_size=binding.vocab_size,
            tokenizer_manifest=binding.binding_sha256,
        )
    config.check_manifest(binding.binding_sha256)
    if not 10 <= arguments.sequence_length <= config.max_sequence:
        parser.error(
            f"sequence length must be between 10 and {config.max_sequence}"
        )
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = EngineeringModel(config).to(device=device, dtype=dtype).train()
    if arguments.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    ids_list, loss_start = _training_example(binding, arguments.sequence_length)
    ids = torch.tensor([ids_list], dtype=torch.long, device=device)
    boundaries = torch.tensor([loss_start], dtype=torch.long, device=device)
    probe_token = ids_list[loss_start]
    before = model.embedding.weight[probe_token].detach().float().clone()
    if arguments.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=arguments.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    logits, routes = model(ids)
    loss = masked_next_token_loss(logits, ids, boundaries)
    loss.backward()

    gradient_square_sum = torch.zeros((), dtype=torch.float32, device=device)
    gradient_finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        gradient_finite = gradient_finite and bool(torch.isfinite(gradient).all())
        gradient_square_sum += gradient.square().sum()
    gradient_norm = float(torch.sqrt(gradient_square_sum).item())
    optimizer.step()
    after = model.embedding.weight[probe_token].detach().float()
    parameter_changed = not torch.equal(before, after)
    loss_value = float(loss.detach().float().item())
    payload = {
        "binding_sha256": binding.binding_sha256,
        "checkpoint_written": False,
        "device": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "full_model": arguments.full_model,
        "gradient_finite": gradient_finite,
        "gradient_checkpointing": model.gradient_checkpointing,
        "gradient_norm": gradient_norm,
        "loss_finite": math.isfinite(loss_value),
        "loss": loss_value,
        "max_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "model_fingerprint": config.fingerprint(),
        "optimizer": type(optimizer).__name__,
        "parameter_changed": parameter_changed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "route_rounds": len(routes),
        "sequence_length": ids.shape[1],
        "tokenizer_sha256": binding.tokenizer_sha256,
        "tokenizers_runtime_path": str(Path(tokenizers.__file__).resolve()),
        "tokenizers_version": tokenizers.__version__,
        "trained_steps": 1,
        "training_ready": False,
    }
    if not payload["loss_finite"] or not gradient_finite or gradient_norm <= 0 or not parameter_changed:
        raise RuntimeError("bounded optimization step did not produce a finite parameter update")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
