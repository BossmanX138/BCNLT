"""Run standalone inference from an immutable BossCrafts model bundle."""
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

from bosscrafts_model.inference_bundle import load_inference_bundle
from bosscrafts_model.inference_protocol import (
    INFERENCE_TASKS,
    build_inference_prompt,
    response_payload_token_ids,
    response_protocol_complete,
    task_end_token_id,
)
from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config

DEFAULT_BUNDLE = (
    PROJECT_ROOT / "artifacts" / "models" / "BossCrafts-Engineering-0.5B-v1"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--task", choices=INFERENCE_TASKS, default="code_to_semantic")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--expected-bundle-sha256")
    arguments = parser.parse_args()
    if not arguments.prompt:
        parser.error("prompt must not be empty")
    if not 1 <= arguments.max_new_tokens <= 256:
        parser.error("max-new-tokens must be between 1 and 256")
    if arguments.expected_bundle_sha256 is not None and (
        len(arguments.expected_bundle_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in arguments.expected_bundle_sha256
        )
    ):
        parser.error("expected-bundle-sha256 must be a lowercase SHA-256 digest")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if (
        arguments.dtype == "bfloat16"
        and arguments.device == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        parser.error("CUDA bfloat16 was requested but is unsupported")

    bundle = arguments.bundle.resolve()
    binding = NativeTokenizerBinding.load(bundle / "tokenizer")
    config = bound_native_65k_config(binding)
    device = torch.device(arguments.device)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model = EngineeringModel(config).to(device="cpu", dtype=dtype)
    state = load_inference_bundle(
        bundle,
        model,
        expected_task_database_identity_sha256=None,
        expected_binding_sha256=binding.binding_sha256,
        expected_model_fingerprint=config.fingerprint(),
        expected_bundle_sha256=arguments.expected_bundle_sha256,
    )
    model = model.to(device=device).eval()
    prompt_ids = build_inference_prompt(
        binding,
        task=arguments.task,
        prompt=arguments.prompt,
    )
    if len(prompt_ids) + arguments.max_new_tokens > config.max_sequence:
        parser.error("prompt plus generated tokens exceeds the model context")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        ids,
        max_new_tokens=arguments.max_new_tokens,
        eos_token_id=task_end_token_id(binding, arguments.task),
    )
    all_ids = generated[0].tolist()
    new_ids = all_ids[len(prompt_ids) :]
    response_ids = response_payload_token_ids(binding, arguments.task, new_ids)
    payload = {
        "bundle_lineage_verified": True,
        "bundle_sha256": state.bundle_sha256,
        "checkpoint_step": state.checkpoint_step,
        "device": device.type,
        "dtype": arguments.dtype,
        "generated_protocol_text": binding.decode(new_ids),
        "generated_text": binding.decode(response_ids),
        "model_fingerprint": state.model_fingerprint,
        "new_token_count": len(new_ids),
        "response_protocol_complete": response_protocol_complete(
            binding,
            task=arguments.task,
            generated_ids=new_ids,
        ),
        "task": arguments.task,
        "task_database_identity_sha256": state.task_database_identity_sha256,
        "tokenizer_sha256": state.tokenizer_sha256,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
