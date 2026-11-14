"""Export a completed training checkpoint as an immutable inference bundle."""
from __future__ import annotations

import argparse
import json
import sys
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

from bosscrafts_model.config import ModelConfig
from bosscrafts_model.inference_bundle import export_inference_bundle
from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.resumable_task_packing import audit_task_database
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-balanced-curriculum-v1"
    / "tasks.sqlite"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "bosscrafts-engineering-v1" / "checkpoint.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "models" / "BossCrafts-Engineering-0.5B-v1"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--full-model", action="store_true")
    arguments = parser.parse_args()

    binding = NativeTokenizerBinding.load()
    audit = audit_task_database(binding, arguments.database)
    config = (
        bound_native_65k_config(binding)
        if arguments.full_model
        else ModelConfig.tiny(
            vocab_size=binding.vocab_size,
            tokenizer_manifest=binding.binding_sha256,
        )
    )
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model = EngineeringModel(config).to(device="cpu", dtype=dtype)
    status = export_inference_bundle(
        arguments.checkpoint,
        arguments.output,
        model,
        binding=binding,
        expected_task_database_identity_sha256=audit.database_identity_sha256,
        expected_model_fingerprint=config.fingerprint(),
    )
    payload = asdict(status)
    payload["bundle_path"] = str(status.bundle_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
