from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from bcnlt.adaptive_vocab.hf_artifact import load_adaptive_tokenizer_artifact
from bcnlt.corpus.packing import PackedBCNLTDataset
from bcnlt.tokenizer import BCNLTTokenizer
from integrations.runeforge_mk2_tokenizer.embedding_extension import grow_bcnlt_extension
from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import (
    QLoRAConfig,
    TrainingLineage,
    _default_definition,
    load_bcnlt_adapter_state,
    load_qlora_model,
    evaluate_packed_loss,
    run_matched_continuation,
    save_bcnlt_adapter,
    verify_training_preflight,
)


def _config(path: Path) -> QLoRAConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return QLoRAConfig(
        target_modules=tuple(raw["target_modules"]),
        rank=int(raw.get("rank", 16)),
        alpha=int(raw.get("alpha", 32)),
        dropout=float(raw.get("dropout", 0.0)),
        maximum_sequence_tokens=int(raw.get("maximum_sequence_tokens", 4096)),
        learning_rate=float(raw.get("learning_rate", 2e-4)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 8)),
    )


def _trace_payload(trace) -> dict[str, object]:
    return {
        "losses": list(trace.losses),
        "optimizer_steps": trace.optimizer_steps,
        "completed_epochs": trace.completed_epochs,
        "validation_steps": list(trace.validation_steps),
        "run_identity": trace.run_identity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Continue matched BCNLT v1 and adaptive arms from the same v4 adapter"
    )
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--v4-adapter", required=True, type=Path)
    parser.add_argument("--v1-tokenizer", required=True, type=Path)
    parser.add_argument("--adaptive-tokenizer", required=True, type=Path)
    parser.add_argument("--v1-packed", required=True, type=Path)
    parser.add_argument("--adaptive-packed", required=True, type=Path)
    parser.add_argument("--v1-validation-packed", required=True, type=Path)
    parser.add_argument("--adaptive-validation-packed", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--maximum-optimizer-steps", type=int, required=True)
    parser.add_argument("--maximum-epochs", type=int, default=1)
    parser.add_argument("--validation-interval", type=int, default=64)
    parser.add_argument("--maximum-validation-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--run-identity", required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        print(f"paired training output already exists: {args.output_root}", file=sys.stderr)
        return 2

    try:
        config = _config(args.training_config)
        v1 = BCNLTTokenizer.from_pretrained(args.v1_tokenizer, args.base_model)
        adaptive = load_adaptive_tokenizer_artifact(
            args.adaptive_tokenizer, args.base_model
        )
        v1_definitions = {
            item.token: _default_definition(item.token)
            for item in v1.manifest.added_tokens
        }
        macro_definitions = {
            item.name: _default_definition(item.name)
            for item in adaptive.manifest.macros
        }
        prior = TrainingLineage(
            **json.loads((args.v4_adapter / "lineage.json").read_text(encoding="utf-8"))
        )
        v1_model, v1_engine = load_qlora_model(
            args.base_model,
            manifest=v1.manifest,
            definitions=v1_definitions,
            config=config,
            seed=args.seed,
            engine=prior.engine,
        )
        adaptive_model, adaptive_engine = load_qlora_model(
            args.base_model,
            manifest=v1.manifest,
            definitions=v1_definitions,
            config=config,
            seed=args.seed,
            engine=prior.engine,
        )
        if v1_engine != adaptive_engine or v1_engine != prior.engine:
            raise ValueError("paired models did not load with the v4 training engine")
        load_bcnlt_adapter_state(v1_model, args.v4_adapter, expected=prior)
        load_bcnlt_adapter_state(adaptive_model, args.v4_adapter, expected=prior)
        grow_bcnlt_extension(
            adaptive_model,
            v1.manifest,
            adaptive.manifest,
            macro_definitions,
            seed=args.seed,
        )

        v1_manifest_path = args.v1_tokenizer / "bcnlt_tokenizer_manifest.json"
        adaptive_manifest_path = args.adaptive_tokenizer / "bcnlt_adaptive_manifest.json"
        v1_lineage = TrainingLineage.from_artifacts(
            base_model=args.base_model,
            tokenizer_manifest=v1_manifest_path,
            corpus_manifest=args.v1_packed / "dataset_manifest.json",
            training_config=args.training_config,
            engine=v1_engine,
        )
        adaptive_lineage = TrainingLineage.from_artifacts(
            base_model=args.base_model,
            tokenizer_manifest=adaptive_manifest_path,
            corpus_manifest=args.adaptive_packed / "dataset_manifest.json",
            training_config=args.training_config,
            engine=adaptive_engine,
        )
        verify_training_preflight(
            v1_model,
            expected=v1_lineage,
            observed=v1_lineage,
            cuda_available=torch.cuda.is_available(),
        )
        verify_training_preflight(
            adaptive_model,
            expected=adaptive_lineage,
            observed=adaptive_lineage,
            cuda_available=torch.cuda.is_available(),
        )
        v1_data = PackedBCNLTDataset(
            args.v1_packed,
            args.split,
            expected_tokenizer_digest=v1.manifest.manifest_sha256,
        )
        adaptive_data = PackedBCNLTDataset(
            args.adaptive_packed,
            args.split,
            expected_tokenizer_digest=adaptive.manifest.manifest_sha256,
        )
        v1_validation = PackedBCNLTDataset(
            args.v1_validation_packed,
            "validation",
            expected_tokenizer_digest=v1.manifest.manifest_sha256,
        )
        adaptive_validation = PackedBCNLTDataset(
            args.adaptive_validation_packed,
            "validation",
            expected_tokenizer_digest=adaptive.manifest.manifest_sha256,
        )
        if tuple(v1_validation[index]["record_id"] for index in range(len(v1_validation))) != tuple(
            adaptive_validation[index]["record_id"]
            for index in range(len(adaptive_validation))
        ):
            raise ValueError("paired validation datasets have different record order")
        if (
            v1_data.manifest.maximum_sequence_tokens > config.maximum_sequence_tokens
            or adaptive_data.manifest.maximum_sequence_tokens
            > config.maximum_sequence_tokens
        ):
            raise ValueError("packed training sequence limit exceeds model configuration")
        validation_values: dict[str, list[dict[str, float | int]]] = {
            "v1": [],
            "adaptive": [],
        }

        def validate_v1(step: int) -> None:
            validation_values["v1"].append(
                {
                    "step": step,
                    "loss": evaluate_packed_loss(
                        v1_model,
                        v1_validation,
                        maximum_examples=args.maximum_validation_examples,
                    ),
                }
            )

        def validate_adaptive(step: int) -> None:
            validation_values["adaptive"].append(
                {
                    "step": step,
                    "loss": evaluate_packed_loss(
                        adaptive_model,
                        adaptive_validation,
                        maximum_examples=args.maximum_validation_examples,
                    ),
                }
            )
        left, right = run_matched_continuation(
            v1_model,
            adaptive_model,
            v1_data,
            adaptive_data,
            config,
            maximum_optimizer_steps=args.maximum_optimizer_steps,
            maximum_epochs=args.maximum_epochs,
            validation_interval=args.validation_interval,
            v1_validation_callback=validate_v1,
            adaptive_validation_callback=validate_adaptive,
            run_identity=args.run_identity,
            require_cuda=True,
        )
        args.output_root.mkdir(parents=True)
        left_artifact = save_bcnlt_adapter(
            v1_model, args.output_root / "v1", args.v1_tokenizer, v1_lineage
        )
        right_artifact = save_bcnlt_adapter(
            adaptive_model,
            args.output_root / "adaptive", args.adaptive_tokenizer, adaptive_lineage
        )
        (args.output_root / "matched-training.json").write_text(
            json.dumps(
                {
                    "schema_version": "bcnlt-matched-continuation-v1",
                    "run_identity": args.run_identity,
                    "v4_lineage": prior.to_dict(),
                    "v1": {
                        "trace": _trace_payload(left),
                        "validation": validation_values["v1"],
                        "adapter_weights_sha256": left_artifact.weights_sha256,
                    },
                    "adaptive": {
                        "trace": _trace_payload(right),
                        "validation": validation_values["adaptive"],
                        "adapter_weights_sha256": right_artifact.weights_sha256,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
