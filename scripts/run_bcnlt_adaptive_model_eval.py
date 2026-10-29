from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _qlora_config(path: Path):
    from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import QLoRAConfig

    raw = _load_json(path)
    return QLoRAConfig(
        target_modules=tuple(raw["target_modules"]),
        rank=int(raw.get("rank", 16)),
        alpha=int(raw.get("alpha", 32)),
        dropout=float(raw.get("dropout", 0.0)),
        maximum_sequence_tokens=int(raw.get("maximum_sequence_tokens", 4096)),
        learning_rate=float(raw.get("learning_rate", 2e-4)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 8)),
    )


def _targets(path: Path) -> tuple[tuple[str, ...], tuple[bytes, ...]]:
    record_ids = []
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            raw = json.loads(line)
            record_ids.append(str(raw["record_id"]))
            values.append(bytes.fromhex(str(raw["target_utf8_hex"])))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid heldout target sidecar row {line_number}: {exc}") from exc
    if not record_ids or len(set(record_ids)) != len(record_ids) or any(not item for item in values):
        raise ValueError("heldout target sidecar must contain unique non-empty targets")
    return tuple(record_ids), tuple(values)


def _trace_losses(path: Path | None) -> tuple[float, ...]:
    if path is None:
        return ()
    raw = _load_json(path)
    losses = raw.get("losses")
    if not isinstance(losses, list):
        raise ValueError(f"training trace has no loss list: {path}")
    return tuple(float(value) for value in losses)


def _adapter_lineage(path: Path):
    from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import TrainingLineage

    return TrainingLineage(**_load_json(path / "lineage.json"))


def _generation(model, tokenizer):
    import torch

    from bcnlt.tokenizer_eval import decode_generated_stream

    def generate(prompt, settings):
        torch.manual_seed(settings.seed)
        prompt_ids = tokenizer.encode_text(prompt)
        embedding = model.get_input_embeddings()
        base_embedding = getattr(embedding, "base", embedding)
        device = base_embedding.weight.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            result = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                temperature=None,
                max_new_tokens=settings.max_new_tokens,
                pad_token_id=model.config.eos_token_id,
            )
        return decode_generated_stream(tokenizer, result[0, input_ids.shape[1] :].tolist())

    return generate


def _decorated_dataset(packed, record_ids, targets, maximum_examples):
    count = len(packed) if maximum_examples is None else min(len(packed), maximum_examples)
    if count != min(len(record_ids), len(targets)):
        raise ValueError("packed heldout rows and target sidecar differ in length")
    rows = []
    for index in range(count):
        rows.append(
            {
                **packed[index],
                "record_id": record_ids[index],
                "target_bytes": targets[index],
            }
        )
    return tuple(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate matched BCNLT v4/v5 model controls")
    parser.add_argument("--experiment-spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--maximum-heldout-examples", type=int)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        parser.error(f"output root already exists: {args.output_root}")

    try:
        spec = _load_json(args.experiment_spec)
        required = {
            "base_model", "v1_tokenizer", "adaptive_tokenizer", "v1_packed",
            "v5_packed", "frozen_v4_adapter", "v1_control_adapter",
            "trained_v5_adapter", "training_config", "heldout_targets",
            "tokenizer_selection", "suite",
        }
        if not required.issubset(spec):
            raise ValueError(f"experiment spec missing keys: {sorted(required - set(spec))}")
        paths = {key: Path(str(spec[key])) for key in required if key != "base_model"}
        base_model = str(spec["base_model"])

        from bcnlt.adaptive_vocab.hf_artifact import load_adaptive_tokenizer_artifact
        from bcnlt.adaptive_vocab.model_eval import (
            AdaptiveEvaluationConfig, AdaptiveModelInput, PairedHeldout,
            evaluate_adaptive_matrix, evaluate_adaptive_promotion,
        )
        from bcnlt.corpus.packing import PackedBCNLTDataset
        from bcnlt.tokenizer import BCNLTTokenizer
        from bcnlt.tokenizer_eval import load_evaluation_suite
        from integrations.runeforge_mk2_tokenizer.embedding_extension import grow_bcnlt_extension
        from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import (
            _default_definition, load_bcnlt_adapter_state, load_qlora_model,
        )

        config = _qlora_config(paths["training_config"])
        v1 = BCNLTTokenizer.from_pretrained(paths["v1_tokenizer"], base_model)
        adaptive = load_adaptive_tokenizer_artifact(paths["adaptive_tokenizer"], base_model)
        if adaptive.manifest.v1_manifest_sha256 != v1.manifest.manifest_sha256:
            raise ValueError("adaptive tokenizer does not extend the supplied v1 tokenizer")
        selection = _load_json(paths["tokenizer_selection"])
        if selection.get("selected_manifest_sha256") != adaptive.manifest.manifest_sha256:
            raise ValueError("tokenizer selection does not identify the adaptive artifact")
        suite = load_evaluation_suite(paths["suite"])
        record_ids, targets = _targets(paths["heldout_targets"])
        if args.maximum_heldout_examples is not None:
            record_ids = record_ids[: args.maximum_heldout_examples]
            targets = targets[: args.maximum_heldout_examples]
        paired = PairedHeldout(record_ids, targets)
        v1_packed = PackedBCNLTDataset(
            paths["v1_packed"], "validation",
            expected_tokenizer_digest=v1.manifest.manifest_sha256,
        )
        v5_packed = PackedBCNLTDataset(
            paths["v5_packed"], "validation",
            expected_tokenizer_digest=adaptive.manifest.manifest_sha256,
        )
        v1_rows = _decorated_dataset(v1_packed, record_ids, targets, args.maximum_heldout_examples)
        v5_rows = _decorated_dataset(v5_packed, record_ids, targets, args.maximum_heldout_examples)
        v1_definitions = {item.token: _default_definition(item.token) for item in v1.manifest.added_tokens}
        macro_definitions = {
            item.name: json.dumps(list(item.expansion), separators=(",", ":"))
            for item in adaptive.manifest.macros
        }

        def v1_model(adapter_path: Path):
            lineage = _adapter_lineage(adapter_path)
            model, _ = load_qlora_model(
                base_model, manifest=v1.manifest, definitions=v1_definitions,
                config=config, seed=int(spec.get("seed", 1729)), engine=lineage.engine,
            )
            load_bcnlt_adapter_state(model, adapter_path, expected=lineage)
            return model

        def v5_model(adapter_path: Path | None):
            model = v1_model(paths["frozen_v4_adapter"])
            grow_bcnlt_extension(
                model, v1.manifest, adaptive.manifest, macro_definitions,
                seed=int(spec.get("seed", 1729)),
            )
            if adapter_path is not None:
                load_bcnlt_adapter_state(model, adapter_path, expected=_adapter_lineage(adapter_path))
            return model

        frozen = v1_model(paths["frozen_v4_adapter"])
        untrained = v5_model(None)
        control = v1_model(paths["v1_control_adapter"])
        trained = v5_model(paths["trained_v5_adapter"])
        v1_tokens = sum(len(row["input_ids"]) for row in v1_rows)
        v5_tokens = sum(len(row["input_ids"]) for row in v5_rows)
        common = {"record_ids": record_ids, "target_bytes": targets, "adapter_reloadable": True}
        report = evaluate_adaptive_matrix(
            AdaptiveModelInput("frozen_v4", frozen, v1_rows, generate=_generation(frozen, v1), semantic_input_tokens=v1_tokens, **common),
            AdaptiveModelInput("untrained_v5", untrained, v5_rows, generate=_generation(untrained, adaptive), semantic_input_tokens=v5_tokens, **common),
            AdaptiveModelInput("v1_control", control, v1_rows, generate=_generation(control, v1), semantic_input_tokens=v1_tokens, training_losses=_trace_losses(Path(str(spec["v1_control_trace"])) if spec.get("v1_control_trace") else None), **common),
            AdaptiveModelInput("trained_v5", trained, v5_rows, generate=_generation(trained, adaptive), semantic_input_tokens=v5_tokens, training_losses=_trace_losses(Path(str(spec["trained_v5_trace"])) if spec.get("trained_v5_trace") else None), **common),
            suite, paired, AdaptiveEvaluationConfig(),
        )
        proposal = evaluate_adaptive_promotion(report, selection)
        args.output_root.mkdir(parents=True)
        (args.output_root / "model-evaluation.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        (args.output_root / "promotion-proposal.json").write_text(
            json.dumps(proposal.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if not report.failure_codes else 2


if __name__ == "__main__":
    raise SystemExit(main())
