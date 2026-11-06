from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare packed BCNLT validation prompts with a framed text control"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--packed-dataset", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--expected-lineage", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")

    from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import (
        QLoRAConfig,
        TrainingLineage,
    )

    expected_lineage = TrainingLineage(
        **json.loads(Path(args.expected_lineage).read_text(encoding="utf-8"))
    )
    if expected_lineage.engine == "unsloth":
        import unsloth  # noqa: F401

    import torch

    from bcnlt.corpus.packing import PackedBCNLTDataset
    from bcnlt.tokenizer import BCNLTTokenizer
    from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import (
        _default_definition,
        load_bcnlt_adapter_state,
        load_qlora_model,
        verify_training_preflight,
    )

    config_raw = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
    config = QLoRAConfig(
        target_modules=tuple(config_raw["target_modules"]),
        rank=int(config_raw.get("lora_rank", config_raw.get("rank", 16))),
        alpha=int(config_raw.get("lora_alpha", config_raw.get("alpha", 32))),
        dropout=float(config_raw.get("lora_dropout", config_raw.get("dropout", 0.0))),
        maximum_sequence_tokens=int(config_raw.get("maximum_sequence_tokens", 4096)),
        learning_rate=float(config_raw.get("learning_rate", 2e-4)),
        gradient_accumulation_steps=int(config_raw.get("gradient_accumulation_steps", 8)),
    )
    tokenizer_root = Path(args.tokenizer_artifact)
    tokenizer = BCNLTTokenizer.from_pretrained(tokenizer_root, args.base_model)
    definitions = {
        item.token: _default_definition(item.token)
        for item in tokenizer.manifest.added_tokens
    }
    model, engine = load_qlora_model(
        args.base_model,
        manifest=tokenizer.manifest,
        definitions=definitions,
        config=config,
        seed=args.seed,
        engine=expected_lineage.engine,
    )
    verify_training_preflight(
        model,
        expected=expected_lineage,
        observed=TrainingLineage.from_artifacts(
            base_model=args.base_model,
            tokenizer_manifest=tokenizer_root / "bcnlt_tokenizer_manifest.json",
            corpus_manifest=Path(args.packed_dataset) / "dataset_manifest.json",
            training_config=args.training_config,
            engine=engine,
        ),
        cuda_available=torch.cuda.is_available(),
    )
    load_bcnlt_adapter_state(model, args.adapter, expected=expected_lineage)
    validation = PackedBCNLTDataset(
        args.packed_dataset,
        "validation",
        expected_tokenizer_digest=tokenizer.manifest.manifest_sha256,
    )

    selected: list[dict[str, object]] = []
    wanted = {"structural_reasoning", "semantic_to_code", "mixed_instruction"}
    for index in range(len(validation)):
        metadata = validation.metadata(index)
        family = metadata["family"]
        if family not in wanted or any(item["family"] == family for item in selected):
            continue
        packed = validation[index]
        labels = packed["labels"]
        prompt_length = next(
            (position for position, value in enumerate(labels) if value != -100),
            len(labels),
        )
        input_ids = list(packed["input_ids"])
        selected.append(
            {
                "case_id": f"packed-{family}",
                "family": family,
                "language": metadata["language"],
                "prompt_ids": input_ids[:prompt_length],
                "expected_ids": input_ids[prompt_length:],
                "source": "packed_validation",
            }
        )
        if {item["family"] for item in selected} == wanted:
            break

    selected.append(
        {
            "case_id": "framed-ordinary-instruction-control",
            "family": "ordinary_instruction",
            "language": "python",
            "prompt_ids": [
                *tokenizer.encode_text("Language: python\n"),
                *tokenizer.encode_text("Reply with the single word ready."),
            ],
            "expected_ids": tokenizer.encode_text("ready"),
            "source": "framed_control",
        }
    )

    embedding = model.get_input_embeddings()
    base_embedding = getattr(embedding, "base", embedding)
    device = base_embedding.weight.device
    stream_end = tokenizer.manifest.token_to_id["<|bcnlt:stream:end|>"]
    results = []
    for case in selected:
        prompt_ids = list(case.pop("prompt_ids"))
        expected_ids = list(case.pop("expected_ids"))
        torch.manual_seed(args.seed)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(inputs)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=None,
                max_new_tokens=max(64, len(expected_ids) + 32),
                pad_token_id=model.config.eos_token_id,
            )[0, inputs.shape[1] :].tolist()
        boundary = generated.index(stream_end) if stream_end in generated else None
        bounded = generated if boundary is None else generated[: boundary + 1]
        decode_error = None
        try:
            observed = tokenizer.decode(bounded)
        except Exception as exc:  # diagnostic output must preserve malformed generations
            observed = ""
            decode_error = f"{type(exc).__name__}: {exc}"
        expected_text = tokenizer.decode(expected_ids)
        results.append(
            {
                **case,
                "prompt_token_count": len(prompt_ids),
                "expected_token_count": len(expected_ids),
                "generated_token_count": len(generated),
                "stream_end_found": boundary is not None,
                "expected": expected_text,
                "observed": observed,
                "exact_match": decode_error is None and observed == expected_text,
                "decode_error": decode_error,
                "generated_ids": generated,
                "generated_structural_tokens": [
                    tokenizer.manifest.id_to_token.get(token_id)
                    for token_id in generated
                    if token_id >= tokenizer.manifest.base_vocab_size
                ],
            }
        )

    payload = {
        "schema_version": "bcnlt-smoke-alignment-diagnostic-v1",
        "adapter": str(Path(args.adapter)),
        "adapter_lineage": expected_lineage.to_dict(),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
