from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an untrained BCNLT extension and a saved smoke adapter"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--packed-dataset", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--expected-lineage", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", default="configs/bcnlt_tokenizer_eval_v1.json")
    parser.add_argument("--maximum-heldout-examples", type=int)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")

    from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import (
        QLoRAConfig,
        TrainingLineage,
    )

    expected = TrainingLineage(
        **json.loads(Path(args.expected_lineage).read_text(encoding="utf-8"))
    )
    if expected.engine == "unsloth":
        import unsloth  # noqa: F401

    import torch

    from bcnlt.corpus.packing import PackedBCNLTDataset
    from bcnlt.tokenizer import BCNLTTokenizer
    from bcnlt.tokenizer_eval import (
        decode_generated_stream,
        evaluate_model,
        evaluate_promotion,
        load_evaluation_suite,
    )
    from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import (
        _default_definition,
        evaluate_packed_loss,
        hash_artifact,
        load_bcnlt_adapter_state,
        load_qlora_model,
        verify_training_preflight,
    )

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
        engine=expected.engine,
    )
    observed = TrainingLineage.from_artifacts(
        base_model=args.base_model,
        tokenizer_manifest=tokenizer_root / "bcnlt_tokenizer_manifest.json",
        corpus_manifest=Path(args.packed_dataset) / "dataset_manifest.json",
        training_config=args.training_config,
        engine=engine,
    )
    verify_training_preflight(
        model,
        expected=expected,
        observed=observed,
        cuda_available=torch.cuda.is_available(),
    )
    heldout = PackedBCNLTDataset(
        args.packed_dataset,
        "validation",
        expected_tokenizer_digest=tokenizer.manifest.manifest_sha256,
    )
    maximum_examples = args.maximum_heldout_examples
    heldout_count = len(heldout) if maximum_examples is None else min(len(heldout), maximum_examples)
    suite = load_evaluation_suite(args.suite)

    def generate(prompt, settings):
        torch.manual_seed(settings.seed)
        prompt_ids = tokenizer.encode_text(prompt)
        embedding = model.get_input_embeddings()
        base_embedding = getattr(embedding, "base", embedding)
        device = base_embedding.weight.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            result = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=None,
                max_new_tokens=settings.max_new_tokens,
                pad_token_id=model.config.eos_token_id,
            )
        generated = result[0, input_ids.shape[1] :].tolist()
        return decode_generated_stream(tokenizer, generated)

    baseline_loss = evaluate_packed_loss(
        model, heldout, maximum_examples=maximum_examples
    )
    baseline = evaluate_model(
        "runeforge-mk2-bcnlt-untrained-extension",
        suite.cases,
        suite.settings,
        generate,
        heldout_loss=baseline_loss,
    )

    adapter_lineage = load_bcnlt_adapter_state(model, args.adapter, expected=expected)
    adapter_loss = evaluate_packed_loss(
        model, heldout, maximum_examples=maximum_examples
    )
    adapter = evaluate_model(
        "runeforge-mk2-bcnlt-smoke-adapter",
        suite.cases,
        suite.settings,
        generate,
        heldout_loss=adapter_loss,
    )
    promotion = evaluate_promotion(baseline, adapter)
    artifact_manifest = json.loads(
        (Path(args.adapter) / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": "bcnlt-smoke-evaluation-v1",
        "baseline": baseline.to_dict(),
        "adapter": adapter.to_dict(),
        "promotion": promotion.to_dict(),
        "provenance": {
            "adapter_lineage": adapter_lineage.to_dict(),
            "adapter_weights_sha256": artifact_manifest["adapter_weights_sha256"],
            "base_model_sha256": hash_artifact(args.base_model),
            "heldout_examples": heldout_count,
            "packed_dataset_manifest_sha256": _sha256(
                Path(args.packed_dataset) / "dataset_manifest.json"
            ),
            "suite_sha256": _sha256(Path(args.suite)),
            "runtime": _runtime_versions(torch),
            "seed": args.seed,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if not baseline.runtime_failures and not adapter.runtime_failures else 2


def _runtime_versions(torch_module) -> dict[str, object]:
    packages = {}
    for name in (
        "unsloth",
        "transformers",
        "peft",
        "bitsandbytes",
        "tokenizers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_module.__version__,
        "cuda": torch_module.version.cuda,
        "packages": packages,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
