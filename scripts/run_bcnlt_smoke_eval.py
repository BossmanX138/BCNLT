from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from bcnlt.languages import CANONICAL_LANGUAGE_IDS
from bcnlt.multilingual_training import (
    TRAINING_FAMILIES,
    IndexedDatasetView,
    LanguageBalancedSampler,
)


def _select_heldout(
    dataset: Sequence[Mapping[str, Any]],
    *,
    examples_per_language: int | None,
    maximum_examples: int | None,
    seed: int,
) -> tuple[IndexedDatasetView, tuple[int, ...], dict[str, dict[str, int]]]:
    if examples_per_language is not None and maximum_examples is not None:
        raise ValueError(
            "--heldout-examples-per-language and --maximum-heldout-examples are mutually exclusive"
        )
    if examples_per_language is not None:
        if examples_per_language <= 0:
            raise ValueError("heldout examples per language must be positive")
        sampler = LanguageBalancedSampler(
            dataset,
            languages=CANONICAL_LANGUAGE_IDS,
            families=TRAINING_FAMILIES,
            seed=seed,
        )
        indices = sampler.sample_indices(
            examples_per_language * len(CANONICAL_LANGUAGE_IDS)
        )
        counts = sampler.balance_counts(indices)
        if set(counts["languages"].values()) != {examples_per_language}:
            raise ValueError("balanced evaluation selection is not exact by language")
    else:
        if maximum_examples is not None and maximum_examples <= 0:
            raise ValueError("maximum heldout examples must be positive")
        count = len(dataset) if maximum_examples is None else min(len(dataset), maximum_examples)
        indices = tuple(range(count))
        sampler = LanguageBalancedSampler(
            dataset,
            languages=CANONICAL_LANGUAGE_IDS,
            families=TRAINING_FAMILIES,
            seed=seed,
        )
        counts = sampler.balance_counts(indices)
    return IndexedDatasetView(dataset, indices), indices, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an untrained BCNLT extension and a saved smoke adapter"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--packed-dataset", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--expected-lineage", required=True)
    parser.add_argument("--adapter-expected-lineage")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", default="configs/bcnlt_tokenizer_eval_v1.json")
    parser.add_argument("--maximum-heldout-examples", type=int)
    parser.add_argument("--heldout-examples-per-language", type=int)
    parser.add_argument("--baseline-cache")
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
    from bcnlt.evaluation_cache import (
        BaselineCacheKey,
        load_baseline_cache,
        selection_sha256,
        write_baseline_cache,
    )
    from bcnlt.tokenizer import BCNLTTokenizer
    from bcnlt.tokenizer_eval import (
        decode_generated_stream,
        encode_evaluation_prompt,
        evaluate_model,
        evaluate_promotion,
        EvaluationRun,
        load_evaluation_suite,
    )
    from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import (
        _default_definition,
        evaluate_packed_loss,
        load_bcnlt_adapter_state,
        load_qlora_model,
        verify_continuation_parent,
        verify_training_preflight,
    )

    adapter_expected = (
        TrainingLineage(
            **json.loads(
                Path(args.adapter_expected_lineage).read_text(encoding="utf-8")
            )
        )
        if args.adapter_expected_lineage
        else expected
    )
    if adapter_expected != expected:
        verify_continuation_parent(child=adapter_expected, parent=expected)

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
    selected_heldout, selected_indices, heldout_counts = _select_heldout(
        heldout,
        examples_per_language=args.heldout_examples_per_language,
        maximum_examples=args.maximum_heldout_examples,
        seed=args.seed,
    )
    heldout_count = len(selected_heldout)
    heldout_selection_sha256 = selection_sha256(heldout, selected_indices)
    suite = load_evaluation_suite(args.suite)

    def generate(prompt, settings):
        torch.manual_seed(settings.seed)
        prompt_ids = encode_evaluation_prompt(tokenizer, prompt, language="python")
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

    baseline_cache_key = BaselineCacheKey(
        base_model_sha256=expected.base_model_sha256,
        tokenizer_manifest_sha256=expected.tokenizer_manifest_sha256,
        packed_dataset_manifest_sha256=expected.corpus_manifest_sha256,
        suite_sha256=_sha256(Path(args.suite)),
        selection_sha256=heldout_selection_sha256,
        seed=args.seed,
    )
    baseline_cache_path = Path(args.baseline_cache) if args.baseline_cache else None
    cached_baseline = (
        load_baseline_cache(baseline_cache_path, expected_key=baseline_cache_key)
        if baseline_cache_path is not None
        else None
    )
    if cached_baseline is not None:
        baseline = EvaluationRun.from_dict(cached_baseline)
        baseline_cache_status = "hit"
    else:
        if baseline_cache_path is not None and baseline_cache_path.exists():
            raise ValueError("baseline cache provenance does not match this evaluation")
        baseline_loss = evaluate_packed_loss(model, selected_heldout)
        baseline = evaluate_model(
            "runeforge-mk2-bcnlt-untrained-extension",
            suite.cases,
            suite.settings,
            generate,
            heldout_loss=baseline_loss,
        )
        if baseline_cache_path is not None:
            write_baseline_cache(
                baseline_cache_path,
                key=baseline_cache_key,
                baseline=baseline.to_dict(),
            )
            baseline_cache_status = "written"
        else:
            baseline_cache_status = "disabled"

    adapter_lineage = load_bcnlt_adapter_state(
        model, args.adapter, expected=adapter_expected
    )
    adapter_loss = evaluate_packed_loss(model, selected_heldout)
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
            "base_model_sha256": expected.base_model_sha256,
            "heldout_examples": heldout_count,
            "heldout_balance": heldout_counts,
            "heldout_selection_sha256": heldout_selection_sha256,
            "baseline_cache": {
                "path": str(baseline_cache_path) if baseline_cache_path else None,
                "status": baseline_cache_status,
            },
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
