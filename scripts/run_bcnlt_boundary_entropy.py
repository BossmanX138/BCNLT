from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bcnlt.adaptive_vocab.boundary_entropy import collect_boundary_report
from bcnlt.tokenizer import BCNLTTokenizer
from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import (
    QLoRAConfig,
    TrainingLineage,
    _default_definition,
    load_bcnlt_adapter_state,
    load_qlora_model,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only BCNLT boundary entropy diagnostic")
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--v4-adapter", required=True, type=Path)
    parser.add_argument("--v1-tokenizer-artifact", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--heldout-documents", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        lineage = json.loads(args.lineage.read_text(encoding="utf-8"))
        if lineage.get("schema_version") != "bcnlt-adaptive-vocab-lineage-v1":
            raise ValueError("lineage_mismatch: unsupported lineage")
        documents = [
            json.loads(line)
            for line in args.heldout_documents.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        config_raw = json.loads(args.training_config.read_text(encoding="utf-8"))
        config = QLoRAConfig(
            target_modules=tuple(config_raw["target_modules"]),
            rank=int(config_raw.get("rank", 16)),
            alpha=int(config_raw.get("alpha", 32)),
            dropout=float(config_raw.get("dropout", 0.0)),
            maximum_sequence_tokens=int(config_raw.get("maximum_sequence_tokens", 4096)),
            learning_rate=float(config_raw.get("learning_rate", 2e-4)),
            gradient_accumulation_steps=int(config_raw.get("gradient_accumulation_steps", 8)),
        )
        v1 = BCNLTTokenizer.from_pretrained(
            args.v1_tokenizer_artifact, args.base_model
        )
        definitions = {
            item.token: _default_definition(item.token)
            for item in v1.manifest.added_tokens
        }
        adapter_lineage = TrainingLineage(
            **json.loads((args.v4_adapter / "lineage.json").read_text(encoding="utf-8"))
        )
        v4_model, selected_engine = load_qlora_model(
            args.base_model,
            manifest=v1.manifest,
            definitions=definitions,
            config=config,
            seed=1729,
            engine=adapter_lineage.engine,
        )
        if selected_engine != adapter_lineage.engine:
            raise ValueError("v4 adapter engine does not match the loaded model")
        load_bcnlt_adapter_state(v4_model, args.v4_adapter, expected=adapter_lineage)
        v4_model.eval()
        for parameter in v4_model.parameters():
            parameter.requires_grad_(False)

        from transformers import AutoModelForCausalLM

        base_control = AutoModelForCausalLM.from_pretrained(
            args.base_model, local_files_only=True
        )
        base_control.eval()
        collect_boundary_report(
            v4_model=v4_model,
            base_model=base_control,
            v1_tokenizer=v1,
            raw_tokenizer=v1.base,
            heldout_documents=documents,
            output=args.output,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
