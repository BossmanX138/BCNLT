from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bcnlt.adaptive_vocab.boundary_entropy import collect_boundary_report
from bcnlt.tokenizer import BCNLTTokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only BCNLT boundary entropy diagnostic")
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--v4-adapter", required=True, type=Path)
    parser.add_argument("--v1-tokenizer-artifact", required=True, type=Path)
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
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        base_control = AutoModelForCausalLM.from_pretrained(
            args.base_model, local_files_only=True
        )
        adapter_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, local_files_only=True
        )
        v4_model = PeftModel.from_pretrained(
            adapter_base, args.v4_adapter, is_trainable=False
        )
        v1 = BCNLTTokenizer.from_pretrained(
            args.v1_tokenizer_artifact, args.base_model
        )
        collect_boundary_report(
            v4_model=v4_model,
            base_model=base_control,
            v1_tokenizer=v1,
            raw_tokenizer=v1.base,
            heldout_documents=documents,
            output=args.output,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
