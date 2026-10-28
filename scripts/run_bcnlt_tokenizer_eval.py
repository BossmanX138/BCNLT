from __future__ import annotations

import argparse
import json
from pathlib import Path

from bcnlt.tokenizer_eval import evaluate_model, load_evaluation_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen BCNLT tokenizer evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--heldout-loss", required=True, type=float)
    parser.add_argument("--suite", default="configs/bcnlt_tokenizer_eval_v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    suite = load_evaluation_suite(args.suite)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    quantization = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        if args.load_in_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quantization,
        trust_remote_code=False,
    )

    def generate(prompt, settings):
        torch.manual_seed(settings.seed)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            result = model.generate(
                **encoded,
                do_sample=False,
                temperature=None,
                max_new_tokens=settings.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = result[0, encoded["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    report = evaluate_model(
        args.identity,
        suite.cases,
        suite.settings,
        generate,
        heldout_loss=args.heldout_loss,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if not report.runtime_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
