from __future__ import annotations

import argparse
import json
from pathlib import Path

from bcnlt.multilingual_eval import (
    evaluate_multilingual_promotion,
    load_language_evaluations,
    load_multilingual_eval_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply frozen BCNLT multilingual semantic promotion gates"
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/bcnlt_multilingual_eval_v1.json"
    )
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")
    config = load_multilingual_eval_config(args.config)
    thresholds = {key: float(value) for key, value in config["thresholds"].items()}
    decision = evaluate_multilingual_promotion(
        load_language_evaluations(args.baseline),
        load_language_evaluations(args.adapter),
        required_languages=tuple(config["languages"]),
        **thresholds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "bcnlt-multilingual-promotion-v1",
                "config": str(Path(args.config)),
                **decision.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"promoted": decision.promoted, "output": str(output)}, sort_keys=True))
    return 0 if decision.promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
