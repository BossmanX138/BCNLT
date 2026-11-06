from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


def select_candidate(
    *,
    parent_loss: float,
    candidates: Sequence[Mapping[str, object]],
    maximum_relative_perplexity_regression: float,
) -> dict[str, object]:
    if parent_loss < 0 or maximum_relative_perplexity_regression < 0:
        raise ValueError("loss and perplexity regression threshold must be non-negative")
    parent_perplexity = math.exp(parent_loss)
    assessed = []
    for value in candidates:
        item = dict(value)
        loss = float(item["heldout_loss"])
        steps = int(item["steps"])
        if loss < 0 or steps <= 0:
            raise ValueError("candidate loss and steps must be positive")
        perplexity = math.exp(loss)
        regression = perplexity / parent_perplexity - 1.0
        item["perplexity"] = perplexity
        item["relative_perplexity_regression"] = regression
        if not bool(item["promoted"]):
            item["status"] = "rejected_behavior"
        elif regression > maximum_relative_perplexity_regression:
            item["status"] = "rejected_retention"
        else:
            item["status"] = "accepted"
        assessed.append(item)
    eligible = [item for item in assessed if item["status"] == "accepted"]
    selected = min(
        eligible,
        key=lambda item: (int(item["steps"]), float(item["heldout_loss"])),
        default=None,
    )
    return {
        "schema_version": "bcnlt-adapter-selection-v1",
        "method": {
            "controlled_variable": "continuation_optimizer_steps",
            "hard_gate": "all frozen behavior categories and runtime checks pass",
            "retention_metric": "relative heldout perplexity regression",
            "maximum_relative_perplexity_regression": (
                maximum_relative_perplexity_regression
            ),
            "selection_rule": "smallest accepted intervention; heldout loss breaks ties",
        },
        "parent": {
            "heldout_loss": parent_loss,
            "perplexity": parent_perplexity,
        },
        "candidates": assessed,
        "selected": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply the evidence-first BCNLT adapter selection gate"
    )
    parser.add_argument("--parent-evaluation", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="STEPS=EVALUATION_JSON",
    )
    parser.add_argument("--maximum-relative-perplexity-regression", type=float, default=0.02)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")
    parent_path = Path(args.parent_evaluation)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_fingerprint = _evaluation_fingerprint(parent)
    evidence = []
    for specification in args.candidate:
        step_text, separator, path_text = specification.partition("=")
        if not separator:
            parser.error(f"invalid candidate specification: {specification}")
        path = Path(path_text)
        report = json.loads(path.read_text(encoding="utf-8"))
        if _evaluation_fingerprint(report) != parent_fingerprint:
            raise ValueError(f"candidate evaluation population differs: {path}")
        adapter = report["adapter"]
        promotion = report["promotion"]
        evidence.append(
            {
                "steps": int(step_text),
                "evaluation_path": str(path),
                "adapter_weights_sha256": report["provenance"]["adapter_weights_sha256"],
                "heldout_loss": float(adapter["heldout_loss"]),
                "promoted": bool(promotion["promoted"]),
                "failure_codes": list(promotion["failure_codes"]),
                "metrics": dict(adapter["metrics"]),
                "runtime_failures": list(adapter["runtime_failures"]),
            }
        )
    decision = select_candidate(
        parent_loss=float(parent["adapter"]["heldout_loss"]),
        candidates=evidence,
        maximum_relative_perplexity_regression=(
            args.maximum_relative_perplexity_regression
        ),
    )
    decision["provenance"] = {
        "parent_evaluation": str(parent_path),
        "evaluation_fingerprint": parent_fingerprint,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(decision, sort_keys=True))
    return 0 if decision["selected"] is not None else 2


def _evaluation_fingerprint(report: Mapping[str, object]) -> dict[str, object]:
    provenance = report["provenance"]
    return {
        key: provenance[key]
        for key in (
            "base_model_sha256",
            "heldout_examples",
            "heldout_selection_sha256",
            "packed_dataset_manifest_sha256",
            "seed",
            "suite_sha256",
        )
    }


if __name__ == "__main__":
    raise SystemExit(main())
