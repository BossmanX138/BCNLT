from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from bcnlt.adaptive_vocab.config import AdaptiveVocabularyConfig
from bcnlt.adaptive_vocab.hf_artifact import load_adaptive_tokenizer_artifact
from bcnlt.adaptive_vocab.tokenizer_eval import (
    apply_platform_artifact_digests,
    evaluate_tokenizer,
    select_inventory,
)
from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.tokenizer import BCNLTTokenizer


def _records(path: Path) -> tuple[CurriculumRecord, ...]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            raw = json.loads(line)
            result.append(
                CurriculumRecord(
                    record_id=raw["record_id"],
                    family=raw["family"],
                    template_id=raw["template_id"],
                    source_document_ids=tuple(raw["source_document_ids"]),
                    prompt_mode=raw["prompt_mode"],
                    target_mode=raw["target_mode"],
                    prompt=raw["prompt"],
                    target=raw["target"],
                    provenance=tuple(sorted(raw["provenance"].items())),
                    split=raw["split"],
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid evaluation record {line_number}: {exc}") from exc
    return tuple(result)


def _write_new(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"evaluation output already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate BCNLT adaptive tokenizer inventories")
    parser.add_argument("--v1-artifact", required=True, type=Path)
    parser.add_argument("--base-tokenizer", required=True, type=Path)
    parser.add_argument("--adaptive-artifact", required=True, action="append", type=Path)
    parser.add_argument("--train-records", required=True, type=Path)
    parser.add_argument("--heldout-records", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--platform-digests", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        config = AdaptiveVocabularyConfig.load(args.config)
        train = _records(args.train_records)
        heldout = _records(args.heldout_records)
        platform = json.loads(args.platform_digests.read_text(encoding="utf-8"))
        v1 = BCNLTTokenizer.from_pretrained(args.v1_artifact, args.base_tokenizer)
        evaluations = []
        for artifact in args.adaptive_artifact:
            adaptive = load_adaptive_tokenizer_artifact(artifact, args.base_tokenizer)
            evaluation = evaluate_tokenizer(v1, adaptive, train, heldout, config)
            supplied = platform.get(evaluation.manifest_sha256, {})
            if not isinstance(supplied, dict):
                supplied = {}
            evaluations.append(apply_platform_artifact_digests(evaluation, supplied))
        selection = select_inventory(evaluations, config)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        destinations = [
            args.output_root / f"tokenizer-eval-{item.budget:04d}.json"
            for item in evaluations
        ]
        selection_path = args.output_root / "selected-inventory.json"
        if any(path.exists() for path in (*destinations, selection_path)):
            raise FileExistsError("tokenizer evaluation output already exists")
        for path, evaluation in zip(destinations, evaluations, strict=True):
            _write_new(path, evaluation.to_dict())
        _write_new(selection_path, selection.to_dict())
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if selection.training_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
