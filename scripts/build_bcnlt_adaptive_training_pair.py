from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bcnlt.adaptive_vocab.hf_artifact import load_adaptive_tokenizer_artifact
from bcnlt.adaptive_vocab.training import (
    PairedTrainingRecord,
    select_paired_training_records,
)
from bcnlt.canonical import record_from_canonical_bytes
from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.corpus.packing import pack_training_shards
from bcnlt.tokenizer import BCNLTTokenizer


def _records(path: Path) -> tuple[CurriculumRecord, ...]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        result.append(
            CurriculumRecord(
                record_id=raw["record_id"], family=raw["family"],
                template_id=raw["template_id"],
                source_document_ids=tuple(raw["source_document_ids"]),
                prompt_mode=raw["prompt_mode"], target_mode=raw["target_mode"],
                prompt=raw["prompt"], target=raw["target"],
                provenance=tuple(sorted(raw["provenance"].items())), split=raw["split"],
            )
        )
    return tuple(result)


def _encode(tokenizer, mode: str, value: str) -> list[int]:
    if mode == "raw_code":
        return tokenizer.encode_raw_code(value)
    if mode == "semantic_code":
        semantic = tuple(
            record_from_canonical_bytes((line + "\n").encode("utf-8"))
            for line in value.splitlines() if line.startswith("{")
        )
        return tokenizer.encode_records(semantic)
    return tokenizer.encode_text(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lineage_identity(path: Path) -> str:
    raw = json.loads(path.read_text(encoding="utf-8"))
    identity = raw.get("identity_sha256") if isinstance(raw, dict) else None
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError(f"lineage has no valid identity_sha256: {path}")
    return identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build matched BCNLT v1/v5 training datasets")
    parser.add_argument("--accepted-records", required=True, type=Path)
    parser.add_argument("--v1-artifact", required=True, type=Path)
    parser.add_argument("--adaptive-artifact", required=True, type=Path)
    parser.add_argument("--base-tokenizer", required=True, type=Path)
    parser.add_argument("--selected-inventory", required=True, type=Path)
    parser.add_argument("--v4-lineage", required=True, type=Path)
    parser.add_argument("--corpus-lineage", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--maximum-examples", type=int, default=10_000)
    parser.add_argument("--minimum-macro-occurrences", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        print(f"paired output already exists: {args.output_root}", file=sys.stderr)
        return 2
    try:
        selected = json.loads(args.selected_inventory.read_text(encoding="utf-8"))
        if selected.get("training_eligible") is not True:
            raise ValueError("no_inventory_passed")
        v4_lineage_identity = _lineage_identity(args.v4_lineage)
        corpus_lineage_identity = _lineage_identity(args.corpus_lineage)
        v1 = BCNLTTokenizer.from_pretrained(args.v1_artifact, args.base_tokenizer)
        adaptive = load_adaptive_tokenizer_artifact(args.adaptive_artifact, args.base_tokenizer)
        if selected.get("selected_manifest_sha256") != adaptive.manifest.manifest_sha256:
            raise ValueError("selected tokenizer manifest does not match adaptive artifact")
        records = _records(args.accepted_records)
        paired = []
        for record in records:
            v1_ids = tuple(
                _encode(v1, record.prompt_mode, record.prompt)
                + _encode(v1, record.target_mode, record.target)
            )
            adaptive_ids = tuple(
                _encode(adaptive, record.prompt_mode, record.prompt)
                + _encode(adaptive, record.target_mode, record.target)
            )
            paired.append(
                PairedTrainingRecord(record, v1_ids, adaptive_ids, record.target.encode("utf-8"))
            )
        selection = select_paired_training_records(
            paired,
            adaptive.manifest,
            minimum_macro_occurrences=args.minimum_macro_occurrences,
            maximum_examples=args.maximum_examples,
            seed=args.seed,
        )
        args.output_root.mkdir(parents=True)
        selection_payload = selection.to_dict()
        selection_payload.update(
            {
                "v4_lineage_identity_sha256": v4_lineage_identity,
                "v4_lineage_file_sha256": _sha256(args.v4_lineage),
                "corpus_lineage_identity_sha256": corpus_lineage_identity,
                "corpus_lineage_file_sha256": _sha256(args.corpus_lineage),
                "selected_inventory_sha256": _sha256(args.selected_inventory),
                "selected_manifest_sha256": adaptive.manifest.manifest_sha256,
            }
        )
        (args.output_root / "paired-selection.json").write_text(
            json.dumps(selection_payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8", newline="\n",
        )
        if selection.inventory_reevaluation_required:
            return 2
        chosen = tuple(item.record for item in selection.records)
        pack_training_shards(chosen, v1, args.output_root / "v1")
        pack_training_shards(chosen, adaptive, args.output_root / "v5")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
