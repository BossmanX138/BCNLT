from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import replace
from pathlib import Path

from bcnlt.adaptive_vocab.candidates import (
    discover_candidates,
    iter_payload_observations,
)
from bcnlt.adaptive_vocab.config import AdaptiveVocabularyConfig
from bcnlt.adaptive_vocab.lineage import LINEAGE_SCHEMA_VERSION, LineageArtifact
from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.tokenizer import BCNLTTokenizer


def _load_curriculum(path: Path) -> tuple[CurriculumRecord, ...]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            raw = json.loads(line)
            records.append(
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
            raise ValueError(f"invalid accepted record row {line_number}: {exc}") from exc
    return tuple(records)


def _canonical_line(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"candidate output already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(content)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover balanced BCNLT payload macros")
    parser.add_argument("--accepted-records", required=True, type=Path)
    parser.add_argument("--tokenizer-artifact", required=True, type=Path)
    parser.add_argument("--base-tokenizer", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        lineage = json.loads(args.lineage.read_text(encoding="utf-8"))
        if not isinstance(audit, dict) or audit.get("passed") is not True:
            raise ValueError("insufficient_shared_candidate_diversity: corpus audit did not pass")
        if (
            not isinstance(lineage, dict)
            or lineage.get("schema_version") != LINEAGE_SCHEMA_VERSION
            or audit.get("lineage_identity_sha256") != lineage.get("identity_sha256")
        ):
            raise ValueError("lineage_mismatch: audit and lineage do not match")
        expected_config_hash = next(
            (
                item.get("sha256")
                for item in lineage.get("artifacts", [])
                if isinstance(item, dict) and item.get("name") == "config"
            ),
            None,
        )
        if (
            expected_config_hash is None
            or LineageArtifact.capture("config", args.config).sha256
            != expected_config_hash
        ):
            raise ValueError("lineage_mismatch: adaptive configuration changed")
        config = AdaptiveVocabularyConfig.load(args.config)
        tokenizer = BCNLTTokenizer.from_pretrained(
            args.tokenizer_artifact, args.base_tokenizer
        )
        records = _load_curriculum(args.accepted_records)
        report = discover_candidates(
            iter_payload_observations(records, tokenizer),
            config,
        )
        report = replace(report, lineage_sha256=str(lineage["identity_sha256"]))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates_bytes = b"".join(
        _canonical_line(item.to_dict()) for item in report.eligible_candidates
    )
    rejections_bytes = b"".join(
        _canonical_line(item.to_dict()) for item in report.rejected_candidates
    )
    summary = report.to_dict()
    summary.update(
        {
            "lineage_identity_sha256": lineage["identity_sha256"],
            "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
            "candidates_sha256": hashlib.sha256(candidates_bytes).hexdigest(),
            "candidate_rejections_sha256": hashlib.sha256(rejections_bytes).hexdigest(),
        }
    )
    destinations = (
        args.output_root / "candidates.jsonl",
        args.output_root / "candidate-rejections.jsonl",
        args.output_root / "candidate-summary.json",
    )
    if any(path.exists() for path in destinations):
        print("candidate output already exists", file=sys.stderr)
        return 2
    try:
        _write_new(destinations[0], candidates_bytes)
        _write_new(destinations[1], rejections_bytes)
        _write_new(destinations[2], _canonical_line(summary))
    except (OSError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
