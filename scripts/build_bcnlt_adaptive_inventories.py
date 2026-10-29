from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

from bcnlt.adaptive_vocab.candidates import (
    CandidateDiscoveryReport,
    CandidateRate,
    MacroCandidate,
    PayloadClassScore,
    SupportCount,
)
from bcnlt.adaptive_vocab.config import AdaptiveVocabularyConfig
from bcnlt.adaptive_vocab.hf_artifact import save_adaptive_tokenizer_artifact
from bcnlt.adaptive_vocab.manifest import AdaptiveVocabularyManifest, build_inventories
from bcnlt.adaptive_vocab.secrets import SensitiveTextFinding
from bcnlt.adaptive_vocab.tokenizer import AdaptiveBCNLTTokenizer
from bcnlt.tokenizer import BCNLTTokenizer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(raw: dict[str, object]) -> MacroCandidate:
    return MacroCandidate(
        expansion=tuple(raw["expansion"]),
        length=int(raw["length"]),
        score=float(raw["score"]),
        languages=tuple(raw["languages"]),
        repositories=tuple(raw["repositories"]),
        documents=tuple(raw["documents"]),
        raw_occurrences=int(raw["raw_occurrences"]),
        capped_occurrences=int(raw["capped_occurrences"]),
        repository_counts=tuple(
            SupportCount(str(item["identity"]), int(item["occurrences"]))
            for item in raw["repository_counts"]
        ),
        document_counts=tuple(
            SupportCount(str(item["identity"]), int(item["occurrences"]))
            for item in raw["document_counts"]
        ),
        rates=tuple(
            CandidateRate(
                language=str(item["language"]),
                payload_class=str(item["payload_class"]),
                capped_occurrences=int(item["capped_occurrences"]),
                eligible_base_tokens=int(item["eligible_base_tokens"]),
                rate_per_million=float(item["rate_per_million"]),
            )
            for item in raw["rates"]
        ),
        payload_class_scores=tuple(
            PayloadClassScore(
                str(item["payload_class"]), float(item["third_language_rate"])
            )
            for item in raw["payload_class_scores"]
        ),
        eligible=bool(raw["eligible"]),
        reason_codes=tuple(raw["reason_codes"]),
        sensitive_evidence=tuple(
            SensitiveTextFinding(
                reason_code=str(item["reason_code"]),
                category=str(item["category"]),
                evidence_sha256=str(item["evidence_sha256"]),
                character_count=int(item["character_count"]),
            )
            for item in raw["sensitive_evidence"]
        ),
    )


def _load_candidates(path: Path) -> tuple[MacroCandidate, ...]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            raw = json.loads(line)
            result.append(_candidate(raw))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid candidate row {line_number}: {exc}") from exc
    return tuple(result)


def write_inventory_artifact(
    destination: str | Path,
    v1_tokenizer: BCNLTTokenizer,
    manifest: AdaptiveVocabularyManifest,
) -> tuple[Path, ...]:
    """Write one independently loadable adaptive tokenizer artifact."""
    return save_adaptive_tokenizer_artifact(
        destination,
        AdaptiveBCNLTTokenizer(v1_tokenizer, manifest),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build append-only BCNLT macro inventories")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--tokenizer-artifact", required=True, type=Path)
    parser.add_argument("--base-tokenizer", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        summary = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
        lineage = json.loads(args.lineage.read_text(encoding="utf-8"))
        if summary.get("candidates_sha256") != _sha256(args.candidates):
            raise ValueError("candidate digest mismatch")
        if summary.get("lineage_identity_sha256") != lineage.get("identity_sha256"):
            raise ValueError("lineage_mismatch: candidate summary and lineage differ")
        config = AdaptiveVocabularyConfig.load(args.config)
        candidates = _load_candidates(args.candidates)
        report = CandidateDiscoveryReport(
            candidates=candidates,
            observation_count=int(summary["observation_count"]),
            eligible_base_tokens=tuple(
                (
                    str(item["language"]),
                    str(item["payload_class"]),
                    int(item["count"]),
                )
                for item in summary["eligible_base_tokens"]
            ),
            lineage_sha256=str(lineage["identity_sha256"]),
            candidate_report_sha256=_sha256(args.candidate_summary),
        )
        tokenizer = BCNLTTokenizer.from_pretrained(
            args.tokenizer_artifact, args.base_tokenizer
        )
        manifests = build_inventories(report, tokenizer.manifest, config)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        args.output_root.mkdir(parents=True, exist_ok=True)
        if "insufficient_eligible_candidates" in str(exc):
            no_pass = args.output_root / "no-pass.json"
            if not no_pass.exists():
                no_pass.write_text(
                    json.dumps(
                        {
                            "schema_version": "bcnlt-adaptive-inventory-no-pass-v1",
                            "passed": False,
                            "failure_codes": ["insufficient_eligible_candidates"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
        print(str(exc), file=sys.stderr)
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        destination = args.output_root / f"budget-{len(manifest.macros):04d}"
        if destination.exists():
            print(f"inventory output already exists: {destination}", file=sys.stderr)
            return 2
        temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
        temporary.mkdir()
        try:
            write_inventory_artifact(temporary, tokenizer, manifest)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
