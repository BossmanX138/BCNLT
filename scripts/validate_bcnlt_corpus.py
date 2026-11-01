from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from bcnlt.corpus.admission import AdmissionPolicy
from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.corpus.database import CorpusDatabase
from bcnlt.corpus.streaming_admission import validate_corpus_streaming
from bcnlt.tokenizer import BCNLTTokenizer


def admission_policy_for_profile(profile: str) -> AdmissionPolicy:
    if profile == "full":
        return AdmissionPolicy()
    if profile == "smoke":
        return AdmissionPolicy(
            minimum_records=100,
            minimum_encoded_tokens=10_000,
            minimum_records_per_language=1,
            maximum_language_fraction=1.0,
            maximum_family_fraction=1.0,
            maximum_document_fraction=1.0,
            maximum_template_family_fraction=1.0,
        )
    raise ValueError(f"Unknown admission policy profile: {profile}")


def _load_curriculum(path: Path) -> Iterable[CurriculumRecord]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            try:
                yield CurriculumRecord(
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
                    language=raw.get(
                        "language", raw["provenance"]["canonical_language"]
                    ),
                    dialect=raw.get("dialect"),
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Invalid curriculum row {line_number}: {exc}") from exc


def _line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate BCNLT corpus admission gates")
    parser.add_argument("--database", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--accepted-output")
    parser.add_argument("--decisions-output")
    parser.add_argument("--holdout-hashes")
    parser.add_argument("--policy-profile", choices=("full", "smoke"), default="full")
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if report_path.exists():
        parser.error(f"report already exists: {report_path}")
    database = CorpusDatabase(args.database)
    documents = tuple(database.documents())
    database.close()
    policy = admission_policy_for_profile(args.policy_profile)
    if args.holdout_hashes:
        hashes = frozenset(
            line.strip()
            for line in Path(args.holdout_hashes).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        policy = replace(policy, holdout_hashes=hashes)
    tokenizer = BCNLTTokenizer.from_pretrained(
        args.tokenizer_artifact, args.base_tokenizer
    )
    curriculum_path = Path(args.curriculum)
    candidate_count = _line_count(curriculum_path)
    accepted_path = Path(args.accepted_output) if args.accepted_output else None
    decisions_path = (
        Path(args.decisions_output)
        if args.decisions_output
        else report_path.with_name(report_path.stem + "-decisions.jsonl")
    )
    for output in (accepted_path, decisions_path):
        if output is not None and output.exists():
            parser.error(f"output already exists: {output}")
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_destination = accepted_path or Path(os.devnull)
    with tempfile.TemporaryDirectory(prefix="bcnlt-validation-") as state_root:
        with accepted_destination.open(
            "x" if accepted_path else "w", encoding="utf-8", newline="\n"
        ) as accepted_handle, decisions_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as decisions_handle:
            report = validate_corpus_streaming(
                documents,
                _load_curriculum(curriculum_path),
                candidate_count=candidate_count,
                tokenizer=tokenizer,
                policy=policy,
                accepted_handle=accepted_handle,
                decisions_handle=decisions_handle,
                state_database=Path(state_root) / "admission-state.sqlite3",
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = report.to_dict()
    report_payload["policy_profile"] = args.policy_profile
    report_payload["decisions_artifact"] = {
        "path": str(decisions_path),
        "sha256": _sha256(decisions_path),
    }
    if accepted_path is not None:
        report_payload["accepted_artifact"] = {
            "path": str(accepted_path),
            "sha256": _sha256(accepted_path),
        }
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"passed": report.passed, "report": str(report_path)}, sort_keys=True))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
