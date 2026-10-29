from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from bcnlt.corpus.admission import AdmissionPolicy, CorpusValidationInput, validate_corpus
from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.corpus.database import CorpusDatabase
from bcnlt.tokenizer import BCNLTTokenizer


def admission_policy_for_profile(profile: str) -> AdmissionPolicy:
    if profile == "full":
        return AdmissionPolicy()
    if profile == "smoke":
        return AdmissionPolicy(
            minimum_records=100,
            minimum_encoded_tokens=10_000,
            maximum_document_fraction=1.0,
            maximum_template_family_fraction=1.0,
        )
    raise ValueError(f"Unknown admission policy profile: {profile}")


def _load_curriculum(path: Path) -> tuple[CurriculumRecord, ...]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = json.loads(line)
        try:
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
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid curriculum row {line_number}: {exc}") from exc
    return tuple(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate BCNLT corpus admission gates")
    parser.add_argument("--database", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--accepted-output")
    parser.add_argument("--holdout-hashes")
    parser.add_argument("--policy-profile", choices=("full", "smoke"), default="full")
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if report_path.exists():
        parser.error(f"report already exists: {report_path}")
    database = CorpusDatabase(args.database)
    corpus = CorpusValidationInput(
        documents=tuple(database.documents()),
        candidates=_load_curriculum(Path(args.curriculum)),
    )
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
    report = validate_corpus(corpus, tokenizer, policy)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = report.to_dict()
    report_payload["policy_profile"] = args.policy_profile
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if args.accepted_output:
        accepted_path = Path(args.accepted_output)
        if accepted_path.exists():
            parser.error(f"accepted output already exists: {accepted_path}")
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        accepted_path.write_text(
            "".join(
                json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for item in report.accepted_records
            ),
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps({"passed": report.passed, "report": str(report_path)}, sort_keys=True))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
