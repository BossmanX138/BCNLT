from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

from bcnlt.adaptive_vocab.config import AdaptiveVocabularyConfig
from bcnlt.adaptive_vocab.corpus_audit import audit_corpus
from bcnlt.adaptive_vocab.lineage import LINEAGE_SCHEMA_VERSION, LineageArtifact
from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.corpus.database import CorpusDatabase
from bcnlt.hf_tokenizer import MANIFEST_FILENAME
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


def _verify_lineage(
    lineage_path: Path,
    *,
    config_path: Path,
    source_config_path: Path,
    corpus_manifest_path: Path,
    tokenizer_artifact: Path,
) -> str:
    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != LINEAGE_SCHEMA_VERSION:
        raise ValueError("lineage_mismatch: unsupported lineage schema")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError("lineage_mismatch: artifacts are missing")
    expected = {
        str(item.get("name")): str(item.get("sha256"))
        for item in raw_artifacts
        if isinstance(item, dict)
    }
    checks = (
        ("config", config_path),
        ("source_config", source_config_path),
        ("corpus_manifest", corpus_manifest_path),
        ("v1_manifest", tokenizer_artifact / MANIFEST_FILENAME),
    )
    for name, path in checks:
        if name not in expected:
            raise ValueError(f"lineage_mismatch: {name} is missing")
        actual = LineageArtifact.capture(name, path).sha256
        if actual != expected[name]:
            raise ValueError(f"lineage_mismatch: {name} hash changed")
    identity = payload.get("identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("lineage_mismatch: identity is invalid")
    identity_body = {
        "schema_version": payload["schema_version"],
        "code_revision": payload.get("code_revision"),
        "artifacts": [
            {
                "name": item.get("name"),
                "kind": item.get("kind"),
                "sha256": item.get("sha256"),
            }
            for item in raw_artifacts
        ],
    }
    actual_identity = hashlib.sha256(
        json.dumps(
            identity_body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if actual_identity != identity:
        raise ValueError("lineage_mismatch: identity hash changed")
    return identity


def _write_new_file(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Audit BCNLT adaptive-vocabulary corpus diversity")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--accepted-records", required=True, type=Path)
    parser.add_argument("--tokenizer-artifact", required=True, type=Path)
    parser.add_argument("--base-tokenizer", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        database = CorpusDatabase(args.database)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        documents = tuple(database.documents())
    finally:
        database.close()

    try:
        config = AdaptiveVocabularyConfig.load(args.config)
        lineage_identity = _verify_lineage(
            args.lineage,
            config_path=args.config,
            source_config_path=args.source_config,
            corpus_manifest_path=args.corpus_manifest,
            tokenizer_artifact=args.tokenizer_artifact,
        )
        records = _load_curriculum(args.accepted_records)
        tokenizer = BCNLTTokenizer.from_pretrained(
            args.tokenizer_artifact, args.base_tokenizer
        )
        report = audit_corpus(documents, records, tokenizer, config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = report.to_dict()
    payload["lineage_identity_sha256"] = lineage_identity
    payload["report_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    _write_new_file(args.output, payload)
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
