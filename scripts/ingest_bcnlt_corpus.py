from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from bcnlt.corpus import (
    CorpusDatabase,
    SourceSpec,
    ingest_local_source,
    ingest_opencodeinstruct_stream,
    ingest_stack_stream,
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Corpus source registry must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest provenance-gated BCNLT sources")
    parser.add_argument("--config", default="configs/bcnlt_corpus_sources_v1.json")
    parser.add_argument("--database", required=True)
    parser.add_argument("--source", action="append", help="source name; repeatable")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--holdout-hashes")
    args = parser.parse_args(argv)

    registry = _load_json(Path(args.config))
    if registry.get("schema_version") != "bcnlt-corpus-sources-v1":
        parser.error("unsupported corpus source registry schema")
    selected = set(args.source or [])
    holdouts = set()
    if args.holdout_hashes:
        holdouts = {
            line.strip()
            for line in Path(args.holdout_hashes).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    database = CorpusDatabase(args.database)
    totals: dict[str, int] = {}
    for raw in registry.get("sources", []):
        if (
            not isinstance(raw, dict)
            or raw.get("enabled") is False
            or (selected and raw.get("name") not in selected)
        ):
            continue
        spec = SourceSpec(
            provider=str(raw["provider"]),
            name=str(raw["name"]),
            revision=str(raw["revision"]),
            upstream_url=str(raw["upstream_url"]),
        )
        if spec.provider == "local":
            admitted = ingest_local_source(
                database,
                spec,
                str(raw["root"]),
                repository_identity=spec.name,
                ownership=str(raw.get("ownership", "owner_provided")),
                license_spdx="owner_provided",
                holdout_hashes=holdouts,
            )
        else:
            from datasets import load_dataset

            kwargs = {
                "path": spec.name,
                "split": "train",
                "streaming": True,
                "revision": spec.revision,
            }
            if raw.get("data_dir"):
                kwargs["data_dir"] = raw["data_dir"]
            rows = load_dataset(**kwargs)
            if args.limit is not None:
                rows = itertools.islice(rows, args.limit)
            admitted = (
                ingest_stack_stream(database, spec, rows, holdout_hashes=holdouts)
                if spec.name == "bigcode/the-stack-dedup"
                else ingest_opencodeinstruct_stream(
                    database, spec, rows, holdout_hashes=holdouts
                )
            )
        totals[spec.name] = len(admitted)
    print(
        json.dumps(
            {
                "admitted_by_source": totals,
                "document_count": database.document_count(),
                "rejection_count": len(database.rejections()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
