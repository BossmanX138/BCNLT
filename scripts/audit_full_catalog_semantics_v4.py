"""Exercise every catalog document through V4 parsing without token encoding."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.bcnlt_source_pipeline import (  # noqa: E402
    BCNLTPreprocessingError,
    BCNLTSourcePipeline,
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
    SIDECAR_PROTOCOL,
)
from bosscrafts_model.current_corpus import (  # noqa: E402
    SQLiteSourceCatalog,
    SourceCandidate,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402


DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "artifacts"
    / "source-catalogs"
    / "legacy-current-source-v1"
    / "corpus.sqlite"
)
DEFAULT_CATALOG_SHA256 = "286d92f79f84b75049b845be0270e4308f32056421c8dea92d82d41b4e4003e2"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "reviews"
    / "bcnlt-v4-full-parser-preflight-v1"
    / "report.json"
)


@dataclass(frozen=True)
class _Outcome:
    language: str
    document_id: str
    relative_path: str
    record_count: int
    unknown_count: int
    canonical_bytes: int
    failure: str | None


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _inspect(pipeline: BCNLTSourcePipeline, candidate: SourceCandidate) -> _Outcome:
    try:
        response = pipeline._invoke_sidecar(  # Intentional parser-only audit boundary.
            candidate.source,
            language=candidate.language,
            path=candidate.relative_path,
            dialect=candidate.dialect,
        )
        canonical = response.get("canonical_jsonl")
        expected_source = hashlib.sha256(candidate.source.encode("utf-8")).hexdigest()
        if type(canonical) is not str or (canonical and not canonical.endswith("\n")):
            raise BCNLTPreprocessingError("canonical JSONL framing mismatch")
        expected_canonical = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        expected_records = len(canonical.splitlines())
        if (
            response.get("protocol") != SIDECAR_PROTOCOL
            or response.get("language") != candidate.language
            or response.get("path") != candidate.relative_path
            or response.get("dialect") != candidate.dialect
            or response.get("semantic_schema_version") != "bcnlt-semantic-record-v3"
            or response.get("semantic_spec_sha256")
            != CURRENT_RUNTIME_SPEC_SHA256[candidate.language]
            or response.get("source_sha256") != expected_source
            or response.get("canonical_sha256") != expected_canonical
            or type(response.get("record_count")) is not int
            or response.get("record_count") != expected_records
        ):
            raise BCNLTPreprocessingError("canonical parser provenance mismatch")
        return _Outcome(
            language=candidate.language,
            document_id=candidate.document_id,
            relative_path=candidate.relative_path,
            record_count=expected_records,
            unknown_count=canonical.count('"event":"UNKNOWN_SEMANTIC"'),
            canonical_bytes=len(canonical.encode("utf-8")),
            failure=None,
        )
    except (BCNLTPreprocessingError, UnicodeEncodeError, ValueError) as error:
        return _Outcome(
            language=candidate.language,
            document_id=candidate.document_id,
            relative_path=candidate.relative_path,
            record_count=0,
            unknown_count=0,
            canonical_bytes=0,
            failure=" ".join(str(error).split())[:512],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--catalog-sha256", default=DEFAULT_CATALOG_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-documents", type=int)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if arguments.max_documents is not None and arguments.max_documents < 1:
        parser.error("max-documents must be positive")

    binding = NativeTokenizerBinding.load()
    pipeline = BCNLTSourcePipeline(binding, persistent_sidecars=True)
    catalog = SQLiteSourceCatalog(
        arguments.catalog,
        expected_sha256=arguments.catalog_sha256,
    )
    totals: dict[str, dict[str, int]] = {}
    failures: list[dict[str, str]] = []
    processed = 0
    exhausted = True
    iterator = iter(catalog.iter_candidates(limit_per_language=None))
    executor = ThreadPoolExecutor(max_workers=arguments.workers)
    try:
        stop = False
        while not stop:
            batch: list[tuple[SourceCandidate, Future[_Outcome]]] = []
            while len(batch) < arguments.workers:
                if (
                    arguments.max_documents is not None
                    and processed + len(batch) >= arguments.max_documents
                ):
                    exhausted = False
                    stop = True
                    break
                try:
                    candidate = next(iterator)
                except StopIteration:
                    stop = True
                    break
                batch.append((candidate, executor.submit(_inspect, pipeline, candidate)))
            for candidate, future in batch:
                outcome = future.result()
                language = totals.setdefault(
                    candidate.language,
                    {
                        "documents": 0,
                        "failures": 0,
                        "semantic_records": 0,
                        "unknown_records": 0,
                        "canonical_bytes": 0,
                    },
                )
                language["documents"] += 1
                language["semantic_records"] += outcome.record_count
                language["unknown_records"] += outcome.unknown_count
                language["canonical_bytes"] += outcome.canonical_bytes
                if outcome.failure is not None:
                    language["failures"] += 1
                    failures.append(
                        {
                            "document_id": outcome.document_id,
                            "language": outcome.language,
                            "relative_path": outcome.relative_path,
                            "detail": outcome.failure,
                        }
                    )
                processed += 1
                if processed % 1000 == 0:
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "failures": len(failures),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if not batch:
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        pipeline.close()

    unsigned = {
        "schema_version": "bosscrafts-full-catalog-parser-preflight-v1",
        "catalog_sha256": arguments.catalog_sha256,
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "persistent_sidecars": True,
        "workers": arguments.workers,
        "processed_documents": processed,
        "catalog_exhausted": exhausted,
        "failure_count": len(failures),
        "per_language": dict(sorted(totals.items())),
        "failures": failures,
        "passed": exhausted and not failures,
    }
    manifest_sha = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    report = {**unsigned, "manifest_sha256": manifest_sha}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(
        f"{_canonical_json(report)}\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(arguments.output)
    print(
        json.dumps(
            {
                "catalog_exhausted": exhausted,
                "failure_count": len(failures),
                "manifest_sha256": manifest_sha,
                "output": str(arguments.output.resolve()),
                "passed": report["passed"],
                "processed_documents": processed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
