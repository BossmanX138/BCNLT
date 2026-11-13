"""Resume full source-catalog admission into a committed SQLite artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from bosscrafts_model.current_corpus import SQLiteSourceCatalog
from bosscrafts_model.resumable_current_corpus import ResumableCurrentCorpusAdmission
from bosscrafts_model.tokenizer import NativeTokenizerBinding


DEFAULT_CATALOG = PROJECT_ROOT / "artifacts" / "source-catalogs" / "legacy-current-source-v1" / "corpus.sqlite"
DEFAULT_CATALOG_SHA256 = "286d92f79f84b75049b845be0270e4308f32056421c8dea92d82d41b4e4003e2"
DEFAULT_REVIEW = PROJECT_ROOT / "artifacts" / "reviews" / "bcnlt-v6-65536-heldout-all-language-adjudicated-v2" / "manifest.json"
DEFAULT_DATABASE = PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-v7-65536-full-resumable-v1" / "admission.sqlite"
HOST_CODE_FILES = (
    "scripts/run_full_resumable_admission.py",
    "bosscrafts_model/resumable_current_corpus.py",
    "bosscrafts_model/current_corpus.py",
    "bosscrafts_model/bcnlt_source_pipeline.py",
    "bosscrafts_model/tokenizer.py",
    "bosscrafts_model/bcnlt_adapter.py",
    "bosscrafts_model/semantic.py",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_code_identity() -> str:
    records = []
    for relative in HOST_CODE_FILES:
        path = PROJECT_ROOT / relative
        records.append(
            {"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size}
        )
    return hashlib.sha256(_canonical_json(records).encode("utf-8")).hexdigest()


def _load_passing_review(path: Path, binding: NativeTokenizerBinding) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256", None)
    actual = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    if claimed != actual:
        raise ValueError("review manifest SHA-256 mismatch")
    if (
        manifest.get("schema_version")
        != "bosscrafts-heldout-semantic-adjudication-manifest-v2"
    ):
        raise ValueError("review adjudication manifest schema mismatch")
    if not manifest.get("review_complete") or not manifest.get("semantic_review_passed"):
        raise ValueError("held-out semantic review has not passed")
    if (
        manifest.get("binding_sha256") != binding.binding_sha256
        or manifest.get("tokenizer_sha256") != binding.tokenizer_sha256
        or manifest.get("parser_runtime_identity_sha256")
        != PARSER_RUNTIME_IDENTITY_SHA256
    ):
        raise ValueError("review tokenizer or parser-runtime lineage mismatch")
    return str(claimed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--worker-mode",
        choices=("threads", "processes"),
        default="processes",
        help="use true process parallelism for CPU-bound semantic encoding",
    )
    parser.add_argument(
        "--persistent-sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse one bounded parser process per admission worker",
    )
    parser.add_argument("--status-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.worker_mode == "processes" and not arguments.persistent_sidecars:
        parser.error("process worker mode requires persistent sidecars")
    catalog_sha256 = (
        DEFAULT_CATALOG_SHA256
        if arguments.catalog.resolve() == DEFAULT_CATALOG.resolve()
        else _sha256(arguments.catalog)
    )
    binding = NativeTokenizerBinding.load()
    review_sha256 = _load_passing_review(arguments.review, binding)
    pipeline = BCNLTSourcePipeline(
        binding,
        persistent_sidecars=arguments.persistent_sidecars,
        timeout_seconds=arguments.timeout_seconds,
    )
    try:
        admission = ResumableCurrentCorpusAdmission(
            pipeline,
            arguments.database,
            source_catalog_sha256=catalog_sha256,
            review_manifest_sha256=review_sha256,
            host_code_identity_sha256=_host_code_identity(),
        )
        if arguments.status_only:
            status = admission.status()
        else:
            catalog = SQLiteSourceCatalog(arguments.catalog, expected_sha256=catalog_sha256)
            status = admission.process(
                catalog.iter_candidates(limit_per_language=None),
                max_documents=arguments.max_documents,
                workers=arguments.workers,
                worker_mode=arguments.worker_mode,
            )
    finally:
        pipeline.close()
    print(json.dumps(status.__dict__, default=str, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
