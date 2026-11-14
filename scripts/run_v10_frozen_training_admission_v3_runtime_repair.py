"""Admit the frozen train split with the repaired V10 semantic runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline  # noqa: E402
from bosscrafts_model.resumable_current_corpus import (  # noqa: E402
    ResumableCurrentCorpusAdmission,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402
from bosscrafts_model.training_corpus_freeze import (  # noqa: E402
    VerifiedTrainingSourceFreeze,
)


DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "artifacts"
    / "source-catalogs"
    / "legacy-current-source-v1"
    / "corpus.sqlite"
)
DEFAULT_FREEZE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-training-source-freeze-v2-code-audit"
)
DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-frozen-train-admission-v3-runtime-repair"
    / "admission.sqlite"
)
FREEZE_MANIFEST_SHA256 = (
    "65bd0014697fce29b64e9a023225914b54257ce9c684192c42dd4572d1a4378c"
)
NATURAL_DETERMINISTIC_AUDIT_MANIFEST_SHA256 = (
    "57973e9c8ffb831df91348042a92add1389f1868ec4ae2eef16f1288fa1d74a8"
)
HOST_CODE_FILES = (
    "scripts/run_v10_frozen_training_admission_v3_runtime_repair.py",
    "bosscrafts_model/training_corpus_freeze.py",
    "bosscrafts_model/resumable_current_corpus.py",
    "bosscrafts_model/current_corpus.py",
    "bosscrafts_model/bcnlt_source_pipeline.py",
    "bosscrafts_model/runtime_identity.py",
    "bosscrafts_model/tokenizer.py",
    "bosscrafts_model/bcnlt_adapter.py",
    "bosscrafts_model/semantic.py",
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_code_inventory() -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "bytes": (PROJECT_ROOT / relative).stat().st_size,
            "sha256": _sha256_file(PROJECT_ROOT / relative),
        }
        for relative in HOST_CODE_FILES
    ]


def host_code_identity() -> str:
    return hashlib.sha256(
        _canonical(host_code_inventory()).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--worker-mode", choices=("threads", "processes"), default="processes"
    )
    parser.add_argument(
        "--persistent-sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--status-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.worker_mode == "processes" and not arguments.persistent_sidecars:
        parser.error("process worker mode requires persistent sidecars")

    freeze = VerifiedTrainingSourceFreeze(
        arguments.freeze,
        arguments.catalog,
        expected_manifest_sha256=FREEZE_MANIFEST_SHA256,
    )
    binding = NativeTokenizerBinding.load()
    pipeline = BCNLTSourcePipeline(
        binding,
        persistent_sidecars=arguments.persistent_sidecars,
        timeout_seconds=arguments.timeout_seconds,
    )
    identity = host_code_identity()
    try:
        admission = ResumableCurrentCorpusAdmission(
            pipeline,
            arguments.database,
            source_catalog_sha256=freeze.source_catalog_sha256,
            review_manifest_sha256=NATURAL_DETERMINISTIC_AUDIT_MANIFEST_SHA256,
            host_code_identity_sha256=identity,
            source_freeze_manifest_sha256=freeze.manifest_sha256,
            semantic_audit_qualification_sha256=freeze.qualification_file_sha256,
            expected_source_document_count=freeze.training_document_count,
        )
        status = (
            admission.status()
            if arguments.status_only
            else admission.process(
                freeze.iter_training_candidates(),
                max_documents=arguments.max_documents,
                workers=arguments.workers,
                worker_mode=arguments.worker_mode,
            )
        )
    finally:
        pipeline.close()
    if status.catalog_exhausted and status.processed_total != freeze.training_document_count:
        raise RuntimeError(
            "frozen admission exhausted before every training document was committed"
        )
    print(
        json.dumps(
            {
                **status.__dict__,
                "database_path": str(status.database_path),
                "expected_source_document_count": freeze.training_document_count,
                "host_code_identity_sha256": identity,
                "parser_runtime_identity_sha256": pipeline.parser_runtime_identity_sha256,
                "source_freeze_manifest_sha256": freeze.manifest_sha256,
                "training_ready": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
