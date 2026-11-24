"""Resume deterministic plus raw-AST auditing of the completed V10 admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.resumable_training_semantic_audit import (
    ResumableTrainingSemanticAudit,
)

DEFAULT_ADMISSION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-frozen-train-admission-v1"
    / "admission.sqlite"
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
    / "bcnlt-v10-training-source-freeze-v1"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "bcnlt-v10-frozen-train-semantic-audit-v1"
    / "audit.sqlite"
)
SOURCE_CATALOG_SHA256 = (
    "286d92f79f84b75049b845be0270e4308f32056421c8dea92d82d41b4e4003e2"
)
SOURCE_FREEZE_MANIFEST_SHA256 = (
    "b1bdb44b55528e01b7d0c551ec9f6bf09092b0723226c1d508002f81fc490b57"
)
QUALIFICATION_FILE_SHA256 = (
    "208fe15e93d439864183e8a936949d71b30063e44c104ee0066a2e1a254d107d"
)
ADMISSION_HOST_CODE_IDENTITY_SHA256 = (
    "a19a8072c0f88c2f7a2004347cbb829d5b09fe892a2555100de33eb2e016f661"
)
AUDIT_HOST_CODE_FILES = (
    "scripts/run_v10_training_semantic_audit.py",
    "bosscrafts_model/resumable_training_semantic_audit.py",
    "bosscrafts_model/deterministic_semantic_audit.py",
    "bosscrafts_model/local_semantic_review.py",
    "bosscrafts_model/heldout_semantic_review.py",
    "bosscrafts_model/bcnlt_source_pipeline.py",
    "bosscrafts_model/raw_ast_witness.py",
    "bosscrafts_model/raw_ast_witness_sidecar.py",
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_host_code_inventory() -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "bytes": (PROJECT_ROOT / relative).stat().st_size,
            "sha256": _sha256_file(PROJECT_ROOT / relative),
        }
        for relative in AUDIT_HOST_CODE_FILES
    ]


def _audit_host_code_identity(
    records: list[dict[str, object]] | None = None,
) -> str:
    if records is None:
        records = _audit_host_code_inventory()
    return hashlib.sha256(_canonical(records).encode("utf-8")).hexdigest()


def _freeze_code_manifest(
    path: Path, records: list[dict[str, object]], identity: str
) -> str:
    payload = {
        "schema_version": "bosscrafts-audit-host-code-manifest-v1",
        "audit_host_code_identity_sha256": identity,
        "files": records,
    }
    encoded = f"{_canonical(payload)}\n".encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("audit host code manifest changed across resume")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--status-only", action="store_true")
    arguments = parser.parse_args()
    audit_inventory = _audit_host_code_inventory()
    audit_identity = _audit_host_code_identity(audit_inventory)
    audit = ResumableTrainingSemanticAudit(
        arguments.admission,
        arguments.catalog,
        arguments.freeze,
        arguments.audit,
        expected_source_catalog_sha256=SOURCE_CATALOG_SHA256,
        expected_source_freeze_manifest_sha256=SOURCE_FREEZE_MANIFEST_SHA256,
        expected_qualification_sha256=QUALIFICATION_FILE_SHA256,
        expected_admission_host_code_identity_sha256=(
            ADMISSION_HOST_CODE_IDENTITY_SHA256
        ),
        audit_host_code_identity_sha256=audit_identity,
        source_split="train",
    )
    code_manifest = arguments.audit.resolve().with_name("host-code-manifest.json")
    code_manifest_sha256 = _freeze_code_manifest(
        code_manifest, audit_inventory, audit_identity
    )
    status = (
        audit.status()
        if arguments.status_only
        else audit.process(max_documents=arguments.max_documents)
    )
    print(
        json.dumps(
            {
                **asdict(status),
                "audit_host_code_identity_sha256": audit_identity,
                "audit_host_code_manifest": str(code_manifest),
                "audit_host_code_manifest_sha256": code_manifest_sha256,
                "database_path": str(status.database_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
