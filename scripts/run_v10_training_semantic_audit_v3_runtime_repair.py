"""Audit the V3 repaired-runtime admission without touching V2 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.immutable_artifacts import publish_new_files  # noqa: E402
from bosscrafts_model.resumable_training_semantic_audit import (  # noqa: E402
    ResumableTrainingSemanticAudit,
)
from scripts.run_v10_frozen_training_admission_v3_runtime_repair import (  # noqa: E402
    DEFAULT_CATALOG,
    DEFAULT_FREEZE,
)
from scripts.run_v10_frozen_training_admission_v3_runtime_repair import (
    DEFAULT_DATABASE as DEFAULT_ADMISSION,
)
from scripts.run_v10_frozen_training_admission_v3_runtime_repair import (
    FREEZE_MANIFEST_SHA256 as SOURCE_FREEZE_MANIFEST_SHA256,
)
from scripts.run_v10_frozen_training_admission_v3_runtime_repair import (
    host_code_identity as admission_host_code_identity,
)

DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "bcnlt-v10-frozen-train-semantic-audit-v3-runtime-repair"
    / "audit.sqlite"
)
SOURCE_CATALOG_SHA256 = (
    "286d92f79f84b75049b845be0270e4308f32056421c8dea92d82d41b4e4003e2"
)
QUALIFICATION_FILE_SHA256 = (
    "94ff8291f7ffe109b306e7b987643505ed1c3ac08184c5e78f6c94c3933c7568"
)
AUDIT_HOST_CODE_FILES = (
    "scripts/run_v10_training_semantic_audit_v3_runtime_repair.py",
    "scripts/run_v10_frozen_training_admission_v3_runtime_repair.py",
    "bosscrafts_model/resumable_training_semantic_audit.py",
    "bosscrafts_model/deterministic_semantic_audit.py",
    "bosscrafts_model/local_semantic_review.py",
    "bosscrafts_model/heldout_semantic_review.py",
    "bosscrafts_model/jsonl.py",
    "bosscrafts_model/bcnlt_source_pipeline.py",
    "bosscrafts_model/runtime_identity.py",
    "bosscrafts_model/raw_ast_witness.py",
    "bosscrafts_model/raw_ast_witness_sidecar.py",
    "bosscrafts_model/immutable_artifacts.py",
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_host_code_inventory() -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "bytes": (PROJECT_ROOT / relative).stat().st_size,
            "sha256": _sha256_file(PROJECT_ROOT / relative),
        }
        for relative in AUDIT_HOST_CODE_FILES
    ]


def audit_host_code_identity(
    records: list[dict[str, object]] | None = None,
) -> str:
    inventory = records if records is not None else audit_host_code_inventory()
    return hashlib.sha256(_canonical(inventory).encode("utf-8")).hexdigest()


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
    try:
        publish_new_files(
            ((path, encoded),),
            artifact_label="audit host code manifest",
        )
    except ValueError:
        if not path.is_file() or path.read_bytes() != encoded:
            raise
    return digest


def _verify_host_code_inventory_unchanged(
    expected: list[dict[str, object]],
    actual: list[dict[str, object]],
) -> None:
    if actual != expected:
        raise RuntimeError("audit host code changed during invocation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--status-only", action="store_true")
    arguments = parser.parse_args()
    inventory = audit_host_code_inventory()
    identity = audit_host_code_identity(inventory)
    audit = ResumableTrainingSemanticAudit(
        arguments.admission,
        arguments.catalog,
        arguments.freeze,
        arguments.audit,
        expected_source_catalog_sha256=SOURCE_CATALOG_SHA256,
        expected_source_freeze_manifest_sha256=SOURCE_FREEZE_MANIFEST_SHA256,
        expected_qualification_sha256=QUALIFICATION_FILE_SHA256,
        expected_admission_host_code_identity_sha256=admission_host_code_identity(),
        audit_host_code_identity_sha256=identity,
        source_split="train",
    )
    code_manifest = arguments.audit.resolve().with_name("host-code-manifest.json")
    code_manifest_sha256 = _freeze_code_manifest(code_manifest, inventory, identity)
    status = (
        audit.status()
        if arguments.status_only
        else audit.process(max_documents=arguments.max_documents)
    )
    _verify_host_code_inventory_unchanged(inventory, audit_host_code_inventory())
    print(
        json.dumps(
            {
                **asdict(status),
                "audit_host_code_identity_sha256": identity,
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
