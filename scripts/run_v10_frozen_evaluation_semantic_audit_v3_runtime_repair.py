"""Audit one repaired-runtime frozen V10 evaluation split."""

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

from bosscrafts_model.resumable_training_semantic_audit import (  # noqa: E402
    ResumableTrainingSemanticAudit,
)
from scripts.run_v10_frozen_evaluation_admission_v3_runtime_repair import (  # noqa: E402
    host_code_identity as evaluation_admission_host_code_identity,
)
from scripts.run_v10_frozen_training_admission_v3_runtime_repair import (  # noqa: E402
    DEFAULT_CATALOG,
    DEFAULT_FREEZE,
    FREEZE_MANIFEST_SHA256,
)
from scripts.run_v10_training_semantic_audit_v3_runtime_repair import (  # noqa: E402
    QUALIFICATION_FILE_SHA256,
    SOURCE_CATALOG_SHA256,
    _canonical,
    _freeze_code_manifest,
    _sha256_file,
)

AUDIT_HOST_CODE_FILES = (
    "scripts/run_v10_frozen_evaluation_semantic_audit_v3_runtime_repair.py",
    "scripts/run_v10_training_semantic_audit_v3_runtime_repair.py",
    "scripts/run_v10_frozen_evaluation_admission_v3_runtime_repair.py",
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
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("validation", "test", "reserved_heldout"), required=True
    )
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--status-only", action="store_true")
    arguments = parser.parse_args()
    admission = (
        arguments.admission.resolve()
        if arguments.admission is not None
        else PROJECT_ROOT
        / "artifacts"
        / "corpora"
        / f"bcnlt-v10-frozen-{arguments.split}-admission-v3-runtime-repair"
        / "admission.sqlite"
    )
    audit_path = (
        arguments.audit.resolve()
        if arguments.audit is not None
        else PROJECT_ROOT
        / "artifacts"
        / "audits"
        / f"bcnlt-v10-frozen-{arguments.split}-semantic-audit-v3-runtime-repair"
        / "audit.sqlite"
    )
    inventory = audit_host_code_inventory()
    identity = audit_host_code_identity(inventory)
    audit = ResumableTrainingSemanticAudit(
        admission,
        arguments.catalog,
        arguments.freeze,
        audit_path,
        expected_source_catalog_sha256=SOURCE_CATALOG_SHA256,
        expected_source_freeze_manifest_sha256=FREEZE_MANIFEST_SHA256,
        expected_qualification_sha256=QUALIFICATION_FILE_SHA256,
        expected_admission_host_code_identity_sha256=(
            evaluation_admission_host_code_identity()
        ),
        audit_host_code_identity_sha256=identity,
        source_split=arguments.split,
    )
    manifest_path = audit_path.with_name("host-code-manifest.json")
    manifest_sha256 = _freeze_code_manifest(manifest_path, inventory, identity)
    status = (
        audit.status()
        if arguments.status_only
        else audit.process(max_documents=arguments.max_documents)
    )
    print(
        json.dumps(
            {
                **asdict(status),
                "database_path": str(status.database_path),
                "source_split": arguments.split,
                "audit_host_code_identity_sha256": identity,
                "audit_host_code_manifest": str(manifest_path),
                "audit_host_code_manifest_sha256": manifest_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
