"""Resume post-code-audit semantic auditing of one frozen evaluation split."""
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

from bosscrafts_model.resumable_training_semantic_audit import (
    ResumableTrainingSemanticAudit,
)
from scripts.run_v10_frozen_evaluation_admission_v2_code_audit import (
    FREEZE_MANIFEST_SHA256,
)
from scripts.run_v10_frozen_evaluation_admission_v2_code_audit import (
    _host_code_identity as evaluation_admission_host_code_identity,
)
from scripts.run_v10_training_semantic_audit_v2_code_audit import (
    _freeze_code_manifest,
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
SOURCE_CATALOG_SHA256 = (
    "286d92f79f84b75049b845be0270e4308f32056421c8dea92d82d41b4e4003e2"
)
QUALIFICATION_FILE_SHA256 = (
    "94ff8291f7ffe109b306e7b987643505ed1c3ac08184c5e78f6c94c3933c7568"
)
AUDIT_HOST_CODE_FILES = (
    "scripts/run_v10_frozen_evaluation_semantic_audit_v2_code_audit.py",
    "scripts/run_v10_training_semantic_audit_v2_code_audit.py",
    "bosscrafts_model/resumable_training_semantic_audit.py",
    "bosscrafts_model/deterministic_semantic_audit.py",
    "bosscrafts_model/local_semantic_review.py",
    "bosscrafts_model/heldout_semantic_review.py",
    "bosscrafts_model/jsonl.py",
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


def _audit_inventory() -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "bytes": (PROJECT_ROOT / relative).stat().st_size,
            "sha256": _sha256_file(PROJECT_ROOT / relative),
        }
        for relative in AUDIT_HOST_CODE_FILES
    ]


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
        else (
            PROJECT_ROOT
            / "artifacts"
            / "corpora"
            / f"bcnlt-v10-frozen-{arguments.split}-admission-v2-code-audit"
            / "admission.sqlite"
        )
    )
    audit_path = (
        arguments.audit.resolve()
        if arguments.audit is not None
        else (
            PROJECT_ROOT
            / "artifacts"
            / "audits"
            / f"bcnlt-v10-frozen-{arguments.split}-semantic-audit-v2-code-audit"
            / "audit.sqlite"
        )
    )
    inventory = _audit_inventory()
    audit_identity = hashlib.sha256(_canonical(inventory).encode()).hexdigest()
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
        audit_host_code_identity_sha256=audit_identity,
        source_split=arguments.split,
    )
    manifest_path = audit_path.with_name("host-code-manifest.json")
    manifest_sha256 = _freeze_code_manifest(
        manifest_path, inventory, audit_identity
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
                "database_path": str(status.database_path),
                "source_split": arguments.split,
                "audit_host_code_identity_sha256": audit_identity,
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
