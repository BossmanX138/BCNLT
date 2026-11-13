"""Materialize the zero-finding V10 training inventory from a completed audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.training_eligibility import AuditedTrainingEligibilityBuilder


DEFAULT_ADMISSION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-frozen-train-admission-v1"
    / "admission.sqlite"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "bcnlt-v10-frozen-train-semantic-audit-v1"
    / "audit.sqlite"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-audited-training-eligibility-v1"
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


def audit_host_code_identity() -> str:
    records = [
        {
            "path": relative,
            "bytes": (PROJECT_ROOT / relative).stat().st_size,
            "sha256": _sha256_file(PROJECT_ROOT / relative),
        }
        for relative in AUDIT_HOST_CODE_FILES
    ]
    return hashlib.sha256(_canonical(records).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    audit_identity = audit_host_code_identity()
    result = AuditedTrainingEligibilityBuilder(
        expected_audit_host_code_identity_sha256=audit_identity
    ).materialize(arguments.audit, arguments.admission, arguments.output)
    print(
        json.dumps(
            {
                **asdict(result),
                "output_root": str(result.output_root),
                "audit_host_code_identity_sha256": audit_identity,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
