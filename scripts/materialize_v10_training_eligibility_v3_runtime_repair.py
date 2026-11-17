"""Materialize zero-finding V10 eligibility for the repaired-runtime lineage."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.training_eligibility import (  # noqa: E402
    AuditedTrainingEligibilityBuilder,
)
from scripts.run_v10_frozen_training_admission_v3_runtime_repair import (  # noqa: E402
    DEFAULT_DATABASE as DEFAULT_ADMISSION,
)
from scripts.run_v10_training_semantic_audit_v3_runtime_repair import (  # noqa: E402
    AUDIT_HOST_CODE_FILES,
    DEFAULT_AUDIT,
    audit_host_code_identity,
)

__all__ = ("AUDIT_HOST_CODE_FILES",)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-audited-training-eligibility-v3-runtime-repair"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    identity = audit_host_code_identity()
    result = AuditedTrainingEligibilityBuilder(
        expected_audit_host_code_identity_sha256=identity,
        expected_source_split="train",
    ).materialize(arguments.audit, arguments.admission, arguments.output)
    print(
        json.dumps(
            {
                **asdict(result),
                "output_root": str(result.output_root),
                "audit_host_code_identity_sha256": identity,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
