"""Materialize zero-finding eligibility for a repaired-runtime evaluation split."""

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
from scripts.run_v10_frozen_evaluation_semantic_audit_v3_runtime_repair import (  # noqa: E402
    audit_host_code_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("validation", "test", "reserved_heldout"), required=True
    )
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path)
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
    audit = (
        arguments.audit.resolve()
        if arguments.audit is not None
        else PROJECT_ROOT
        / "artifacts"
        / "audits"
        / f"bcnlt-v10-frozen-{arguments.split}-semantic-audit-v3-runtime-repair"
        / "audit.sqlite"
    )
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else PROJECT_ROOT
        / "artifacts"
        / "corpora"
        / f"bcnlt-v10-frozen-{arguments.split}-eligibility-v3-runtime-repair"
    )
    identity = audit_host_code_identity()
    result = AuditedTrainingEligibilityBuilder(
        expected_audit_host_code_identity_sha256=identity,
        expected_source_split=arguments.split,
    ).materialize(audit, admission, output)
    print(
        json.dumps(
            {
                **asdict(result),
                "output_root": str(result.output_root),
                "source_split": arguments.split,
                "audit_host_code_identity_sha256": identity,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
