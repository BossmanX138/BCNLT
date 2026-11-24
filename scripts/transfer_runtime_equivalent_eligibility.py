"""Transfer a frozen eligibility partition across exact-equivalent runtimes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.eligibility_runtime_transfer import (  # noqa: E402
    transfer_runtime_equivalent_eligibility,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-admission-database", type=Path, required=True)
    parser.add_argument("--candidate-admission-database", type=Path, required=True)
    parser.add_argument("--source-eligibility-root", type=Path, required=True)
    parser.add_argument("--equivalence-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = transfer_runtime_equivalent_eligibility(
        baseline_admission_database=arguments.baseline_admission_database,
        candidate_admission_database=arguments.candidate_admission_database,
        source_eligibility_root=arguments.source_eligibility_root,
        equivalence_receipt=arguments.equivalence_receipt,
        output_root=arguments.output_root,
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "eligible_document_count": result.eligible_document_count,
                "excluded_document_count": result.excluded_document_count,
                "manifest_sha256": result.manifest_sha256,
                "candidate_admission_identity_sha256": result.candidate_admission_identity_sha256,
                "equivalence_manifest_sha256": result.equivalence_manifest_sha256,
                "training_ready": result.training_ready,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
