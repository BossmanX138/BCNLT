"""Promote repaired-runtime audited uncertainty data under the frozen V10 policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = (
    PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate" / "source"
)
for path in (PROJECT_ROOT, RUNTIME_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bosscrafts_model.semantic_calibration_promotion import (  # noqa: E402
    promote_semantic_uncertainty,
)
from scripts.build_v10_audited_semantic_calibration_v3_runtime_repair import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_DATASET,
)
from scripts.promote_semantic_uncertainty import frozen_policy  # noqa: E402

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "bcnlt-v10-audited-semantic-uncertainty-promotion-v3-runtime-repair"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print-policy", action="store_true")
    arguments = parser.parse_args()
    policy = frozen_policy()
    if arguments.print_policy:
        print(json.dumps(policy.to_dict(), indent=2, sort_keys=True))
        return 0
    result = promote_semantic_uncertainty(
        arguments.dataset,
        arguments.output,
        policy=policy,
    )
    print(
        json.dumps(
            {
                "calibration_artifact_sha256": result.calibration_artifact_sha256,
                "failure_codes": result.failure_codes,
                "manifest_sha256": result.manifest_sha256,
                "output_root": str(result.output_root),
                "promoted": result.promoted,
                "strict_receipt_sha256": result.strict_receipt_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
