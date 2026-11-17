"""Run the fixed Docker containment probes and publish their receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.containment_acceptance import (
    build_containment_receipt,
    publish_containment_receipt,
    run_containment_probes,
)
from bosscrafts_model.model_category_evidence import checkpoint_sha256
from bosscrafts_model.scientific_sandbox import DockerScientificSandbox


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sandbox-manifest",
        type=Path,
        default=PROJECT_ROOT / "sandbox" / "images.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    sandbox = DockerScientificSandbox(arguments.sandbox_manifest)
    records = run_containment_probes(sandbox)
    receipt = build_containment_receipt(
        records,
        checkpoint_sha256=checkpoint_sha256(arguments.checkpoint),
        checkpoint_step=arguments.checkpoint_step,
    )
    publish_containment_receipt(arguments.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
