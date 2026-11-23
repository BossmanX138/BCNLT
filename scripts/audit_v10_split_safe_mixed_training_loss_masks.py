#!/usr/bin/env python3
"""Audit a train-only V10 semantic and reviewed-instruction curriculum."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.mixed_task_loss_mask_audit import (  # noqa: E402
    SplitSafeMixedTrainingLossMaskAuditor,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path)
    arguments = parser.parse_args()
    result = SplitSafeMixedTrainingLossMaskAuditor(
        NativeTokenizerBinding.load()
    ).materialize(
        arguments.database,
        arguments.output,
        audit_manifest=arguments.audit_manifest,
    )
    payload = asdict(result)
    payload["output_path"] = str(result.output_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
