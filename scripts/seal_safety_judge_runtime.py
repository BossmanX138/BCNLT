"""Seal or verify the exact pinned local NVIDIA safety-judge snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.safety_judge_runtime import (
    SafetyJudgeRuntime,
    seal_safety_judge_runtime,
    verify_safety_judge_runtime,
)


def _payload(runtime: SafetyJudgeRuntime) -> dict[str, object]:
    return {
        "root": str(runtime.root),
        "repository": runtime.repository,
        "revision": runtime.revision,
        "manifest_sha256": runtime.manifest_sha256,
        "total_bytes": runtime.total_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("seal", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--runtime-root", type=Path, required=True)
    arguments = parser.parse_args()
    runtime = (
        seal_safety_judge_runtime(arguments.runtime_root)
        if arguments.command == "seal"
        else verify_safety_judge_runtime(arguments.runtime_root)
    )
    print(json.dumps(_payload(runtime), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
