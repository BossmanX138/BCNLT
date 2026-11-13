"""Resume task packing for repaired-runtime zero-finding V10 training data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.resumable_task_packing import (  # noqa: E402
    ResumableCurrentTaskPacking,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402
from scripts.materialize_v10_training_eligibility_v3_runtime_repair import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_ELIGIBILITY,
)
from scripts.run_v10_frozen_training_admission_v3_runtime_repair import (  # noqa: E402
    DEFAULT_CATALOG,
)
from scripts.run_v10_frozen_training_admission_v3_runtime_repair import (
    DEFAULT_DATABASE as DEFAULT_ADMISSION,
)

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-audited-training-tasks-v3-runtime-repair"
    / "tasks.sqlite"
)


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eligibility-manifest-sha256", required=True, type=_lower_sha256
    )
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-sequence", type=int, default=2048)
    parser.add_argument(
        "--semantic-wire", choices=("compact_v2", "full_v3"), default="compact_v2"
    )
    parser.add_argument("--status-only", action="store_true")
    arguments = parser.parse_args()

    binding = NativeTokenizerBinding.load()
    packing = ResumableCurrentTaskPacking(
        binding,
        arguments.admission,
        arguments.catalog,
        arguments.database,
        max_sequence=arguments.max_sequence,
        semantic_wire=arguments.semantic_wire,
        eligibility_root=arguments.eligibility,
        expected_eligibility_manifest_sha256=arguments.eligibility_manifest_sha256,
    )
    status = (
        packing.status()
        if arguments.status_only
        else packing.process(max_documents=arguments.max_documents)
    )
    payload: dict[str, object] = {
        **asdict(status),
        "database_path": str(status.database_path),
        "eligibility_manifest_sha256": arguments.eligibility_manifest_sha256,
        "semantic_wire": arguments.semantic_wire,
        "training_ready": False,
    }
    if status.packing_complete:
        receipt = audit_task_database(binding, status.database_path)
        payload["task_database_audit"] = {
            **asdict(receipt),
            "database_path": str(receipt.database_path),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
