"""Resume exact-split task packing for repaired-runtime V10 evaluation data."""

from __future__ import annotations

import argparse
import json
import sqlite3
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
from scripts.run_v10_frozen_training_admission_v3_runtime_repair import (  # noqa: E402
    DEFAULT_CATALOG,
)


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256")
    return value


def _verify_admission_split(path: Path, split: str) -> None:
    if not path.is_file():
        raise ValueError("evaluation admission database does not exist")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    finally:
        connection.close()
    identity = metadata.get("frozen_source_split_identity_sha256")
    if (
        metadata.get("catalog_exhausted") != "true"
        or metadata.get("frozen_source_split") != split
        or type(identity) is not str
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError("evaluation admission split lineage is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("validation", "test", "reserved_heldout"), required=True
    )
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--eligibility", type=Path)
    parser.add_argument(
        "--eligibility-manifest-sha256", required=True, type=_lower_sha256
    )
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-sequence", type=int, default=2048)
    parser.add_argument(
        "--semantic-wire", choices=("compact_v2", "full_v3"), default="compact_v2"
    )
    parser.add_argument("--status-only", action="store_true")
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
    database = (
        arguments.database.resolve()
        if arguments.database is not None
        else PROJECT_ROOT
        / "artifacts"
        / "corpora"
        / f"bcnlt-v10-frozen-{arguments.split}-tasks-v3-runtime-repair"
        / "tasks.sqlite"
    )
    eligibility = (
        arguments.eligibility.resolve()
        if arguments.eligibility is not None
        else PROJECT_ROOT
        / "artifacts"
        / "corpora"
        / f"bcnlt-v10-frozen-{arguments.split}-eligibility-v3-runtime-repair"
    )
    _verify_admission_split(admission, arguments.split)
    task_split = "validation" if arguments.split == "validation" else "test"
    binding = NativeTokenizerBinding.load()
    packing = ResumableCurrentTaskPacking(
        binding,
        admission,
        arguments.catalog,
        database,
        max_sequence=arguments.max_sequence,
        semantic_wire=arguments.semantic_wire,
        eligibility_root=eligibility,
        expected_eligibility_manifest_sha256=arguments.eligibility_manifest_sha256,
        forced_split=task_split,
    )
    status = (
        packing.status()
        if arguments.status_only
        else packing.process(max_documents=arguments.max_documents)
    )
    payload: dict[str, object] = {
        **asdict(status),
        "database_path": str(status.database_path),
        "frozen_source_split": arguments.split,
        "task_split": task_split,
        "semantic_wire": arguments.semantic_wire,
        "eligibility_manifest_sha256": arguments.eligibility_manifest_sha256,
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
