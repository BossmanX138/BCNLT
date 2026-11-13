"""Resume exact-split task packing for a frozen V10 evaluation admission."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.resumable_task_packing import (
    ResumableCurrentTaskPacking,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "artifacts"
    / "source-catalogs"
    / "legacy-current-source-v1"
    / "corpus.sqlite"
)


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
        "--eligibility-manifest-sha256", required=True
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
        else (
            PROJECT_ROOT
            / "artifacts"
            / "corpora"
            / f"bcnlt-v10-frozen-{arguments.split}-admission-v1"
            / "admission.sqlite"
        )
    )
    database = (
        arguments.database.resolve()
        if arguments.database is not None
        else (
            PROJECT_ROOT
            / "artifacts"
            / "corpora"
            / f"bcnlt-v10-frozen-{arguments.split}-tasks-v1"
            / "tasks.sqlite"
        )
    )
    eligibility = (
        arguments.eligibility.resolve()
        if arguments.eligibility is not None
        else (
            PROJECT_ROOT
            / "artifacts"
            / "corpora"
            / f"bcnlt-v10-frozen-{arguments.split}-eligibility-v1"
        )
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
        expected_eligibility_manifest_sha256=(
            arguments.eligibility_manifest_sha256
        ),
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
        result = audit_task_database(binding, status.database_path)
        payload["task_database_audit"] = {
            **asdict(result),
            "database_path": str(result.database_path),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
