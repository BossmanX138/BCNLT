"""Resume full current-lineage task packing into a committed SQLite artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.resumable_task_packing import ResumableCurrentTaskPacking  # noqa: E402
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402

DEFAULT_ADMISSION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-full-resumable-v1"
    / "admission.sqlite"
)
DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "artifacts"
    / "source-catalogs"
    / "legacy-current-source-v1"
    / "corpus.sqlite"
)
DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-full-resumable-compact-v2-tasks-v1"
    / "tasks.sqlite"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--max-sequence", type=int, default=2048)
    parser.add_argument(
        "--semantic-wire",
        choices=("compact_v2", "full_v3"),
        default="compact_v2",
    )
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--status-only", action="store_true")
    arguments = parser.parse_args()
    binding = NativeTokenizerBinding.load()
    packer = ResumableCurrentTaskPacking(
        binding,
        arguments.admission,
        arguments.catalog,
        arguments.database,
        max_sequence=arguments.max_sequence,
        semantic_wire=arguments.semantic_wire,
    )
    status = packer.status() if arguments.status_only else packer.process(
        max_documents=arguments.max_documents
    )
    print(json.dumps(status.__dict__, default=str, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
