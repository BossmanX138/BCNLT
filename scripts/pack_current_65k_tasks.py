"""Pack admitted current-spec documents into bounded, loss-masked task examples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.current_task_packing import CurrentCorpusTaskPacker
from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_ADMISSION = (
    PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-current-spec-65536-pilot-v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-current-spec-65536-pilot-tasks-v1"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequence", type=int, default=2048)
    arguments = parser.parse_args()
    binding = NativeTokenizerBinding.load()
    result = CurrentCorpusTaskPacker(
        binding,
        max_sequence=arguments.max_sequence,
    ).materialize(arguments.admission, arguments.output)
    print(
        json.dumps(
            {
                "example_count": result.example_count,
                "examples_sha256": result.examples_sha256,
                "family_counts": result.family_counts,
                "manifest_sha256": result.manifest_sha256,
                "output_root": str(result.output_root),
                "packing_ready": result.packing_ready,
                "rejected_count": result.rejected_count,
                "split_counts": result.split_counts,
                "training_ready": result.training_ready,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
