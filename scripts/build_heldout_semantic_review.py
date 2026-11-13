"""Build a deterministic source-aligned held-out semantic review queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.heldout_semantic_review import HeldoutSemanticReviewQueue
from bosscrafts_model.tokenizer import NativeTokenizerBinding


DEFAULT_ADMISSION = (
    PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-v6-65536-heldout-admission-v2"
)
DEFAULT_TASKS = (
    PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-v6-65536-heldout-tasks-v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "reviews" / "bcnlt-v6-65536-heldout-review-v2"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-language", type=int, default=2)
    parser.add_argument(
        "--admission-only",
        action="store_true",
        help="Select review items directly from a held-out admission artifact.",
    )
    arguments = parser.parse_args()
    binding = NativeTokenizerBinding.load()
    queue = HeldoutSemanticReviewQueue(binding, per_language=arguments.per_language)
    if arguments.admission_only:
        result = queue.materialize_admission_only(arguments.admission, arguments.output)
    else:
        result = queue.materialize(arguments.admission, arguments.tasks, arguments.output)
    print(
        json.dumps(
            {
                "item_count": result.item_count,
                "manifest_sha256": result.manifest_sha256,
                "missing_languages": result.missing_languages,
                "output_root": str(result.output_root),
                "queue_sha256": result.queue_sha256,
                "represented_languages": result.represented_languages,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
