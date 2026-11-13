"""Build or verify the blinded V10 semantic challenge-review queue."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.semantic_challenge_review import (  # noqa: E402
    SemanticChallengeReviewBuilder,
)

DEFAULT_SOURCE_QUEUE = (
    PROJECT_ROOT
    / "artifacts"
    / "reviews"
    / "bcnlt-v10-current-heldout-review-v2-context"
)
DEFAULT_NATURAL_LABELS = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "bcnlt-v10-semantic-uncertainty-v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "reviews"
    / "bcnlt-v10-semantic-challenge-v1"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-queue", type=Path, default=DEFAULT_SOURCE_QUEUE)
    parser.add_argument("--natural-labels", type=Path, default=DEFAULT_NATURAL_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs-per-language", type=int, default=10)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    builder = SemanticChallengeReviewBuilder(
        pairs_per_language=arguments.pairs_per_language
    )
    result = (
        builder.verify(arguments.output)
        if arguments.verify_only
        else builder.materialize(
            arguments.source_queue,
            arguments.natural_labels,
            arguments.output,
        )
    )
    print(
        json.dumps(
            {
                "answer_key_sha256": result.answer_key_sha256,
                "item_count": result.item_count,
                "languages": result.languages,
                "manifest_sha256": result.manifest_sha256,
                "output_root": str(result.output_root),
                "queue_sha256": result.queue_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
