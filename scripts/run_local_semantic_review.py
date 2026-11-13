"""Run or resume two local semantic reviewers against an immutable queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.local_semantic_review import (
    DualLocalSemanticReviewer,
    LocalReviewerSpec,
)


DEFAULT_QUEUE = (
    PROJECT_ROOT
    / "artifacts"
    / "reviews"
    / "bcnlt-v10-current-heldout-review-v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "reviews"
    / "bcnlt-v10-current-heldout-dual-local-v1"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect resumable local-only dual semantic review evidence."
    )
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviewer-a-id", required=True)
    parser.add_argument("--reviewer-a-url", required=True)
    parser.add_argument("--reviewer-a-model", required=True)
    parser.add_argument("--reviewer-b-id", required=True)
    parser.add_argument("--reviewer-b-url", required=True)
    parser.add_argument("--reviewer-b-model", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument(
        "--limit",
        type=int,
        help="Maximum pending reviewer-item pairs to process this invocation.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Record invalid reviewer responses and continue with other pending calls.",
    )
    limits.add_argument(
        "--limit-per-reviewer",
        type=int,
        help="Maximum missing items to process for each reviewer this invocation.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    reviewers = (
        LocalReviewerSpec(
            arguments.reviewer_a_id,
            arguments.reviewer_a_url,
            arguments.reviewer_a_model,
            timeout_seconds=arguments.timeout_seconds,
        ),
        LocalReviewerSpec(
            arguments.reviewer_b_id,
            arguments.reviewer_b_url,
            arguments.reviewer_b_model,
            timeout_seconds=arguments.timeout_seconds,
        ),
    )
    result = DualLocalSemanticReviewer(
        arguments.queue, arguments.output, reviewers
    ).run(
        limit=arguments.limit,
        limit_per_reviewer=arguments.limit_per_reviewer,
        continue_on_failure=arguments.continue_on_failure,
    )
    print(
        json.dumps(
            {
                "completed_reviews": result.completed_reviews,
                "failed_attempts": result.failed_attempts,
                "consensus_counts": result.consensus_counts,
                "manifest_sha256": result.manifest_sha256,
                "output_root": str(result.output_root),
                "required_reviews": result.required_reviews,
                "review_complete": result.review_complete,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
