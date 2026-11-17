"""Build the bounded V10 curriculum from current-runtime audited eligibility."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.audited_balanced_curriculum import (  # noqa: E402
    build_audited_balanced_curriculum,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402
from scripts.run_v10_frozen_training_admission_v3_runtime_repair import (  # noqa: E402
    DEFAULT_CATALOG,
)


def _admission(split: str) -> Path:
    return (
        PROJECT_ROOT
        / "artifacts"
        / "corpora"
        / f"bcnlt-v10-frozen-{split}-admission-v3-runtime-repair"
        / "admission.sqlite"
    )


def _eligibility(split: str) -> Path:
    name = (
        "bcnlt-v10-audited-training-eligibility-v3-runtime-repair"
        if split == "train"
        else f"bcnlt-v10-frozen-{split}-eligibility-v3-runtime-repair"
    )
    return PROJECT_ROOT / "artifacts" / "corpora" / name


DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-current-audited-balanced-curriculum-v1"
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
    for split in ("train", "validation", "test"):
        parser.add_argument(
            f"--{split}-admission",
            type=Path,
            default=_admission(split),
        )
        parser.add_argument(
            f"--{split}-eligibility",
            type=Path,
            default=_eligibility(split),
        )
        parser.add_argument(
            f"--{split}-eligibility-manifest-sha256",
            required=True,
            type=_lower_sha256,
        )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--max-sequence", type=int, default=2048)
    parser.add_argument(
        "--semantic-wire",
        choices=("compact_v2", "full_v3"),
        default="compact_v2",
    )
    parser.add_argument("--train-per-cell", type=int, default=512)
    parser.add_argument("--validation-per-cell", type=int, default=128)
    parser.add_argument("--test-per-cell", type=int, default=128)
    parser.add_argument("--oversample-factor", type=int, default=4)
    arguments = parser.parse_args()

    admissions = {
        split: getattr(arguments, f"{split}_admission")
        for split in ("train", "validation", "test")
    }
    eligibility = {
        split: getattr(arguments, f"{split}_eligibility")
        for split in ("train", "validation", "test")
    }
    manifest_sha256 = {
        split: getattr(arguments, f"{split}_eligibility_manifest_sha256")
        for split in ("train", "validation", "test")
    }
    result = build_audited_balanced_curriculum(
        NativeTokenizerBinding.load(),
        admissions,
        eligibility,
        manifest_sha256,
        arguments.catalog,
        arguments.database,
        quotas={
            "train": arguments.train_per_cell,
            "validation": arguments.validation_per_cell,
            "test": arguments.test_per_cell,
        },
        oversample_factor=arguments.oversample_factor,
        max_sequence=arguments.max_sequence,
        semantic_wire=arguments.semantic_wire,
    )
    print(
        json.dumps(
            {**asdict(result), "database_path": str(result.database_path)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
