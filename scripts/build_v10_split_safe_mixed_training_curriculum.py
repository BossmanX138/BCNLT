#!/usr/bin/env python3
"""Build a train-only semantic and reviewed-instruction V10 curriculum."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.mixed_curriculum import (  # noqa: E402
    build_split_safe_mixed_training_curriculum,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-train-database", type=Path, required=True)
    parser.add_argument("--instruction-database", type=Path, required=True)
    parser.add_argument("--output-database", type=Path, required=True)
    parser.add_argument(
        "--semantic-train-identity-sha256", required=True, type=_lower_sha256
    )
    parser.add_argument(
        "--instruction-identity-sha256", required=True, type=_lower_sha256
    )
    parser.add_argument("--semantic-train-audit-manifest", type=Path)
    parser.add_argument("--instruction-audit-manifest", type=Path)
    parser.add_argument("--output-audit-manifest", type=Path)
    parser.add_argument("--instruction-weight-fraction", type=float, default=0.5)
    arguments = parser.parse_args()
    result = build_split_safe_mixed_training_curriculum(
        NativeTokenizerBinding.load(),
        semantic_train_database=arguments.semantic_train_database,
        instruction_database=arguments.instruction_database,
        output_database=arguments.output_database,
        expected_semantic_train_identity_sha256=(
            arguments.semantic_train_identity_sha256
        ),
        expected_instruction_identity_sha256=arguments.instruction_identity_sha256,
        instruction_training_weight_fraction=arguments.instruction_weight_fraction,
        semantic_train_audit_manifest=arguments.semantic_train_audit_manifest,
        instruction_audit_manifest=arguments.instruction_audit_manifest,
        output_audit_manifest=arguments.output_audit_manifest,
    )
    payload = asdict(result)
    payload["database_path"] = str(result.database_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
