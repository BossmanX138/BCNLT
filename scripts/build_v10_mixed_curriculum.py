#!/usr/bin/env python3
"""Merge audited semantic and reviewed instruction task databases."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.mixed_curriculum import build_mixed_curriculum  # noqa: E402
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-database", type=Path, required=True)
    parser.add_argument("--instruction-database", type=Path, required=True)
    parser.add_argument("--output-database", type=Path, required=True)
    parser.add_argument("--semantic-identity-sha256", required=True)
    parser.add_argument("--instruction-identity-sha256", required=True)
    parser.add_argument("--instruction-weight-fraction", type=float, default=0.5)
    arguments = parser.parse_args()
    result = build_mixed_curriculum(
        NativeTokenizerBinding.load(),
        semantic_database=arguments.semantic_database,
        instruction_database=arguments.instruction_database,
        output_database=arguments.output_database,
        expected_semantic_identity_sha256=arguments.semantic_identity_sha256,
        expected_instruction_identity_sha256=arguments.instruction_identity_sha256,
        instruction_training_weight_fraction=arguments.instruction_weight_fraction,
    )
    payload = asdict(result)
    payload["database_path"] = str(result.database_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
