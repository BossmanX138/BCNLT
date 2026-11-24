#!/usr/bin/env python3
"""Build the frozen reviewed Wizard-Phi instruction curriculum for BCNLT."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.reviewed_instruction_curriculum import (  # noqa: E402
    build_reviewed_instruction_curriculum,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-database", type=Path, required=True)
    parser.add_argument("--max-sequence", type=int, default=2048)
    args = parser.parse_args()
    result = build_reviewed_instruction_curriculum(
        NativeTokenizerBinding.load(),
        source_root=args.source_root,
        output_database=args.output_database,
        max_sequence=args.max_sequence,
    )
    payload = asdict(result)
    payload["database_path"] = str(result.database_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
