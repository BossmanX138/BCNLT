"""Build an immutable two-split held-out suite before model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.heldout_suite_builder import (
    build_heldout_suite_manifest,
    load_category_case_inventory,
    publish_heldout_suite_manifest,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-database", type=Path, required=True)
    parser.add_argument("--test-database", type=Path, required=True)
    parser.add_argument("--category-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    manifest = build_heldout_suite_manifest(
        NativeTokenizerBinding.load(),
        validation_database=arguments.validation_database,
        test_database=arguments.test_database,
        category_case_inventory=load_category_case_inventory(
            arguments.category_inventory
        ),
    )
    publish_heldout_suite_manifest(arguments.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
