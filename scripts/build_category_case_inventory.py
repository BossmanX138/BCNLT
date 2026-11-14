"""Freeze content-addressed case catalogs for every acceptance category."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.category_case_inventory import (
    publish_category_case_bundle,
)
from bosscrafts_model.model_acceptance import ACCEPTANCE_CATEGORIES


def main() -> int:
    parser = argparse.ArgumentParser()
    for category in ACCEPTANCE_CATEGORIES:
        parser.add_argument(
            f"--{category.replace('_', '-')}-cases",
            dest=f"{category}_cases",
            type=Path,
            required=True,
        )
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    bundle = publish_category_case_bundle(
        arguments.output_directory,
        {
            category: getattr(arguments, f"{category}_cases")
            for category in ACCEPTANCE_CATEGORIES
        },
    )
    print(
        json.dumps(
            {
                "bundle_root": str(bundle.bundle_root),
                "manifest_sha256": bundle.manifest_sha256,
                "inventory_sha256": bundle.inventory.inventory_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
