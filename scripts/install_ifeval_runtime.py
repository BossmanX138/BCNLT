"""Install the exact hash-pinned IFEval evaluator and NLTK data runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.ifeval_runtime import install_ifeval_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    runtime = install_ifeval_runtime(arguments.output_root)
    print(
        json.dumps(
            {
                "root": str(runtime.root),
                "manifest_sha256": runtime.manifest_sha256,
                "evaluator_revision": runtime.evaluator_revision,
                "nltk_revision": runtime.nltk_revision,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
