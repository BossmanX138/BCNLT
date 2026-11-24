"""Check whether the pinned language pack fetches a grammar on cache miss."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SITE = PROJECT_ROOT / "vendor" / "bcnlt_runtime_v2" / "site-packages"
sys.path.insert(0, str(RUNTIME_SITE))

from tree_sitter_language_pack import PackConfig, configure, get_parser  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="bcnlt-language-pack-audit-") as temporary:
        root = Path(temporary)
        configure(PackConfig(cache_dir=str(root)))
        before = sorted(path.name for path in root.rglob("*") if path.is_file())
        started = time.perf_counter()
        error = None
        parser_loaded = False
        try:
            parser_loaded = get_parser("python") is not None
        except Exception as exc:  # audit records the exact cache-miss behavior
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        after = sorted(
            {str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file()}
        )
        print(
            json.dumps(
                {
                    "before": before,
                    "after": after,
                    "parser_loaded": parser_loaded,
                    "error": error,
                    "elapsed_seconds": elapsed,
                    "cache_populated_on_miss": bool(after),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
