from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcnlt.parsers.bundle import freeze_parser_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze BCNLT's pinned multilingual parsers")
    parser.add_argument("--config", default="configs/bcnlt_parser_bundle_v1.json")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = freeze_parser_bundle(args.config, args.cache_root, args.output_root)
    except Exception as exc:
        parser.exit(1, f"BCNLT parser bundle freeze failed: {exc}\n")
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
