"""Freeze one exact upstream benchmark source into acceptance-case JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.category_case_sources import (
    FrozenCategoryCases,
    freeze_aegis_test_cases,
    freeze_ifeval_cases,
)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)


def _result_payload(result: FrozenCategoryCases) -> dict[str, object]:
    return {
        "output": str(result.output),
        "record_count": result.record_count,
        "source_sha256": result.source_sha256,
        "revision": result.revision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="source_kind", required=True)
    _add_source_arguments(subparsers.add_parser("ifeval"))
    _add_source_arguments(subparsers.add_parser("aegis-test"))
    arguments = parser.parse_args()

    keyword_arguments = {
        "revision": arguments.revision,
        "expected_source_sha256": arguments.source_sha256,
    }
    if arguments.source_kind == "ifeval":
        result = freeze_ifeval_cases(
            arguments.source,
            arguments.output,
            **keyword_arguments,
        )
    elif arguments.source_kind == "aegis-test":
        result = freeze_aegis_test_cases(
            arguments.source,
            arguments.output,
            **keyword_arguments,
        )
    else:
        raise AssertionError(f"unhandled source kind: {arguments.source_kind}")
    print(json.dumps(_result_payload(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
