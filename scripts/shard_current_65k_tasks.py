"""Write current-spec task examples as compact balanced uint16 shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.balanced_binary_shards import BalancedBinaryShardWriter
from bosscrafts_model.tokenizer import NativeTokenizerBinding


DEFAULT_TASKS = (
    PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-current-spec-65536-pilot-tasks-v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-current-spec-65536-pilot-binary-v1"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens-per-shard", type=int, default=8_000_000)
    arguments = parser.parse_args()
    binding = NativeTokenizerBinding.load()
    result = BalancedBinaryShardWriter(
        binding, max_tokens_per_shard=arguments.max_tokens_per_shard
    ).materialize(arguments.tasks, arguments.output)
    print(
        json.dumps(
            {
                "example_count": result.example_count,
                "manifest_sha256": result.manifest_sha256,
                "output_root": str(result.output_root),
                "shard_count": result.shard_count,
                "token_count": result.token_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
