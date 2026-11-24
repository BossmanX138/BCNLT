"""Build a resumable final-code curriculum from pinned CodeX-2M shards."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PINNED_TOKENIZERS_SITE = PROJECT_ROOT / "vendor" / "tokenizers_runtime_0_23"
if not (PINNED_TOKENIZERS_SITE / "tokenizers" / "tokenizers.pyd").is_file():
    raise RuntimeError("project-contained tokenizers 0.23 runtime is missing")
if "tokenizers" in sys.modules:
    raise RuntimeError("tokenizers was imported before the pinned runtime was selected")
sys.path.insert(0, str(PINNED_TOKENIZERS_SITE))

import tokenizers

if tokenizers.__version__ != "0.23.2":
    raise RuntimeError("project-contained tokenizers runtime version mismatch")

from bosscrafts_model.codex_code_curriculum import build_codex_code_curriculum
from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_DATASET = Path(r"D:\codex-2m-thinking")
DEFAULT_OUTPUT = Path(r"D:\BCNLT-runs\bosscrafts-engineering-v12-codex-phase1")
DEFAULT_HUMANEVAL = (
    PROJECT_ROOT
    / "artifacts"
    / "benchmarks"
    / "openai-humaneval-v1"
    / "source"
    / "HumanEval.jsonl.gz"
)
DEFAULT_PHASE1_INDICES = (0, 17, 33, 50, 67, 83, 100, 117, 133, 150, 167, 183, 200, 217, 233, 250)


def _parse_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("shard indices must be comma-separated integers") from error
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(not 0 <= index <= 250 for index in indices)
    ):
        raise argparse.ArgumentTypeError("shard indices must be unique values in [0,250]")
    return indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--humaneval-dataset", type=Path, default=DEFAULT_HUMANEVAL)
    parser.add_argument(
        "--shard-indices",
        type=_parse_indices,
        default=DEFAULT_PHASE1_INDICES,
        help="Comma-separated source shard indices; defaults to 16 evenly spaced shards.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-sequence", type=int, default=2048)
    arguments = parser.parse_args()
    result = build_codex_code_curriculum(
        NativeTokenizerBinding.load(),
        dataset_root=arguments.dataset_root,
        shard_indices=arguments.shard_indices,
        output_root=arguments.output_root,
        humaneval_dataset=arguments.humaneval_dataset,
        max_sequence=arguments.max_sequence,
        workers=arguments.workers,
        batch_size=arguments.batch_size,
    )
    payload = asdict(result)
    payload["database_path"] = str(result.database_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
