"""Build the frozen V12 CodeX plus BCNLT replay curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


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

from bosscrafts_model.codex_replay_curriculum import build_codex_replay_curriculum
from bosscrafts_model.tokenizer import NativeTokenizerBinding


EXPECTED_CODEX_IDENTITY = "249c068873764d4594427ac06bf840a1729545a2f83a416c8706a38d916783c9"
EXPECTED_REPLAY_IDENTITY = "626186864cf4444db997bea5bd8ea4a18f18b5042509033a0d6939d0bd35afd2"
DEFAULT_CODEX = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-codex-2m-final-code-phase1-v12"
    / "tasks.sqlite"
)
DEFAULT_REPLAY = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-current-spec-65536-deterministic-clean-v5"
    / "tasks.sqlite"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-codex-2m-bcnlt-replay-weighted-v12"
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex", type=Path, default=DEFAULT_CODEX)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--replay-per-train-cell", type=int, default=512)
    parser.add_argument("--codex-weight-fraction", type=float, default=0.80)
    parser.add_argument("--order-window-size", type=int, default=800)
    arguments = parser.parse_args()
    output_root = arguments.output_root.resolve()
    report = build_codex_replay_curriculum(
        NativeTokenizerBinding.load(),
        codex_database=arguments.codex.resolve(),
        replay_database=arguments.replay.resolve(),
        destination=output_root / "tasks.sqlite",
        replay_per_train_cell=arguments.replay_per_train_cell,
        codex_weight_fraction=arguments.codex_weight_fraction,
        order_window_size=arguments.order_window_size,
        expected_codex_identity=EXPECTED_CODEX_IDENTITY,
        expected_replay_identity=EXPECTED_REPLAY_IDENTITY,
    )
    _atomic_json(output_root / "build-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
