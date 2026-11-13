"""Build a held-out-safe Python signature/docstring-to-body curriculum."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

from bosscrafts_model.instruction_curriculum import build_instruction_curriculum
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--admission-database", type=Path, required=True)
    parser.add_argument("--split-database", type=Path, required=True)
    parser.add_argument("--humaneval-dataset", type=Path, required=True)
    parser.add_argument("--output-database", type=Path, required=True)
    parser.add_argument("--train", type=int, default=4096)
    parser.add_argument("--validation", type=int, default=256)
    parser.add_argument("--test", type=int, default=256)
    parser.add_argument("--max-sequence", type=int, default=2048)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()

    status = build_instruction_curriculum(
        NativeTokenizerBinding.load(),
        source_catalog=arguments.source_catalog,
        admission_database=arguments.admission_database,
        split_database=arguments.split_database,
        humaneval_dataset=arguments.humaneval_dataset,
        output_database=arguments.output_database,
        quotas={
            "train": arguments.train,
            "validation": arguments.validation,
            "test": arguments.test,
        },
        max_sequence=arguments.max_sequence,
    )
    payload = asdict(status)
    payload["database_path"] = str(status.database_path)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if arguments.report is not None:
        destination = arguments.report.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(destination)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
