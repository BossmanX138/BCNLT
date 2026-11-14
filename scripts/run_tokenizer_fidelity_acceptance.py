"""Run frozen tokenizer-fidelity cases and publish an immutable receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.tokenizer import NativeTokenizerBinding
from bosscrafts_model.tokenizer_fidelity_acceptance import (
    build_tokenizer_fidelity_receipt,
    publish_tokenizer_fidelity_receipt,
    run_tokenizer_fidelity_cases,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-bundle", type=Path, required=True)
    parser.add_argument("--test-database", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    binding = NativeTokenizerBinding.load()
    result = run_tokenizer_fidelity_cases(
        binding,
        case_bundle_root=arguments.case_bundle,
        test_task_database=arguments.test_database,
    )
    receipt = build_tokenizer_fidelity_receipt(
        result,
        checkpoint_sha256=_sha256_file(arguments.checkpoint),
        checkpoint_step=arguments.checkpoint_step,
    )
    publish_tokenizer_fidelity_receipt(arguments.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
