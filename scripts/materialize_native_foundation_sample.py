from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bcnlt.native_foundation.config import NativeFoundationConfig
from bcnlt.native_foundation.corpus import materialize_balanced_sample


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic balanced tokenizer sample")
    parser.add_argument("--config", required=True)
    parser.add_argument("--accepted-corpus", required=True)
    parser.add_argument("--english-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bytes-per-language", type=int, required=True)
    parser.add_argument("--english-bytes", type=int, required=True)
    parser.add_argument("--progress-every", type=int, default=5_000)
    args = parser.parse_args()
    config = NativeFoundationConfig.load(args.config)
    manifest = materialize_balanced_sample(
        accepted_corpus_path=args.accepted_corpus,
        english_dataset_path=args.english_dataset,
        output_dir=args.output_dir,
        languages=config.canonical_code_languages,
        bytes_per_language=args.bytes_per_language,
        english_bytes=args.english_bytes,
        progress_every=args.progress_every,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

