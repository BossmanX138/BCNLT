from __future__ import annotations

import argparse
from pathlib import Path

from bcnlt.corpus.packing import pack_training_shards
from bcnlt.tokenizer import BCNLTTokenizer

from scripts.validate_bcnlt_corpus import _load_curriculum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack admitted BCNLT curriculum shards")
    parser.add_argument("--records", required=True)
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-sequence-tokens", type=int, default=4096)
    parser.add_argument("--shard-token-limit", type=int, default=8_000_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    tokenizer = BCNLTTokenizer.from_pretrained(
        args.tokenizer_artifact, args.base_tokenizer
    )
    manifest = pack_training_shards(
        _load_curriculum(Path(args.records)),
        tokenizer,
        args.output,
        maximum_sequence_tokens=args.maximum_sequence_tokens,
        shard_token_limit=args.shard_token_limit,
        resume=args.resume,
    )
    print(manifest.dataset_manifest_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
