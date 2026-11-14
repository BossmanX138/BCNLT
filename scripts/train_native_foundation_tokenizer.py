from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bcnlt.native_foundation.trainer import train_byte_bpe_tokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_corpus_files(corpus_dir: Path) -> tuple[Path, ...]:
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("sample manifest is not complete")
    names = [f"{language}.txt" for language in manifest["languages"]] + ["english.txt"]
    expected_hashes = manifest.get("file_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("sample manifest has no file hashes")
    files = tuple(corpus_dir / name for name in names)
    for path in files:
        if not path.is_file() or _sha256(path) != expected_hashes.get(path.name):
            raise ValueError(f"sample file failed hash validation: {path}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an isolated BCNLT byte-BPE baseline")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--structural-tokens", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocabulary-size", type=int, required=True)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    manifest = train_byte_bpe_tokenizer(
        corpus_files=_validated_corpus_files(Path(args.corpus_dir)),
        structural_tokens_path=args.structural_tokens,
        output_dir=args.output_dir,
        vocabulary_size=args.vocabulary_size,
        min_frequency=args.min_frequency,
        threads=args.threads,
        label=args.label,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
