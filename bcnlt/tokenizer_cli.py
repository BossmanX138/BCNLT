from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .hf_tokenizer import (
    MANIFEST_FILENAME,
    HuggingFaceBaseTokenizer,
    identity_from_directory,
    load_tokenizer_artifact,
)
from .tokenizer import BCNLTTokenizer
from .tokenizer_manifest import build_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bcnlt-tokenizer")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build a BCNLT tokenizer artifact")
    build.add_argument("--base", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--name", required=True)
    build.add_argument("--revision", required=True)

    inspect = commands.add_parser("inspect", help="inspect an artifact manifest")
    inspect.add_argument("--artifact", required=True)

    encode = commands.add_parser("encode", help="encode text or Python source")
    encode.add_argument("--artifact", required=True)
    encode.add_argument("--base", required=True)
    encode.add_argument(
        "--mode", choices=("text", "raw_code", "semantic_code"), required=True
    )
    content = encode.add_mutually_exclusive_group(required=True)
    content.add_argument("--text")
    content.add_argument("--file")

    verify = commands.add_parser("verify", help="verify an artifact and its base")
    verify.add_argument("--artifact", required=True)
    verify.add_argument("--base", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            return _build(args)
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "encode":
            return _encode(args)
        if args.command == "verify":
            return _verify(args)
    except Exception as exc:
        prefix = "verification failed" if args.command == "verify" else f"{args.command} failed"
        print(f"{prefix}: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"Unhandled command: {args.command}")


def _build(args: argparse.Namespace) -> int:
    base = HuggingFaceBaseTokenizer.from_pretrained(args.base)
    identity = identity_from_directory(
        args.base,
        name=args.name,
        revision=args.revision,
    )
    tokenizer = BCNLTTokenizer(base, build_manifest(base, identity))
    files = tokenizer.save_pretrained(args.output)
    _print_json(
        {
            "manifest_sha256": tokenizer.manifest.manifest_sha256,
            "vocab_size": tokenizer.manifest.vocab_size,
            "files": [str(path) for path in files],
        }
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    manifest_path = Path(args.artifact) / MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "schema_version": payload["schema_version"],
            "canonicalization_version": payload["canonicalization_version"],
            "base_tokenizer": payload["base_tokenizer"],
            "base_vocab_size": payload["base_vocab_size"],
            "added_token_count": len(payload["added_tokens"]),
            "manifest_sha256": payload["manifest_sha256"],
        }
    )
    return 0


def _encode(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer_artifact(args.artifact, args.base)
    content = args.text if args.text is not None else Path(args.file).read_text(encoding="utf-8")
    if args.mode == "text":
        ids = tokenizer.encode_text(content)
    elif args.mode == "raw_code":
        ids = tokenizer.encode_raw_code(content)
    else:
        ids = tokenizer.encode_source(content, path=args.file or "<cli>")
    payload = {"mode": args.mode, "ids": ids, "decoded": tokenizer.decode(ids)}
    if args.mode == "semantic_code":
        payload["record_count"] = len(tokenizer.decode_records(ids))
    _print_json(payload)
    return 0


def _verify(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer_artifact(args.artifact, args.base)
    text = "BCNLT round trip\n"
    text_ids = tokenizer.encode_text(text)
    if tokenizer.decode(text_ids) != text:
        raise ValueError("text round-trip check failed")
    semantic_ids = tokenizer.encode_source("answer = 42\n", path="<verify>")
    records = tokenizer.decode_records(semantic_ids)
    if not records or tokenizer.encode_records(records) != semantic_ids:
        raise ValueError("semantic round-trip check failed")
    _print_json(
        {
            "verified": True,
            "manifest_sha256": tokenizer.manifest.manifest_sha256,
            "base_tokenizer_sha256": tokenizer.manifest.base_tokenizer_sha256,
            "semantic_record_count": len(records),
        }
    )
    return 0


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
