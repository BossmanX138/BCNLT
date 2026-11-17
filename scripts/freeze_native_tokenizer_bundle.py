"""Freeze metadata for an already copied tokenizer bundle.

This script never downloads or mutates the source tokenizer. It refuses to
overwrite binding artifacts and writes only inside the supplied bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_TOKENIZER_SHA256 = (
    "e1930449f3efe99bbb3f63730f75f71788d4ff21928bcae4d3ee2a520d30d407"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    root = Path(args.bundle).resolve()
    generated_names = (
        "structural-token-ids.json",
        "model-config-65536.json",
        "binding.json",
        "binding.sha256",
    )
    existing = [name for name in generated_names if (root / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite binding artifacts: {existing}")

    tokenizer_path = root / "tokenizer.json"
    tokenizer_hash = sha256(tokenizer_path)
    if tokenizer_hash != EXPECTED_TOKENIZER_SHA256:
        raise ValueError(f"source tokenizer SHA-256 mismatch: {tokenizer_hash}")
    raw = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    structural_path = root / "structural-tokens.json"
    structural_payload = json.loads(structural_path.read_text(encoding="utf-8"))
    tokens = structural_payload["tokens"]
    vocab = raw["model"]["vocab"]
    ids = {token: vocab[token] for token in tokens}

    ids_path = root / "structural-token-ids.json"
    write_json(
        ids_path,
        {
            "schema_version": "bosscrafts-structural-token-ids-v1",
            "tokenizer_sha256": tokenizer_hash,
            "token_count": len(ids),
            "id_min": min(ids.values()),
            "id_max": max(ids.values()),
            "ids": ids,
        },
    )
    model_config_path = root / "model-config-65536.json"
    write_json(
        model_config_path,
        {
            "schema_version": "bosscrafts-engineering-model-config-v1",
            "vocab_size": 65_536,
            "width": 1_280,
            "layers": 24,
            "heads": 20,
            "ff_width": 3_072,
            "module_width": 1_024,
            "module_rounds": 2,
            "max_sequence": 2_048,
            "modular": True,
            "tied_input_output_embeddings": True,
            "parameter_count": 540_094_726,
            "tokenizer_sha256": tokenizer_hash,
            "token_id_policy": "frozen_exact_no_resize_no_renumber",
        },
    )
    names = (
        "tokenizer.json",
        "source-manifest.json",
        "structural-tokens.json",
        "structural-token-ids.json",
        "model-config-65536.json",
    )
    files = {
        name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size}
        for name in names
    }
    binding_path = root / "binding.json"
    write_json(
        binding_path,
        {
            "schema_version": "bosscrafts-native-tokenizer-binding-v1",
            "lineage": "bcnlt-native-foundation-65536-v1",
            "vocabulary_size": 65_536,
            "minimum_token_id": 0,
            "maximum_token_id": 65_535,
            "tokenizer_json_version": "1.0",
            "tokenizer_model_type": "BPE",
            "token_id_policy": "frozen_exact_no_resize_no_renumber",
            "runtime": {
                "package": "tokenizers",
                "required_major_minor": "0.23",
            },
            "structural_tokens": {
                "count": 73,
                "minimum_id": 4,
                "maximum_id": 76,
                "contiguous": True,
                "atomic_required": True,
            },
            "semantic_boundary": {
                "mode": "raw_byte_bpe",
                "semantic_record_parser_integrated": False,
                "claim": "serialized semantic records round-trip as raw text only",
            },
            "files": files,
            "source_copy": {
                "source_tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
                "copy_policy": "verified_copy_preserve_source",
            },
        },
    )
    binding_hash = sha256(binding_path)
    (root / "binding.sha256").write_text(
        f"{binding_hash}  binding.json\n", encoding="ascii"
    )
    print(json.dumps({"binding_sha256": binding_hash, "files": files}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
