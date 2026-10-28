from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from bcnlt.hf_tokenizer import HuggingFaceBaseTokenizer, identity_from_directory
from bcnlt.tokenizer import BCNLTTokenizer
from bcnlt.tokenizer_manifest import build_manifest


SCHEMA_VERSION = "runeforge-mk2-bcnlt-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def build_tokenizer_artifact(
    config_path: str | Path,
    preflight_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path)
    preflight_file = Path(preflight_path)
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"Tokenizer output already exists: {output}")

    config = _load_json(config_file, "Runeforge tokenizer configuration")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Runeforge tokenizer configuration schema")
    evidence = _load_json(preflight_file, "Runeforge preflight evidence")
    if evidence.get("status") != "passed":
        raise ValueError("Runeforge preflight evidence did not pass")
    config_sha256 = _sha256(config_file)
    if evidence.get("config_sha256") != config_sha256:
        raise ValueError("Runeforge preflight configuration hash mismatch")

    base_config = config.get("base")
    if not isinstance(base_config, dict):
        raise ValueError("Runeforge base configuration is missing")
    base_path = Path(str(base_config.get("windows_path", "")))
    if not base_path.is_dir():
        raise ValueError(f"Runeforge base tokenizer directory is unavailable: {base_path}")
    expected_hashes = base_config.get("required_file_sha256")
    evidence_hashes = evidence.get("base_file_sha256")
    if not isinstance(expected_hashes, dict) or not isinstance(evidence_hashes, dict):
        raise ValueError("Runeforge base file hashes are missing")
    actual_hashes: dict[str, str] = {}
    for filename, expected in sorted(expected_hashes.items()):
        source = base_path / filename
        if not source.is_file():
            raise ValueError(f"Runeforge base file is missing: {filename}")
        actual = _sha256(source)
        actual_hashes[filename] = actual
        if actual != expected or evidence_hashes.get(filename) != expected:
            raise ValueError(f"Runeforge base file hash mismatch: {filename}")

    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    identity = identity_from_directory(
        base_path,
        name=str(base_config.get("name", "")),
        revision=str(base_config.get("revision", "")),
    )
    tokenizer = BCNLTTokenizer(base, build_manifest(base, identity))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        tokenizer.save_pretrained(temporary)
        verified = BCNLTTokenizer.from_pretrained(temporary, base_path)
        probe = "Runeforge BCNLT verification\n"
        if verified.decode(verified.encode_text(probe)) != probe:
            raise ValueError("Runeforge BCNLT tokenizer text round-trip failed")
        semantic_ids = verified.encode_source("answer = 42\n", path="<lineage-check>")
        if verified.encode_records(verified.decode_records(semantic_ids)) != semantic_ids:
            raise ValueError("Runeforge BCNLT tokenizer semantic round-trip failed")

        lineage = {
            "schema_version": "runeforge-mk2-bcnlt-lineage-v1",
            "config_sha256": config_sha256,
            "preflight_sha256": _sha256(preflight_file),
            "base_name": identity.name,
            "base_revision": identity.revision,
            "base_tokenizer_sha256": identity.sha256,
            "base_file_sha256": actual_hashes,
            "manifest_sha256": tokenizer.manifest.manifest_sha256,
            "vocab_size": tokenizer.manifest.vocab_size,
        }
        (temporary / "tokenizer_lineage.json").write_text(
            json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)
        return lineage
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the pinned Runeforge BCNLT tokenizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        lineage = build_tokenizer_artifact(args.config, args.preflight, args.output)
    except Exception as exc:
        parser.exit(1, f"Runeforge BCNLT tokenizer build failed: {exc}\n")
    print(json.dumps(lineage, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
