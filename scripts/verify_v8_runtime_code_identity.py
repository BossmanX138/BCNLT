"""Verify the exact V8 tokenizer, model, training, evaluation and inference code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "candidate-manifest.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runtime = manifest.get("runtime_code")
    if type(runtime) is not dict or type(runtime.get("files")) is not dict:
        raise ValueError("V8 runtime-code manifest is invalid")
    expected_files = runtime["files"]
    records: list[dict[str, object]] = []
    for relative, expected_sha256 in expected_files.items():
        if type(relative) is not str or type(expected_sha256) is not str:
            raise ValueError("V8 runtime-code inventory is invalid")
        payload = (PROJECT_ROOT / relative).read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected_sha256:
            raise ValueError(f"V8 runtime-code SHA-256 mismatch: {relative}")
        records.append(
            {"path": relative, "sha256": observed, "bytes": len(payload)}
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    identity = hashlib.sha256(canonical).hexdigest()
    if identity != runtime.get("identity_sha256"):
        raise ValueError("V8 runtime-code aggregate identity mismatch")
    print(identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
