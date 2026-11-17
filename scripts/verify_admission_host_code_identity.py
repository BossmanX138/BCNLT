"""Verify the exact host-side code frozen for the V7 resumable admission run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "candidate-manifest.json"
HOST_FILES = (
    "scripts/run_full_resumable_admission.py",
    "bosscrafts_model/resumable_current_corpus.py",
    "bosscrafts_model/current_corpus.py",
    "bosscrafts_model/bcnlt_source_pipeline.py",
    "bosscrafts_model/tokenizer.py",
    "bosscrafts_model/bcnlt_adapter.py",
    "bosscrafts_model/semantic.py",
)


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    admission = manifest["full_admission"]
    expected_files = admission["host_code_files"]
    if set(expected_files) != set(HOST_FILES):
        raise ValueError("admission host-code file inventory mismatch")

    records: list[dict[str, object]] = []
    for relative in HOST_FILES:
        payload = (PROJECT_ROOT / relative).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_files[relative]:
            raise ValueError(f"admission host-code SHA-256 mismatch: {relative}")
        records.append({"path": relative, "sha256": digest, "bytes": len(payload)})

    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    aggregate = hashlib.sha256(canonical).hexdigest()
    if aggregate != admission["host_code_identity_sha256"]:
        raise ValueError("admission host-code aggregate identity mismatch")
    print(aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
