"""Verify exact V9 tokenizer, model, training, evaluation and inference code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "candidate-manifest.json"
POST_TRAINING_SCRIPTS = (
    "scripts/evaluate_checkpoint.py",
    "scripts/evaluate_generation_gate.py",
    "scripts/run_checkpoint_inference.py",
    "scripts/verify_model_acceptance.py",
    "scripts/export_inference_bundle.py",
    "scripts/run_inference_bundle.py",
    "scripts/verify_v9_runtime_code_identity.py",
)


def _complete_inventory() -> tuple[str, ...]:
    modules = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "bosscrafts_model").glob("*.py")
    )
    return tuple((*modules, *POST_TRAINING_SCRIPTS))


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runtime = manifest.get("runtime_code")
    if type(runtime) is not dict or type(runtime.get("files")) is not dict:
        raise ValueError("V9 runtime-code manifest is invalid")
    expected_files = runtime["files"]
    explicit_hashes: dict[str, str] = {}
    for relative, expected_sha256 in expected_files.items():
        if type(relative) is not str or type(expected_sha256) is not str:
            raise ValueError("V9 runtime-code inventory is invalid")
        payload = (PROJECT_ROOT / relative).read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected_sha256:
            raise ValueError(f"V9 runtime-code SHA-256 mismatch: {relative}")
        explicit_hashes[relative] = observed

    inventory = _complete_inventory()
    if not set(explicit_hashes).issubset(inventory):
        raise ValueError("V9 explicit runtime inventory is outside the frozen policy")
    records: list[dict[str, object]] = []
    for relative in inventory:
        payload = (PROJECT_ROOT / relative).read_bytes()
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    identity = hashlib.sha256(canonical).hexdigest()
    if identity != runtime.get("identity_sha256"):
        raise ValueError("V9 complete runtime-code aggregate identity mismatch")
    print(identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
