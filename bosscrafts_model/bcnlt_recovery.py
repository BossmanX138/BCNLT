"""Fail-closed readiness gate for the recovered BCNLT parser lineage."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .tokenizer import EXPECTED_BINDING_SHA256, EXPECTED_TOKENIZER_SHA256

EXPECTED_RECOVERY_MANIFEST_SHA256 = "e2cda5ca4785a2e4d79845ab2dd942cb116546a81b37d999651bce92354fafdc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class BCNLTRecoveryStatus:
    source_contract_recovered: bool
    output_consumer_integrated: bool
    live_parser_ready: bool
    admitted_corpus_tokenizer_compatible: bool
    training_ready: bool
    blockers: tuple[str, ...]


def load_bcnlt_recovery_status(project_root: str | Path | None = None) -> BCNLTRecoveryStatus:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    recovery_root = root / "artifacts" / "bcnlt" / "runtime-recovery-v1"
    manifest_path = recovery_root / "recovery-manifest.json"
    if _sha256(manifest_path) != EXPECTED_RECOVERY_MANIFEST_SHA256:
        raise ValueError("BCNLT recovery manifest hash mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "bosscrafts-bcnlt-runtime-recovery-v1":
        raise ValueError("unsupported BCNLT recovery manifest")

    contract = payload["parser_contract"]
    specs_path = recovery_root / contract["semantic_specs_file"]
    if _sha256(specs_path) != contract["semantic_specs_file_sha256"]:
        raise ValueError("BCNLT semantic spec snapshot hash mismatch")
    for relative, identity in payload["source"]["files"].items():
        path = recovery_root / "source" / relative
        if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"BCNLT recovered source mismatch: {relative}")
    for relative, identity in payload["admitted_corpus"]["evidence_files"].items():
        path = recovery_root / "admitted-corpus" / relative
        if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"BCNLT admitted-corpus evidence mismatch: {relative}")
    for relative, identity in payload["runtime"]["metadata_files"].items():
        path = recovery_root / "runtime-metadata" / relative
        if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"BCNLT runtime metadata mismatch: {relative}")

    frozen = payload["frozen_model_identity"]
    if (
        frozen["tokenizer_vocab_size"] != 65_536
        or frozen["tokenizer_sha256"] != EXPECTED_TOKENIZER_SHA256
        or frozen["binding_sha256"] != EXPECTED_BINDING_SHA256
    ):
        raise ValueError("BCNLT recovery evidence does not preserve frozen tokenizer identity")
    readiness = payload["readiness"]
    status = BCNLTRecoveryStatus(
        source_contract_recovered=readiness["source_contract_recovered"],
        output_consumer_integrated=readiness["output_consumer_integrated"],
        live_parser_ready=readiness["live_parser_ready"],
        admitted_corpus_tokenizer_compatible=readiness["admitted_corpus_tokenizer_compatible"],
        training_ready=readiness["training_ready"],
        blockers=tuple(readiness["blockers"]),
    )
    if status.live_parser_ready or status.training_ready or not status.blockers:
        raise ValueError("BCNLT recovery readiness must remain fail-closed while blockers exist")
    return status
