from __future__ import annotations

from pathlib import Path

from bcnlt.tokenizer import BCNLTTokenizer

from .manifest import load_adaptive_manifest, write_adaptive_manifest
from .tokenizer import AdaptiveBCNLTTokenizer


ADAPTIVE_MANIFEST_FILENAME = "bcnlt_adaptive_manifest.json"
V1_DIRECTORY = "v1"


def save_adaptive_tokenizer_artifact(
    path: str | Path, tokenizer: AdaptiveBCNLTTokenizer
) -> tuple[Path, ...]:
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Adaptive tokenizer artifact directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    v1_files = tokenizer.v1.save_pretrained(destination / V1_DIRECTORY)
    manifest_path = destination / ADAPTIVE_MANIFEST_FILENAME
    write_adaptive_manifest(manifest_path, tokenizer.manifest)
    return (manifest_path, *v1_files)


def load_adaptive_tokenizer_artifact(
    path: str | Path, base_path: str | Path
) -> AdaptiveBCNLTTokenizer:
    artifact = Path(path)
    v1 = BCNLTTokenizer.from_pretrained(artifact / V1_DIRECTORY, base_path)
    manifest = load_adaptive_manifest(
        artifact / ADAPTIVE_MANIFEST_FILENAME,
        v1.manifest,
    )
    return AdaptiveBCNLTTokenizer(v1, manifest)
