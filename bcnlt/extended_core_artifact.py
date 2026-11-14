from __future__ import annotations

from pathlib import Path

from .extended_core import ExtendedCoreLibrary
from .extended_core_tokenizer import ExtendedCoreBCNLTTokenizer
from .extended_core_vocab import (
    load_extended_core_vocab,
    write_extended_core_vocab,
)
from .tokenizer import BCNLTTokenizer


EXTENDED_CORE_MANIFEST_FILENAME = "bcnlt_extended_core_manifest.json"
V1_DIRECTORY = "v1"


def save_extended_core_tokenizer_artifact(
    path: str | Path,
    tokenizer: ExtendedCoreBCNLTTokenizer,
) -> tuple[Path, ...]:
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            f"Extended Core tokenizer artifact directory is not empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    v1_files = tokenizer.v1.save_pretrained(destination / V1_DIRECTORY)
    manifest_path = destination / EXTENDED_CORE_MANIFEST_FILENAME
    write_extended_core_vocab(
        manifest_path,
        tokenizer.manifest,
        tokenizer.v1.manifest,
    )
    return (manifest_path, *v1_files)


def load_extended_core_tokenizer_artifact(
    path: str | Path,
    base_path: str | Path,
    *,
    library: ExtendedCoreLibrary | None = None,
) -> ExtendedCoreBCNLTTokenizer:
    artifact = Path(path)
    v1 = BCNLTTokenizer.from_pretrained(artifact / V1_DIRECTORY, base_path)
    manifest = load_extended_core_vocab(
        artifact / EXTENDED_CORE_MANIFEST_FILENAME,
        v1.manifest,
    )
    return ExtendedCoreBCNLTTokenizer(v1, manifest, library=library)
