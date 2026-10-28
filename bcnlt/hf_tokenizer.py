from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Sequence

from .tokenizer_manifest import (
    DEFAULT_INVENTORY_PATH,
    BaseTokenizerIdentity,
    TokenizerManifest,
    load_manifest,
    write_manifest,
)


MANIFEST_FILENAME = "bcnlt_tokenizer_manifest.json"
INVENTORY_FILENAME = "bcnlt-tokenizer-v1.tokens.json"
_TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
)


def resolve_tokenizer_source(
    source: str | Path,
    *,
    revision: str | None,
) -> tuple[Path, str]:
    """Resolve a local directory or a revision-pinned Hub model to local files."""

    local = Path(source)
    if local.is_dir():
        if not revision or not revision.strip():
            raise ValueError("A base tokenizer revision is required")
        return local, revision
    if not revision or not revision.strip():
        raise ValueError("A Hugging Face Hub tokenizer requires an explicit revision")

    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repo_id=str(source), revision=revision)
    commit = getattr(info, "sha", None)
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in commit)
    ):
        raise ValueError("Hugging Face did not resolve the tokenizer revision to a commit SHA")
    snapshot = Path(
        snapshot_download(
            repo_id=str(source),
            revision=commit,
            allow_patterns=list(_TOKENIZER_FILES),
        )
    )
    if not snapshot.is_dir():
        raise ValueError("Hugging Face tokenizer snapshot was not materialized locally")
    return snapshot, commit.lower()


class HuggingFaceBaseTokenizer:
    """A narrow, deterministic adapter around a Hugging Face tokenizer."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        revision: str | None = None,
        local_files_only: bool | None = None,
    ) -> "HuggingFaceBaseTokenizer":
        from transformers import AutoTokenizer

        if local_files_only is None:
            local_files_only = revision is None
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
            use_fast=True,
        )
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(
            self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        )

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self._tokenizer.get_vocab())


def identity_from_directory(
    path: str | Path,
    *,
    name: str,
    revision: str,
) -> BaseTokenizerIdentity:
    directory = Path(path)
    if not directory.is_dir():
        raise ValueError("Base tokenizer identity requires a local tokenizer directory")
    files = {
        filename: (directory / filename).read_bytes()
        for filename in _TOKENIZER_FILES
        if (directory / filename).is_file()
    }
    if not files:
        raise ValueError("Base tokenizer directory contains no recognized tokenizer files")
    return BaseTokenizerIdentity.from_files(name=name, revision=revision, files=files)


def save_tokenizer_artifact(
    path: str | Path,
    manifest: TokenizerManifest,
) -> tuple[Path, ...]:
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Tokenizer artifact directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_FILENAME
    inventory_path = destination / INVENTORY_FILENAME
    write_manifest(manifest_path, manifest)
    shutil.copyfile(DEFAULT_INVENTORY_PATH, inventory_path)
    _verify_inventory(inventory_path, manifest.structural_inventory_sha256)
    return manifest_path, inventory_path


def load_tokenizer_artifact(
    path: str | Path,
    base_path: str | Path,
):
    from .tokenizer import BCNLTTokenizer

    artifact = Path(path)
    manifest_path = artifact / MANIFEST_FILENAME
    inventory_path = artifact / INVENTORY_FILENAME
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)

    # Read the declared name and revision only to reconstruct the exact source identity.
    # load_manifest performs the schema and digest validation before accepting it.
    import json

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = raw["base_tokenizer"]
        name = declared["name"]
        revision = declared["revision"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read tokenizer artifact identity: {exc}") from exc
    if not isinstance(name, str) or not isinstance(revision, str):
        raise ValueError("Tokenizer artifact base identity is invalid")
    identity = identity_from_directory(
        base_path,
        name=name,
        revision=revision,
    )
    manifest = load_manifest(manifest_path, base, identity)
    _verify_inventory(inventory_path, manifest.structural_inventory_sha256)
    return BCNLTTokenizer(
        base,
        manifest,
        learned_memory_path=artifact / "bcnlt_learned_definitions.jsonl",
    )


def _verify_inventory(path: Path, expected_sha256: str) -> None:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot load tokenizer structural inventory: {exc}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Tokenizer artifact structural inventory hash mismatch")
