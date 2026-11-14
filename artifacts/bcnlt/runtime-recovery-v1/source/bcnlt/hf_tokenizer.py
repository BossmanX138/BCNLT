from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Literal, Sequence

from .tokenizer_manifest import (
    DEFAULT_INVENTORY_PATH,
    BaseTokenizerIdentity,
    TokenizerManifest,
    load_manifest,
    write_manifest,
)


MANIFEST_FILENAME = "bcnlt_tokenizer_manifest.json"
INVENTORY_FILENAME = "bcnlt-tokenizer-v1.tokens.json"
LEARNED_DEFINITIONS_FILENAME = "bcnlt_learned_definitions.jsonl"
LEARNED_DEFINITIONS_SQLITE_FILENAME = "bcnlt_learned_definitions.sqlite3"
LEARNED_STATE_FILENAME = "bcnlt_learned_state.json"
LEARNED_STATE_SCHEMA = "bcnlt-learned-state-v1"
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
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        if local_files_only is None:
            local_files_only = revision is None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(path),
                revision=revision,
                local_files_only=local_files_only,
                trust_remote_code=False,
                use_fast=True,
            )
        except ValueError as exc:
            local = Path(path)
            config_path = local / "tokenizer_config.json"
            tokenizer_path = local / "tokenizer.json"
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise exc
            if config.get("tokenizer_class") != "TokenizersBackend" or not tokenizer_path.is_file():
                raise exc
            special = {
                name: config[name]
                for name in ("bos_token", "eos_token", "unk_token", "pad_token")
                if isinstance(config.get(name), str)
            }
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_path),
                clean_up_tokenization_spaces=False,
                **special,
            )
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(
            self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        )

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        if not getattr(self._tokenizer, "is_fast", False):
            raise ValueError("Offset mapping requires a verified fast tokenizer")
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = list(encoded["input_ids"])
        offsets = [tuple(map(int, item)) for item in encoded["offset_mapping"]]
        if len(ids) != len(offsets):
            raise ValueError("Fast tokenizer returned misaligned offsets")
        return ids, offsets

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
    *,
    learned_memory_path: str | Path | None = None,
    learned_memory_storage: Literal["jsonl", "sqlite"] = "jsonl",
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
    files = [manifest_path, inventory_path]
    if learned_memory_storage not in {"jsonl", "sqlite"}:
        raise ValueError("Learned-memory storage must be jsonl or sqlite")
    if learned_memory_path is not None:
        learned_source = Path(learned_memory_path)
        if learned_source.is_file():
            learned_filename = (
                LEARNED_DEFINITIONS_SQLITE_FILENAME
                if learned_memory_storage == "sqlite"
                else LEARNED_DEFINITIONS_FILENAME
            )
            learned_path = destination / learned_filename
            state_path = destination / LEARNED_STATE_FILENAME
            if learned_memory_storage == "sqlite":
                source_connection = sqlite3.connect(learned_source)
                destination_connection = sqlite3.connect(learned_path)
                try:
                    source_connection.backup(destination_connection)
                    destination_connection.commit()
                    destination_connection.execute("PRAGMA journal_mode=DELETE")
                    check = destination_connection.execute("PRAGMA quick_check").fetchone()
                    if check != ("ok",):
                        raise ValueError("SQLite learned-state snapshot failed integrity check")
                finally:
                    destination_connection.close()
                    source_connection.close()
            else:
                shutil.copyfile(learned_source, learned_path)
            payload = learned_path.read_bytes()
            state = {
                "schema_version": LEARNED_STATE_SCHEMA,
                "storage": learned_memory_storage,
                "filename": learned_filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
            }
            state_path.write_text(
                json.dumps(
                    state,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            files.extend((learned_path, state_path))
    return tuple(files)


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
    learned_path, learned_storage = _verify_learned_state(artifact)
    return BCNLTTokenizer(
        base,
        manifest,
        learned_memory_path=learned_path,
        learned_memory_storage=learned_storage,
        learned_memory_read_only=True,
    )


def _verify_inventory(path: Path, expected_sha256: str) -> None:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot load tokenizer structural inventory: {exc}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Tokenizer artifact structural inventory hash mismatch")


def _verify_learned_state(
    artifact: Path,
) -> tuple[Path, Literal["jsonl", "sqlite"]]:
    jsonl_path = artifact / LEARNED_DEFINITIONS_FILENAME
    sqlite_path = artifact / LEARNED_DEFINITIONS_SQLITE_FILENAME
    state_path = artifact / LEARNED_STATE_FILENAME
    if not jsonl_path.exists() and not sqlite_path.exists() and not state_path.exists():
        return jsonl_path, "jsonl"
    if not state_path.is_file():
        raise ValueError("Tokenizer artifact learned semantic state is incomplete")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read tokenizer learned semantic state: {exc}") from exc
    storage = state.get("storage") if isinstance(state, dict) else None
    expected_filename = {
        "jsonl": LEARNED_DEFINITIONS_FILENAME,
        "sqlite": LEARNED_DEFINITIONS_SQLITE_FILENAME,
    }.get(storage)
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != LEARNED_STATE_SCHEMA
        or expected_filename is None
        or state.get("filename") != expected_filename
        or not isinstance(state.get("sha256"), str)
        or not isinstance(state.get("byte_count"), int)
        or isinstance(state.get("byte_count"), bool)
    ):
        raise ValueError("Tokenizer artifact learned semantic state metadata is invalid")
    learned_path = artifact / expected_filename
    if not learned_path.is_file():
        raise ValueError("Tokenizer artifact learned semantic state is incomplete")
    payload = learned_path.read_bytes()
    if (
        len(payload) != state["byte_count"]
        or hashlib.sha256(payload).hexdigest() != state["sha256"]
    ):
        raise ValueError("Tokenizer artifact learned semantic state hash mismatch")
    if storage == "sqlite":
        connection = sqlite3.connect(
            f"{learned_path.resolve().as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        try:
            if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("Tokenizer artifact SQLite learned state is invalid")
        finally:
            connection.close()
    return learned_path, storage
