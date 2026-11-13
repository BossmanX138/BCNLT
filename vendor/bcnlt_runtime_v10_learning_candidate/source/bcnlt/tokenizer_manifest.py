from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .events import CURRENT_EVENT_VALUES
from .tokenizer_base import BaseTokenizer


SCHEMA_VERSION = "bcnlt-tokenizer-v1"
CANONICALIZATION_VERSION = "bcnlt-canonical-record-v1"
INVENTORY_SCHEMA_VERSION = "bcnlt-structural-tokens-v1"
DEFAULT_INVENTORY_PATH = Path(__file__).with_name("vocab") / "bcnlt-tokenizer-v1.tokens.json"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_INVENTORY_BYTES = 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class TokenizerFileHash:
    path: str
    sha256: str


@dataclass(frozen=True)
class BaseTokenizerIdentity:
    name: str
    revision: str
    files: tuple[TokenizerFileHash, ...]

    @classmethod
    def from_files(
        cls,
        *,
        name: str,
        revision: str,
        files: Mapping[str, bytes],
    ) -> "BaseTokenizerIdentity":
        if not name.strip() or not revision.strip() or not files:
            raise ValueError("Base tokenizer identity requires name, revision, and source files")
        hashed = []
        for path, content in files.items():
            normalized = _validate_relative_path(path)
            if not isinstance(content, bytes):
                raise ValueError(f"Tokenizer source file {path} must be bytes")
            hashed.append(TokenizerFileHash(normalized, _sha256(content)))
        return cls(name=name, revision=revision, files=tuple(sorted(hashed, key=lambda item: item.path)))

    @property
    def sha256(self) -> str:
        return _canonical_hash(_identity_payload(self))


@dataclass(frozen=True)
class AddedToken:
    token: str
    id: int


@dataclass(frozen=True)
class TokenizerManifest:
    schema_version: str
    canonicalization_version: str
    base_tokenizer: BaseTokenizerIdentity
    base_tokenizer_sha256: str
    base_vocab_size: int
    base_vocabulary_sha256: str
    structural_inventory_sha256: str
    added_tokens: tuple[AddedToken, ...]
    manifest_sha256: str

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.added_tokens)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {item.token: item.id for item in self.added_tokens}

    @property
    def id_to_token(self) -> dict[int, str]:
        return {item.id: item.token for item in self.added_tokens}

    def to_dict(self) -> dict[str, Any]:
        return _manifest_payload(self, include_digest=True)


def build_manifest(
    base: BaseTokenizer,
    identity: BaseTokenizerIdentity,
    inventory_path: str | Path = DEFAULT_INVENTORY_PATH,
) -> TokenizerManifest:
    inventory_file = Path(inventory_path)
    tokens = _load_inventory(inventory_file)
    vocab = _validated_base_vocab(base)
    collisions = sorted(set(tokens).intersection(vocab))
    if collisions:
        raise ValueError(f"BCNLT structural token collides with base vocabulary: {collisions[0]}")
    added = tuple(
        AddedToken(token=token, id=base.vocab_size + offset)
        for offset, token in enumerate(tokens)
    )
    provisional = TokenizerManifest(
        schema_version=SCHEMA_VERSION,
        canonicalization_version=CANONICALIZATION_VERSION,
        base_tokenizer=identity,
        base_tokenizer_sha256=identity.sha256,
        base_vocab_size=base.vocab_size,
        base_vocabulary_sha256=_vocabulary_hash(vocab),
        structural_inventory_sha256=_sha256(inventory_file.read_bytes()),
        added_tokens=added,
        manifest_sha256="",
    )
    digest = _canonical_hash(_manifest_payload(provisional, include_digest=False))
    return TokenizerManifest(**{**provisional.__dict__, "manifest_sha256": digest})


def write_manifest(path: str | Path, manifest: TokenizerManifest) -> None:
    destination = Path(path)
    _validate_manifest(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _canonical_json(manifest.to_dict()) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_manifest(
    path: str | Path,
    base: BaseTokenizer,
    identity: BaseTokenizerIdentity,
) -> TokenizerManifest:
    if identity is None:
        raise ValueError("Loading a tokenizer manifest requires a verified base identity")
    source = Path(path)
    try:
        if source.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("tokenizer manifest exceeds the bounded file limit")
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot load tokenizer manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Tokenizer manifest must be a JSON object")
    required_fields = {
        "schema_version",
        "canonicalization_version",
        "base_tokenizer",
        "base_tokenizer_sha256",
        "base_vocab_size",
        "base_vocabulary_sha256",
        "structural_inventory_sha256",
        "added_tokens",
        "manifest_sha256",
    }
    if set(payload) != required_fields:
        raise ValueError("Tokenizer manifest fields do not match the v1 schema")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported tokenizer manifest schema: {payload['schema_version']}")
    supplied_digest = payload["manifest_sha256"]
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual_digest = _canonical_hash(body)
    if supplied_digest != actual_digest:
        raise ValueError("Tokenizer manifest digest mismatch")
    manifest = _manifest_from_payload(payload)
    _validate_manifest(manifest)
    validate_manifest_for_base(manifest, base)
    if identity.sha256 != manifest.base_tokenizer_sha256:
        raise ValueError("Tokenizer manifest base tokenizer source hash mismatch")
    expected_tokens = _load_inventory(DEFAULT_INVENTORY_PATH)
    if tuple(item.token for item in manifest.added_tokens) != expected_tokens:
        raise ValueError("Tokenizer manifest is missing required structural tokens or has changed order")
    if _sha256(DEFAULT_INVENTORY_PATH.read_bytes()) != manifest.structural_inventory_sha256:
        raise ValueError("Tokenizer structural inventory hash mismatch")
    canonical = _canonical_json(manifest.to_dict()) + "\n"
    if source.read_text(encoding="utf-8") != canonical:
        raise ValueError("Tokenizer manifest is not in canonical form")
    return manifest


def validate_manifest_for_base(manifest: TokenizerManifest, base: BaseTokenizer) -> None:
    _validate_manifest(manifest)
    vocab = _validated_base_vocab(base)
    if (
        base.vocab_size != manifest.base_vocab_size
        or _vocabulary_hash(vocab) != manifest.base_vocabulary_sha256
    ):
        raise ValueError("Tokenizer manifest base vocabulary mismatch")
    collisions = sorted(set(manifest.token_to_id).intersection(vocab))
    if collisions:
        raise ValueError(f"BCNLT structural token collides with base vocabulary: {collisions[0]}")


def _load_inventory(path: Path) -> tuple[str, ...]:
    try:
        if path.stat().st_size > MAX_INVENTORY_BYTES:
            raise ValueError("structural-token inventory exceeds the bounded file limit")
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot load structural-token inventory: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "tokens"}:
        raise ValueError("Structural-token inventory has invalid fields")
    if payload["schema_version"] != INVENTORY_SCHEMA_VERSION:
        raise ValueError("Unsupported structural-token inventory schema")
    tokens = payload["tokens"]
    if not isinstance(tokens, list) or not tokens or any(not isinstance(item, str) for item in tokens):
        raise ValueError("Structural-token inventory tokens must be a non-empty string list")
    if len(tokens) != len(set(tokens)):
        raise ValueError("Structural-token inventory contains duplicate tokens")
    if any(not token.startswith("<|bcnlt:") or not token.endswith("|>") for token in tokens):
        raise ValueError("Structural tokens must use the reserved <|bcnlt:...|> namespace")
    event_tokens = {token for token in tokens if token.startswith("<|bcnlt:event:")}
    expected_events = {f"<|bcnlt:event:{event.lower()}|>" for event in CURRENT_EVENT_VALUES}
    if event_tokens != expected_events:
        raise ValueError("Structural-token inventory does not cover the semantic event contract")
    return tuple(tokens)


def _validated_base_vocab(base: BaseTokenizer) -> dict[str, int]:
    vocab = base.get_vocab()
    if type(base.vocab_size) is not int or base.vocab_size <= 0:
        raise ValueError("Base tokenizer vocabulary size must be positive")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Base tokenizer vocabulary must be a non-empty mapping")
    if any(
        type(token) is not str or type(token_id) is not int
        for token, token_id in vocab.items()
    ):
        raise ValueError("Base vocabulary entries must map strings to integer IDs")
    ids = list(vocab.values())
    if len(ids) != len(set(ids)):
        raise ValueError("Base vocabulary contains duplicate IDs")
    if set(ids) != set(range(base.vocab_size)):
        raise ValueError("Base vocabulary IDs must be contiguous and match vocab_size")
    return vocab


def _validate_manifest(manifest: TokenizerManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported tokenizer manifest schema: {manifest.schema_version}")
    if manifest.canonicalization_version != CANONICALIZATION_VERSION:
        raise ValueError("Unsupported tokenizer canonicalization version")
    if manifest.base_tokenizer.sha256 != manifest.base_tokenizer_sha256:
        raise ValueError("Base tokenizer identity digest mismatch")
    if type(manifest.base_vocab_size) is not int or manifest.base_vocab_size <= 0:
        raise ValueError("Tokenizer manifest base vocabulary size must be a positive integer")
    if any(
        type(item.token) is not str or not item.token or type(item.id) is not int
        for item in manifest.added_tokens
    ):
        raise ValueError("Tokenizer manifest added token entries have invalid types")
    tokens = [item.token for item in manifest.added_tokens]
    ids = [item.id for item in manifest.added_tokens]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Tokenizer manifest contains duplicate added tokens")
    if len(ids) != len(set(ids)):
        raise ValueError("Tokenizer manifest contains duplicate added token IDs")
    expected_ids = list(range(manifest.base_vocab_size, manifest.base_vocab_size + len(ids)))
    if ids != expected_ids:
        raise ValueError("Tokenizer manifest added IDs must be contiguous")
    expected_digest = _canonical_hash(_manifest_payload(manifest, include_digest=False))
    if manifest.manifest_sha256 != expected_digest:
        raise ValueError("Tokenizer manifest digest mismatch")


def _manifest_from_payload(payload: dict[str, Any]) -> TokenizerManifest:
    base = payload["base_tokenizer"]
    if not isinstance(base, dict) or set(base) != {"name", "revision", "files"}:
        raise ValueError("base_tokenizer has invalid fields")
    files = base["files"]
    if not isinstance(files, list):
        raise ValueError("base_tokenizer.files must be a list")
    file_hashes = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("base_tokenizer file entry is invalid")
        file_hashes.append(TokenizerFileHash(str(item["path"]), str(item["sha256"])))
    added_raw = payload["added_tokens"]
    if not isinstance(added_raw, list):
        raise ValueError("added_tokens must be a list")
    added = []
    for item in added_raw:
        if not isinstance(item, dict) or set(item) != {"token", "id"}:
            raise ValueError("added token entry is invalid")
        if type(item["token"]) is not str or type(item["id"]) is not int:
            raise ValueError("added token entries require string tokens and integer IDs")
        added.append(AddedToken(item["token"], item["id"]))
    identity = BaseTokenizerIdentity(str(base["name"]), str(base["revision"]), tuple(file_hashes))
    return TokenizerManifest(
        schema_version=str(payload["schema_version"]),
        canonicalization_version=str(payload["canonicalization_version"]),
        base_tokenizer=identity,
        base_tokenizer_sha256=str(payload["base_tokenizer_sha256"]),
        base_vocab_size=int(payload["base_vocab_size"]),
        base_vocabulary_sha256=str(payload["base_vocabulary_sha256"]),
        structural_inventory_sha256=str(payload["structural_inventory_sha256"]),
        added_tokens=tuple(added),
        manifest_sha256=str(payload["manifest_sha256"]),
    )


def _manifest_payload(manifest: TokenizerManifest, *, include_digest: bool) -> dict[str, Any]:
    payload = {
        "schema_version": manifest.schema_version,
        "canonicalization_version": manifest.canonicalization_version,
        "base_tokenizer": _identity_payload(manifest.base_tokenizer),
        "base_tokenizer_sha256": manifest.base_tokenizer_sha256,
        "base_vocab_size": manifest.base_vocab_size,
        "base_vocabulary_sha256": manifest.base_vocabulary_sha256,
        "structural_inventory_sha256": manifest.structural_inventory_sha256,
        "added_tokens": [
            {"token": item.token, "id": item.id} for item in manifest.added_tokens
        ],
    }
    if include_digest:
        payload["manifest_sha256"] = manifest.manifest_sha256
    return payload


def _identity_payload(identity: BaseTokenizerIdentity) -> dict[str, Any]:
    return {
        "name": identity.name,
        "revision": identity.revision,
        "files": [{"path": item.path, "sha256": item.sha256} for item in identity.files],
    }


def _vocabulary_hash(vocab: dict[str, int]) -> str:
    return _canonical_hash(vocab)


def _canonical_hash(payload: Any) -> str:
    return _sha256(_canonical_json(payload).encode("utf-8"))


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_relative_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or not path.strip():
        raise ValueError(f"Tokenizer source file path must be relative: {path}")
    return candidate.as_posix()
