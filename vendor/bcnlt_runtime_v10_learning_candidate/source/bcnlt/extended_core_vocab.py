from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .extended_core import ExtendedCoreEntry, ExtendedCoreLibrary
from .tokenizer_manifest import TokenizerManifest


SCHEMA_VERSION = "bcnlt-extended-core-vocab-v1"


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ExtendedCoreToken:
    name: str
    id: int
    extended_core_id: str
    signature: str
    revision: int
    language: str
    semantic_spec_sha256: str
    definition: str
    validated_domain: str
    evidence_sha256: tuple[str, ...]
    active: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "id": self.id,
            "extended_core_id": self.extended_core_id,
            "signature": self.signature,
            "revision": self.revision,
            "language": self.language,
            "semantic_spec_sha256": self.semantic_spec_sha256,
            "definition": self.definition,
            "validated_domain": self.validated_domain,
            "evidence_sha256": list(self.evidence_sha256),
            "active": self.active,
        }


@dataclass(frozen=True, slots=True)
class ExtendedCoreVocabularyManifest:
    schema_version: str
    v1_manifest_sha256: str
    parent_manifest_sha256: str
    extended_core_snapshot_sha256: str
    base_vocab_size: int
    v1_vocab_size: int
    vocab_size: int
    tokens: tuple[ExtendedCoreToken, ...]
    manifest_sha256: str

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "v1_manifest_sha256": self.v1_manifest_sha256,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "extended_core_snapshot_sha256": self.extended_core_snapshot_sha256,
            "base_vocab_size": self.base_vocab_size,
            "v1_vocab_size": self.v1_vocab_size,
            "vocab_size": self.vocab_size,
            "tokens": [token.to_dict() for token in self.tokens],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "manifest_sha256": self.manifest_sha256}


def build_extended_core_vocab(
    v1_manifest: TokenizerManifest,
    library: ExtendedCoreLibrary,
    *,
    previous: ExtendedCoreVocabularyManifest | None = None,
) -> ExtendedCoreVocabularyManifest:
    if previous is not None:
        validate_extended_core_vocab(previous, v1_manifest)

    current = {entry.extended_core_id: entry for entry in library}
    tokens: list[ExtendedCoreToken] = []
    if previous is not None:
        for old in previous.tokens:
            entry = current.pop(old.extended_core_id, None)
            if entry is None:
                tokens.append(replace(old, active=False))
                continue
            candidate = _token_from_entry(entry, name=old.name, token_id=old.id)
            if _immutable_token_identity(candidate) != _immutable_token_identity(old):
                raise ValueError(
                    "an allocated Extended Core token changed without a new revision"
                )
            tokens.append(candidate)

    for entry in sorted(current.values(), key=lambda item: item.extended_core_id):
        index = len(tokens)
        tokens.append(
            _token_from_entry(
                entry,
                name=f"<|bcnlt:extended-core:{index:04d}|>",
                token_id=v1_manifest.vocab_size + index,
            )
        )

    active_snapshot = [
        token.to_dict() for token in tokens if token.active
    ]
    provisional = ExtendedCoreVocabularyManifest(
        schema_version=SCHEMA_VERSION,
        v1_manifest_sha256=v1_manifest.manifest_sha256,
        parent_manifest_sha256=(
            previous.manifest_sha256
            if previous is not None
            else v1_manifest.manifest_sha256
        ),
        extended_core_snapshot_sha256=_digest(active_snapshot),
        base_vocab_size=v1_manifest.base_vocab_size,
        v1_vocab_size=v1_manifest.vocab_size,
        vocab_size=v1_manifest.vocab_size + len(tokens),
        tokens=tuple(tokens),
        manifest_sha256="",
    )
    manifest = replace(provisional, manifest_sha256=_digest(provisional._body()))
    validate_extended_core_vocab(manifest, v1_manifest)
    return manifest


def validate_extended_core_vocab(
    manifest: ExtendedCoreVocabularyManifest,
    v1_manifest: TokenizerManifest,
) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported Extended Core vocabulary schema")
    if not _is_digest(manifest.parent_manifest_sha256):
        raise ValueError("Extended Core parent manifest requires a SHA-256 digest")
    if manifest.v1_manifest_sha256 != v1_manifest.manifest_sha256:
        raise ValueError("v1 manifest digest mismatch")
    if (
        manifest.base_vocab_size != v1_manifest.base_vocab_size
        or manifest.v1_vocab_size != v1_manifest.vocab_size
    ):
        raise ValueError("v1 vocabulary size mismatch")
    expected_ids = list(
        range(manifest.v1_vocab_size, manifest.v1_vocab_size + len(manifest.tokens))
    )
    if [token.id for token in manifest.tokens] != expected_ids:
        raise ValueError("Extended Core token IDs must append contiguously after v1")
    expected_names = [
        f"<|bcnlt:extended-core:{index:04d}|>"
        for index in range(len(manifest.tokens))
    ]
    if [token.name for token in manifest.tokens] != expected_names:
        raise ValueError("Extended Core token names must preserve allocation order")
    if len({token.extended_core_id for token in manifest.tokens}) != len(
        manifest.tokens
    ):
        raise ValueError("Extended Core IDs must be unique")
    if any(type(token.active) is not bool for token in manifest.tokens):
        raise ValueError("Extended Core token active state must be boolean")
    if any(
        not token.evidence_sha256
        or any(not _is_digest(item) for item in token.evidence_sha256)
        for token in manifest.tokens
    ):
        raise ValueError("Extended Core token evidence requires SHA-256 digests")
    if manifest.vocab_size != manifest.v1_vocab_size + len(manifest.tokens):
        raise ValueError("Extended Core vocabulary size mismatch")
    if manifest.extended_core_snapshot_sha256 != _digest(
        [token.to_dict() for token in manifest.tokens if token.active]
    ):
        raise ValueError("Extended Core snapshot digest mismatch")
    if manifest.manifest_sha256 != _digest(manifest._body()):
        raise ValueError("Extended Core manifest digest mismatch")


def write_extended_core_vocab(
    path: str | Path,
    manifest: ExtendedCoreVocabularyManifest,
    v1_manifest: TokenizerManifest,
) -> None:
    validate_extended_core_vocab(manifest, v1_manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical(manifest.to_dict()))


def load_extended_core_vocab(
    path: str | Path,
    v1_manifest: TokenizerManifest,
) -> ExtendedCoreVocabularyManifest:
    source = Path(path)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
        tokens = tuple(
            ExtendedCoreToken(
                name=str(item["name"]),
                id=int(item["id"]),
                extended_core_id=str(item["extended_core_id"]),
                signature=str(item["signature"]),
                revision=int(item["revision"]),
                language=str(item["language"]),
                semantic_spec_sha256=str(item["semantic_spec_sha256"]),
                definition=str(item["definition"]),
                validated_domain=str(item["validated_domain"]),
                evidence_sha256=tuple(item["evidence_sha256"]),
                active=item["active"],
            )
            for item in payload["tokens"]
        )
        manifest = ExtendedCoreVocabularyManifest(
            schema_version=str(payload["schema_version"]),
            v1_manifest_sha256=str(payload["v1_manifest_sha256"]),
            parent_manifest_sha256=str(payload["parent_manifest_sha256"]),
            extended_core_snapshot_sha256=str(
                payload["extended_core_snapshot_sha256"]
            ),
            base_vocab_size=int(payload["base_vocab_size"]),
            v1_vocab_size=int(payload["v1_vocab_size"]),
            vocab_size=int(payload["vocab_size"]),
            tokens=tokens,
            manifest_sha256=str(payload["manifest_sha256"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Extended Core manifest: {exc}") from exc
    validate_extended_core_vocab(manifest, v1_manifest)
    if raw != _canonical(manifest.to_dict()):
        raise ValueError("Extended Core manifest JSON is not canonical")
    return manifest


def _token_from_entry(
    entry: ExtendedCoreEntry,
    *,
    name: str,
    token_id: int,
) -> ExtendedCoreToken:
    projection = entry.common_core_projection
    projection_text = (
        f"; decomposes to {projection.event}/{projection.ir_kind}"
        if projection is not None
        else ""
    )
    label = entry.extension_name or entry.relation
    definition = (
        f"Extended Core meaning {label}: {entry.category}; "
        f"validated for {entry.validated_domain}{projection_text}"
    )
    return ExtendedCoreToken(
        name=name,
        id=token_id,
        extended_core_id=entry.extended_core_id,
        signature=entry.signature,
        revision=entry.revision,
        language=entry.language,
        semantic_spec_sha256=entry.semantic_spec_sha256,
        definition=definition,
        validated_domain=entry.validated_domain,
        evidence_sha256=entry.evidence_sha256,
        active=True,
    )


def _immutable_token_identity(token: ExtendedCoreToken) -> tuple[object, ...]:
    return (
        token.name,
        token.id,
        token.extended_core_id,
        token.signature,
        token.revision,
        token.language,
        token.semantic_spec_sha256,
        token.definition,
        token.validated_domain,
    )
