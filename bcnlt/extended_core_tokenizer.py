from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from .canonical import canonical_record_bytes
from .extended_core import ExtendedCoreLibrary
from .extended_core_vocab import (
    ExtendedCoreToken,
    ExtendedCoreVocabularyManifest,
    validate_extended_core_vocab,
)
from .semantic_record import SemanticTokenRecord
from .tokenizer import BCNLTTokenizer


class ExtendedCoreBCNLTTokenizer:
    """BCNLT v1 plus append-only native identities for validated meanings."""

    def __init__(
        self,
        v1: BCNLTTokenizer,
        manifest: ExtendedCoreVocabularyManifest,
        *,
        library: ExtendedCoreLibrary | None = None,
    ) -> None:
        validate_extended_core_vocab(manifest, v1.manifest)
        active_library = library if library is not None else v1.extended_core_library()
        active_ids = {entry.extended_core_id for entry in active_library}
        manifest_active_ids = {
            token.extended_core_id for token in manifest.tokens if token.active
        }
        if active_ids != manifest_active_ids:
            raise ValueError(
                "active Extended Core vocabulary does not match the validated library"
            )
        active_tokens = {
            token.extended_core_id: token
            for token in manifest.tokens
            if token.active
        }
        for entry in active_library:
            token = active_tokens[entry.extended_core_id]
            identity_matches = (
                token.signature == entry.signature
                and token.revision == entry.revision
                and token.language == entry.language
                and token.semantic_spec_sha256 == entry.semantic_spec_sha256
                and token.validated_domain == entry.validated_domain
                and token.evidence_sha256 == entry.evidence_sha256
            )
            if not identity_matches:
                raise ValueError(
                    f"Extended Core token identity does not match validated evidence: "
                    f"{entry.extended_core_id}"
                )
        self.v1 = v1
        self.base = v1.base
        self.manifest = manifest
        self.library = active_library
        self._token_by_id = {token.id: token for token in manifest.tokens}
        self._token_by_meaning = {
            token.extended_core_id: token for token in manifest.tokens
        }
        self._active_by_meaning = {
            token.extended_core_id: token
            for token in manifest.tokens
            if token.active
        }

    def encode_text(self, text: str) -> list[int]:
        return self.v1.encode_text(text)

    def save_pretrained(self, path: str | Path) -> tuple[Path, ...]:
        from .extended_core_artifact import save_extended_core_tokenizer_artifact

        return save_extended_core_tokenizer_artifact(path, self)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        base_path: str | Path,
        *,
        library: ExtendedCoreLibrary | None = None,
    ) -> ExtendedCoreBCNLTTokenizer:
        from .extended_core_artifact import load_extended_core_tokenizer_artifact

        return load_extended_core_tokenizer_artifact(
            path,
            base_path,
            library=library,
        )

    def encode_raw_code(self, source: str) -> list[int]:
        return self.v1.encode_raw_code(source)

    def encode_records(self, records: Iterable[SemanticTokenRecord]) -> list[int]:
        materialized = tuple(records)
        v1_ids = self.v1.encode_records(materialized)
        record_begin = self.v1.manifest.token_to_id["<|bcnlt:record:begin|>"]
        record_end = self.v1.manifest.token_to_id["<|bcnlt:record:end|>"]
        output = list(v1_ids[:2])
        cursor = 2
        stream_end = len(v1_ids) - 1
        for record in materialized:
            if cursor >= stream_end or v1_ids[cursor] != record_begin:
                raise ValueError("v1 semantic stream and records are misaligned")
            output.append(v1_ids[cursor])
            cursor += 1
            if cursor >= stream_end:
                raise ValueError("v1 semantic stream is missing its event token")
            output.append(v1_ids[cursor])
            cursor += 1
            token = self._native_token_for_record(record)
            if token is not None:
                output.append(token.id)
            while cursor < stream_end and v1_ids[cursor] != record_end:
                output.append(v1_ids[cursor])
                cursor += 1
            if cursor >= stream_end:
                raise ValueError("v1 semantic stream is missing a record boundary")
            output.append(v1_ids[cursor])
            cursor += 1
        if cursor != stream_end:
            raise ValueError("v1 semantic stream contains an unexpected extra record")
        output.append(v1_ids[-1])
        return output

    def encode_source(
        self,
        source: str,
        path: str = "<memory>",
        *,
        language: str = "python",
        dialect: str | None = None,
    ) -> list[int]:
        v1_ids = self.v1.encode_source(
            source,
            path=path,
            language=language,
            dialect=dialect,
        )
        return self.encode_records(self.v1.decode_records(v1_ids))

    def decode_records(
        self, ids: Sequence[int]
    ) -> tuple[SemanticTokenRecord, ...]:
        v1_ids, native_tokens = self._strip_native_tokens(ids)
        records = self.v1.decode_records(v1_ids)
        if len(records) != len(native_tokens):
            raise ValueError("native Extended Core stream is misaligned")
        for record, observed in zip(records, native_tokens, strict=True):
            expected = self._native_token_for_record(record, active_only=False)
            if expected is None and observed is not None:
                raise ValueError("unexpected native Extended Core token")
            if expected is not None and observed is None:
                raise ValueError("missing native Extended Core token")
            if expected is not None and observed is not None and expected.id != observed.id:
                raise ValueError("native Extended Core token does not match record metadata")
        return records

    def decode(self, ids: Sequence[int]) -> str:
        semantic_mode = self.v1.manifest.token_to_id["<|bcnlt:mode:semantic_code|>"]
        if len(ids) >= 2 and ids[1] == semantic_mode:
            return b"".join(
                canonical_record_bytes(record) for record in self.decode_records(ids)
            ).decode("utf-8")
        if any(token_id >= self.manifest.v1_vocab_size for token_id in ids):
            raise ValueError("native Extended Core tokens are valid only in semantic mode")
        return self.v1.decode(ids)

    def _native_token_for_record(
        self,
        record: SemanticTokenRecord,
        *,
        active_only: bool = True,
    ) -> ExtendedCoreToken | None:
        metadata = dict(record.metadata)
        state = metadata.get("extended_core_state")
        meaning_id = metadata.get("extended_core_id")
        if state is None and meaning_id is None:
            return None
        if state != "validated" or not isinstance(meaning_id, str):
            raise ValueError("Extended Core metadata is incomplete")
        token = (
            self._active_by_meaning.get(meaning_id)
            if active_only
            else self._token_by_meaning.get(meaning_id)
        )
        if token is None:
            qualifier = "active " if active_only else ""
            raise ValueError(
                f"validated Extended Core meaning has no {qualifier}native token: {meaning_id}"
            )
        return token

    def _strip_native_tokens(
        self, ids: Sequence[int]
    ) -> tuple[list[int], list[ExtendedCoreToken | None]]:
        for token_id in ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError(f"Token ID must be an integer: {token_id!r}")
            if token_id < 0 or token_id >= self.manifest.vocab_size:
                raise ValueError(f"token ID outside native vocabulary: {token_id}")
        if len(ids) < 3:
            raise ValueError("BCNLT stream is truncated")
        record_begin = self.v1.manifest.token_to_id["<|bcnlt:record:begin|>"]
        record_end = self.v1.manifest.token_to_id["<|bcnlt:record:end|>"]
        output = list(ids[:2])
        observed: list[ExtendedCoreToken | None] = []
        cursor = 2
        stream_end = len(ids) - 1
        while cursor < stream_end:
            if ids[cursor] != record_begin:
                raise ValueError("Expected record begin before native Extended Core token")
            output.append(ids[cursor])
            cursor += 1
            if cursor >= stream_end or ids[cursor] >= self.manifest.v1_vocab_size:
                raise ValueError("Expected v1 semantic event before native Extended Core token")
            output.append(ids[cursor])
            cursor += 1
            native = None
            if cursor < stream_end and ids[cursor] >= self.manifest.v1_vocab_size:
                native = self._token_by_id.get(ids[cursor])
                if native is None:
                    raise ValueError("unknown native Extended Core token")
                cursor += 1
            observed.append(native)
            while cursor < stream_end and ids[cursor] != record_end:
                if ids[cursor] >= self.manifest.v1_vocab_size:
                    raise ValueError(
                        "native Extended Core token must immediately follow the event"
                    )
                output.append(ids[cursor])
                cursor += 1
            if cursor >= stream_end:
                raise ValueError("native Extended Core stream is missing record end")
            output.append(ids[cursor])
            cursor += 1
        output.append(ids[-1])
        return output, observed
