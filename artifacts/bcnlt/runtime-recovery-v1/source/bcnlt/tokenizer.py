from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from .adapters.carousel import BCNLTMultilingualAdapter
from .adapters.python import PythonAdapter
from .adapters.tree_sitter import TreeSitterSemanticEngine
from .canonical import canonical_record_bytes, record_from_canonical_bytes
from .extended_core import ExtendedCoreLibrary
from .semantic_record import SemanticTokenRecord, records_from_assessments
from .tokenizer_base import BaseTokenizer
from .tokenizer_manifest import TokenizerManifest, validate_manifest_for_base
from .wildcard import WildcardMemory


_FIELD_ORDER = (
    "surface",
    "scope",
    "symbol_id",
    "defines",
    "span",
    "ast_relation",
    "ir",
    "recent_history",
    "metadata",
    "confidence",
    "uncertainty",
)
_OPTIONAL_STRING_FIELDS = {"symbol_id", "defines", "ast_relation"}
_JSON_FIELDS = {"span", "ir", "recent_history", "metadata"}
_SCORE_FIELDS = {"confidence", "uncertainty"}


class BCNLTTokenizer:
    def __init__(
        self,
        base: BaseTokenizer,
        manifest: TokenizerManifest,
        *,
        learned_memory_path: str | Path | None = None,
        learned_memory_storage: Literal["jsonl", "sqlite"] = "jsonl",
        learned_memory_read_only: bool = False,
    ) -> None:
        validate_manifest_for_base(manifest, base)
        self.base = base
        self.manifest = manifest
        self.learned_memory_path = (
            Path(learned_memory_path) if learned_memory_path is not None else None
        )
        if learned_memory_storage not in {"jsonl", "sqlite"}:
            raise ValueError("Learned-memory storage must be jsonl or sqlite")
        self.learned_memory_storage = learned_memory_storage
        self.learned_memory_read_only = learned_memory_read_only
        self._token_to_id = manifest.token_to_id
        self._id_to_token = manifest.id_to_token
        self._semantic_adapter = BCNLTMultilingualAdapter(
            engine=TreeSitterSemanticEngine(
                wildcard_memory_path=self.learned_memory_path,
                wildcard_memory_storage=self.learned_memory_storage,
                wildcard_memory_read_only=self.learned_memory_read_only,
            )
        )
        self._python_adapter = PythonAdapter(
            wildcard_memory_path=self.learned_memory_path,
            wildcard_memory_storage=self.learned_memory_storage,
            wildcard_memory_read_only=self.learned_memory_read_only,
        )

    def encode_text(self, text: str) -> list[int]:
        return self._encode_base_mode("text", text)

    def reload_learned_state(self) -> None:
        """Reload an externally updated JSONL learning ledger before future encodes."""

        self._semantic_adapter = BCNLTMultilingualAdapter(
            engine=TreeSitterSemanticEngine(
                wildcard_memory_path=self.learned_memory_path,
                wildcard_memory_storage=self.learned_memory_storage,
                wildcard_memory_read_only=self.learned_memory_read_only,
            )
        )
        self._python_adapter = PythonAdapter(
            wildcard_memory_path=self.learned_memory_path,
            wildcard_memory_storage=self.learned_memory_storage,
            wildcard_memory_read_only=self.learned_memory_read_only,
        )

    def extended_core_library(self) -> ExtendedCoreLibrary:
        """Return validated learned meanings without modifying Common Core IDs."""

        if self.learned_memory_path is None or not self.learned_memory_path.is_file():
            return ExtendedCoreLibrary()
        memory_kwargs: dict[str, Any] = {"storage": self.learned_memory_storage}
        if self.learned_memory_storage == "sqlite":
            with sqlite3.connect(self.learned_memory_path) as connection:
                metadata = dict(
                    connection.execute("SELECT key, value FROM wildcard_metadata")
                )
            memory_kwargs.update(
                language=metadata.get("language", "python"),
                semantic_spec_sha256=metadata.get(
                    "semantic_spec_sha256", "legacy-python"
                ),
            )
        memory = WildcardMemory(
            self.learned_memory_path,
            read_only=True,
            **memory_kwargs,
        )
        try:
            return memory.validated_extended_core()
        finally:
            memory.close()

    def save_pretrained(self, path: str | Path) -> tuple[Path, ...]:
        from .hf_tokenizer import save_tokenizer_artifact

        return save_tokenizer_artifact(
            path,
            self.manifest,
            learned_memory_path=self.learned_memory_path,
            learned_memory_storage=self.learned_memory_storage,
        )

    @classmethod
    def from_pretrained(
        cls, path: str | Path, base_path: str | Path
    ) -> "BCNLTTokenizer":
        from .hf_tokenizer import load_tokenizer_artifact

        return load_tokenizer_artifact(path, base_path)

    def encode_raw_code(self, source: str) -> list[int]:
        return self._encode_base_mode("raw_code", source)

    def encode_records(self, records: Iterable[SemanticTokenRecord]) -> list[int]:
        result = [self._id("stream:begin"), self._id("mode:semantic_code")]
        for record in records:
            payload = json.loads(canonical_record_bytes(record))
            result.append(self._id("record:begin"))
            result.append(self._id(f"event:{record.event.lower()}"))
            for field in _FIELD_ORDER:
                value = payload[f"{field}_band"] if field in _SCORE_FIELDS else payload[field]
                result.extend(self._encode_field(field, value))
            result.append(self._id("record:end"))
        result.append(self._id("stream:end"))
        return result

    def encode_source(
        self,
        source: str,
        path: str = "<memory>",
        *,
        language: str = "python",
        dialect: str | None = None,
    ) -> list[int]:
        if language == "python" and dialect in {None, "python3"}:
            document = self._python_adapter.parse(source, path=path)
            assessments = self._python_adapter.extract(document).assessments
        else:
            document = self._semantic_adapter.parse(
                source, language=language, path=path, dialect=dialect
            )
            assessments = self._semantic_adapter.extract(document).assessments
        records = records_from_assessments(assessments)
        return self.encode_records(records)

    def decode(self, ids: Sequence[int]) -> str:
        mode = self._mode(ids)
        if mode in {"text", "raw_code"}:
            return self._decode_base_body(ids)
        return b"".join(canonical_record_bytes(record) for record in self.decode_records(ids)).decode(
            "utf-8"
        )

    def decode_records(self, ids: Sequence[int]) -> tuple[SemanticTokenRecord, ...]:
        if self._mode(ids) != "semantic_code":
            raise ValueError("decode_records requires the semantic_code mode")
        cursor = 2
        stream_end = len(ids) - 1
        records: list[SemanticTokenRecord] = []
        while cursor < stream_end:
            cursor = self._expect(ids, cursor, self._id("record:begin"), "record begin")
            if cursor >= stream_end:
                raise ValueError("Semantic stream is truncated before its event token")
            event_token = self._structural_token(ids[cursor])
            if not event_token.startswith("<|bcnlt:event:"):
                raise ValueError(f"Expected semantic event token, got {event_token}")
            event = event_token.removeprefix("<|bcnlt:event:").removesuffix("|>").upper()
            cursor += 1
            payload: dict[str, Any] = {"event": event}
            for field in _FIELD_ORDER:
                value, cursor = self._decode_field(ids, cursor, field, stream_end)
                payload[f"{field}_band" if field in _SCORE_FIELDS else field] = value
            cursor = self._expect(ids, cursor, self._id("record:end"), "record end")
            canonical = (
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            records.append(record_from_canonical_bytes(canonical))
        return tuple(records)

    def _encode_base_mode(self, mode: str, text: str) -> list[int]:
        payload = self.base.encode(text, add_special_tokens=False)
        self._validate_base_payload(payload)
        return [
            self._id("stream:begin"),
            self._id(f"mode:{mode}"),
            *payload,
            self._id("stream:end"),
        ]

    def _encode_field(self, field: str, value: Any) -> list[int]:
        result = [self._id(f"field:{field}:begin")]
        if value is None:
            result.append(self._id("value:null"))
        elif field in _SCORE_FIELDS:
            result.append(self._id(f"value:{value}"))
        else:
            text = (
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if field in _JSON_FIELDS
                else value
            )
            if not isinstance(text, str):
                raise ValueError(f"{field} payload must serialize to text")
            payload = self.base.encode(text, add_special_tokens=False)
            self._validate_base_payload(payload)
            result.extend(payload)
        result.append(self._id(f"field:{field}:end"))
        return result

    def _decode_field(
        self,
        ids: Sequence[int],
        cursor: int,
        field: str,
        stream_end: int,
    ) -> tuple[Any, int]:
        begin = self._id(f"field:{field}:begin")
        if cursor >= stream_end or ids[cursor] != begin:
            actual = self._describe_id(ids[cursor]) if cursor < stream_end else "end of stream"
            raise ValueError(
                f"Expected {field} field; found duplicated or out-of-order token {actual}"
            )
        cursor += 1
        end = self._id(f"field:{field}:end")
        payload_ids: list[int] = []
        while cursor < stream_end and ids[cursor] != end:
            token_id = ids[cursor]
            if token_id >= self.manifest.base_vocab_size:
                token = self._structural_token(token_id)
                if not payload_ids and token == "<|bcnlt:value:null|>":
                    payload_ids.append(token_id)
                    cursor += 1
                    continue
                if not payload_ids and field in _SCORE_FIELDS and token in {
                    "<|bcnlt:value:low|>",
                    "<|bcnlt:value:medium|>",
                    "<|bcnlt:value:high|>",
                }:
                    payload_ids.append(token_id)
                    cursor += 1
                    continue
                raise ValueError(
                    f"Truncated {field} field or duplicated or out-of-order structural token {token}"
                )
            payload_ids.append(token_id)
            cursor += 1
        if cursor >= stream_end:
            raise ValueError(f"Truncated {field} field")
        cursor += 1
        if payload_ids == [self._id("value:null")]:
            if field not in _OPTIONAL_STRING_FIELDS and field != "ir":
                raise ValueError(f"{field} field cannot be null")
            return None, cursor
        if field in _SCORE_FIELDS:
            if len(payload_ids) != 1:
                raise ValueError(f"{field} field must contain one score-band token")
            token = self._structural_token(payload_ids[0])
            return token.removeprefix("<|bcnlt:value:").removesuffix("|>"), cursor
        self._validate_base_payload(payload_ids)
        try:
            text = self.base.decode(payload_ids)
        except Exception as exc:
            raise ValueError(f"Cannot decode {field} base-token payload: {exc}") from exc
        if field in _JSON_FIELDS:
            try:
                return json.loads(text), cursor
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} payload is not canonical JSON: {exc}") from exc
        return text, cursor

    def _mode(self, ids: Sequence[int]) -> str:
        self._validate_ids(ids)
        if len(ids) < 3:
            raise ValueError("BCNLT stream is truncated")
        if ids[0] != self._id("stream:begin") or ids[-1] != self._id("stream:end"):
            raise ValueError("BCNLT stream boundaries are missing or out of order")
        modes = {
            self._id("mode:text"): "text",
            self._id("mode:raw_code"): "raw_code",
            self._id("mode:semantic_code"): "semantic_code",
        }
        if ids[1] not in modes:
            raise ValueError(f"Unexpected BCNLT mode marker: {self._describe_id(ids[1])}")
        return modes[ids[1]]

    def _decode_base_body(self, ids: Sequence[int]) -> str:
        payload = list(ids[2:-1])
        self._validate_base_payload(payload)
        try:
            return self.base.decode(payload)
        except Exception as exc:
            raise ValueError(f"Cannot decode base-token payload: {exc}") from exc

    def _validate_ids(self, ids: Sequence[int]) -> None:
        for token_id in ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError(f"Token ID must be an integer: {token_id!r}")
            if token_id < 0 or token_id >= self.manifest.vocab_size:
                raise ValueError(f"unknown token ID outside declared vocabulary: {token_id}")

    def _validate_base_payload(self, ids: Sequence[int]) -> None:
        for token_id in ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError(f"Base token ID must be an integer: {token_id!r}")
            if token_id < 0 or token_id >= self.manifest.base_vocab_size:
                raise ValueError(f"Base token ID outside declared base vocabulary: {token_id}")

    def _id(self, suffix: str) -> int:
        token = f"<|bcnlt:{suffix}|>"
        try:
            return self._token_to_id[token]
        except KeyError as exc:
            raise ValueError(f"Tokenizer manifest is missing required token {token}") from exc

    def _structural_token(self, token_id: int) -> str:
        try:
            return self._id_to_token[token_id]
        except KeyError as exc:
            raise ValueError(f"unknown token ID absent from manifest: {token_id}") from exc

    def _describe_id(self, token_id: int) -> str:
        if token_id < self.manifest.base_vocab_size:
            return f"base token {token_id}"
        return self._structural_token(token_id)

    @staticmethod
    def _expect(ids: Sequence[int], cursor: int, expected: int, label: str) -> int:
        if cursor >= len(ids) or ids[cursor] != expected:
            raise ValueError(f"Expected {label}")
        return cursor + 1
