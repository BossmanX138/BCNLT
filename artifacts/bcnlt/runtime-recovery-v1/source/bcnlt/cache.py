from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import Span


@dataclass(frozen=True)
class SymbolRecord:
    path: str
    symbol_id: str
    surface: str
    span: Span
    language: str = "python"


@dataclass(frozen=True)
class SemanticCacheIdentity:
    language: str
    dialect: str | None
    parser_bundle_sha256: str
    semantic_spec_sha256: str

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def parser_bundle_sha256() -> str:
    path = Path(__file__).parents[1] / "configs" / "bcnlt_parser_bundle_v1.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_cache_identity(
    adapter: Any, language: str, dialect: str | None = None
) -> SemanticCacheIdentity:
    cartridge = adapter.carousel.for_language(language)
    return SemanticCacheIdentity(
        language=language,
        dialect=dialect,
        parser_bundle_sha256=parser_bundle_sha256(),
        semantic_spec_sha256=cartridge.spec.canonical_hash,
    )


def semantic_cache_key(
    *, path: str, source_sha256: str, identity: SemanticCacheIdentity
) -> str:
    payload = {
        "path": path.replace("\\", "/"),
        "source_sha256": source_sha256,
        **identity.__dict__,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class ProjectIndex:
    language: str = "python"
    dialect: str | None = None
    parser_bundle_sha256: str = "legacy-python"
    semantic_spec_sha256: str = "legacy-python"
    definitions: dict[str, SymbolRecord] = field(default_factory=dict)
    definitions_by_name: dict[str, list[SymbolRecord]] = field(default_factory=dict)
    references: dict[str, list[SymbolRecord]] = field(default_factory=dict)

    def define(self, path: str, symbol_id: str, surface: str, span: Span) -> SymbolRecord:
        record = SymbolRecord(path, symbol_id, surface, span, self.language)
        self.definitions[symbol_id] = record
        self.definitions_by_name.setdefault(surface, []).append(record)
        return record

    def reference(
        self, path: str, symbol_id: str | None, surface: str, span: Span
    ) -> SymbolRecord:
        key = symbol_id or f"unresolved:{surface}"
        record = SymbolRecord(path, key, surface, span, self.language)
        self.references.setdefault(key, []).append(record)
        return record

    def find_definition(self, surface: str, scope: str | None = None) -> SymbolRecord | None:
        candidates = self.definitions_by_name.get(surface, [])
        if scope:
            scoped = [record for record in candidates if record.symbol_id.startswith(scope)]
            if scoped:
                return scoped[-1]
        return candidates[-1] if candidates else None

    def references_to(self, symbol_id: str) -> list[SymbolRecord]:
        return self.references.get(symbol_id, [])
