from __future__ import annotations

from dataclasses import dataclass, field

from .core import Span


@dataclass(frozen=True)
class SymbolRecord:
    path: str
    symbol_id: str
    surface: str
    span: Span


@dataclass
class ProjectIndex:
    definitions: dict[str, SymbolRecord] = field(default_factory=dict)
    definitions_by_name: dict[str, list[SymbolRecord]] = field(default_factory=dict)
    references: dict[str, list[SymbolRecord]] = field(default_factory=dict)

    def define(self, path: str, symbol_id: str, surface: str, span: Span) -> SymbolRecord:
        record = SymbolRecord(path, symbol_id, surface, span)
        self.definitions[symbol_id] = record
        self.definitions_by_name.setdefault(surface, []).append(record)
        return record

    def reference(
        self, path: str, symbol_id: str | None, surface: str, span: Span
    ) -> SymbolRecord:
        key = symbol_id or f"unresolved:{surface}"
        record = SymbolRecord(path, key, surface, span)
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
