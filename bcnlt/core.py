from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .ir import IRNode


Span = tuple[int, int, int, int]


@dataclass(frozen=True)
class SourceDocument:
    path: str
    source: str
    tree: Any
    index: Any


@dataclass(frozen=True)
class ContextualUnit:
    surface: str
    event: str
    scope: str
    span: Span
    symbol_id: str | None = None
    defines: str | None = None
    ast_relation: str | None = None
    ir: IRNode | None = None
    recent_history: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["span"] = {
            "start_line": self.span[0],
            "start_col": self.span[1],
            "end_line": self.span[2],
            "end_col": self.span[3],
        }
        if self.ir is not None:
            data["ir"] = self.ir.to_dict()
        return data
