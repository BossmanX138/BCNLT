from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from bcnlt.core import Span


@dataclass(slots=True)
class _ScopeFrame:
    kind: str
    name: str
    identity: str
    definitions: dict[str, str] = field(default_factory=dict)


class LexicalSymbolTable:
    """Document-local lexical definitions with deterministic shadowing."""

    def __init__(self, *, language: str, path: str) -> None:
        self.language = language
        self.path = path
        self._frames = [_ScopeFrame("module", "module", "module")]

    @property
    def scope(self) -> str:
        return ".".join(frame.name for frame in self._frames)

    @property
    def depth(self) -> int:
        return len(self._frames)

    def enter_scope(self, kind: str, name: str, start_byte: int) -> str:
        identity = self._stable_id(f"scope|{self.scope}|{kind}|{name}|{start_byte}")
        self._frames.append(_ScopeFrame(kind, name, identity))
        return identity

    def exit_scope(self) -> None:
        if len(self._frames) == 1:
            raise ValueError("Cannot exit the module scope")
        self._frames.pop()

    def define(self, name: str, span: Span) -> str:
        identity = self._stable_id(f"symbol|{self.scope}|{name}|{span[0]}:{span[1]}")
        self._frames[-1].definitions[name] = identity
        return identity

    def resolve(self, name: str) -> str | None:
        for frame in reversed(self._frames):
            identity = frame.definitions.get(name)
            if identity is not None:
                return identity
        return None

    def _stable_id(self, payload: str) -> str:
        digest = hashlib.sha256(
            f"{self.language}\0{self.path}\0{payload}".encode("utf-8")
        ).hexdigest()[:20]
        return f"{self.language}:{digest}"
