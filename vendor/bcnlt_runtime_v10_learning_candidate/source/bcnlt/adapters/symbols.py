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
        identity, _ = self._define_in_frame(len(self._frames) - 1, name, span)
        return identity

    def define_in_enclosing(
        self,
        name: str,
        span: Span,
        *,
        kinds: frozenset[str],
    ) -> tuple[str, str]:
        """Define in the nearest lexical frame whose kind owns the binding."""
        for index in range(len(self._frames) - 1, -1, -1):
            if self._frames[index].kind in kinds:
                return self._define_in_frame(index, name, span)
        raise ValueError(f"No enclosing scope found for kinds: {sorted(kinds)}")

    def resolve_current(self, name: str) -> str | None:
        """Resolve only a binding already declared in the current lexical frame."""
        return self._frames[-1].definitions.get(name)

    def resolve(self, name: str) -> str | None:
        for index in range(len(self._frames) - 1, -1, -1):
            frame = self._frames[index]
            # Python class namespaces own methods but are not lexical closures.
            # Keep direct class-body lookup; nested functions/classes skip them.
            # Implicit __class__ cells and definition-time defaults are not modeled.
            if self.language == "python" and frame.kind == "class" and index != len(self._frames) - 1:
                continue
            identity = frame.definitions.get(name)
            if identity is not None:
                return identity
        return None

    def _define_in_frame(self, index: int, name: str, span: Span) -> tuple[str, str]:
        scope = ".".join(frame.name for frame in self._frames[: index + 1])
        identity = self._stable_id(f"symbol|{scope}|{name}|{span[0]}:{span[1]}")
        self._frames[index].definitions[name] = identity
        return identity, scope

    def _stable_id(self, payload: str) -> str:
        digest = hashlib.sha256(
            f"{self.language}\0{self.path}\0{payload}".encode("utf-8")
        ).hexdigest()[:20]
        return f"{self.language}:{digest}"
