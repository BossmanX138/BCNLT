from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IRKind(str, Enum):
    MODULE = "module"
    VALUE = "value"
    VARIABLE = "variable"
    FUNCTION = "function"
    PARAMETER = "parameter"
    CALL = "call"
    ASSIGNMENT = "assignment"
    STATE = "state"
    LOOP = "loop"
    BRANCH = "branch"
    COMPARISON = "comparison"
    DATA_STRUCTURE = "data_structure"
    IO = "io"
    ERROR = "error"
    RETURN = "return"
    OPERATOR = "operator"
    MEMBER_ACCESS = "member_access"
    IMPORT = "import"
    EXTENSION = "extension"


@dataclass(frozen=True)
class IRNode:
    kind: IRKind
    name: str | None = None
    value: Any = None
    children: tuple["IRNode", ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata,
        }


class CommonCodingCore:
    """Compact, language-neutral IR factory for shared programming concepts."""

    def module(self, children: list[IRNode]) -> IRNode:
        return IRNode(IRKind.MODULE, children=tuple(children))

    def literal(self, value: Any) -> IRNode:
        return IRNode(IRKind.VALUE, value=value)

    def variable(self, name: str) -> IRNode:
        return IRNode(IRKind.VARIABLE, name=name)

    def parameter(self, name: str) -> IRNode:
        return IRNode(IRKind.PARAMETER, name=name)

    def function(self, name: str, parameters: list[IRNode], body: list[IRNode]) -> IRNode:
        return IRNode(
            IRKind.FUNCTION,
            name=name,
            children=tuple(parameters + body),
            metadata={"parameter_count": len(parameters)},
        )

    def call(self, name: str, args: list[IRNode] | None = None) -> IRNode:
        return IRNode(IRKind.CALL, name=name, children=tuple(args or []))

    def assignment(self, target: str, value: IRNode | None = None) -> IRNode:
        return IRNode(
            IRKind.ASSIGNMENT,
            name=target,
            children=tuple(child for child in [value] if child is not None),
        )

    def branch(self, condition: IRNode | None = None) -> IRNode:
        return IRNode(IRKind.BRANCH, children=tuple(child for child in [condition] if child))

    def loop(self, target: str | None = None) -> IRNode:
        return IRNode(IRKind.LOOP, name=target)

    def operator(self, name: str) -> IRNode:
        return IRNode(IRKind.OPERATOR, name=name)

    def member_access(self, name: str) -> IRNode:
        return IRNode(IRKind.MEMBER_ACCESS, name=name)

    def import_name(self, name: str) -> IRNode:
        return IRNode(IRKind.IMPORT, name=name)

    def return_value(self, value: IRNode | None = None) -> IRNode:
        return IRNode(IRKind.RETURN, children=tuple(child for child in [value] if child))

    def extension(self, name: str, metadata: dict[str, Any] | None = None) -> IRNode:
        return IRNode(IRKind.EXTENSION, name=name, metadata=metadata or {})
