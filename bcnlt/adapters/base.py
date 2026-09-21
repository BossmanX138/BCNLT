from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from bcnlt.core import ContextualUnit, SourceDocument
from bcnlt.ir import CommonCodingCore, IRNode


class LanguageAdapter(ABC):
    language: str

    def __init__(self) -> None:
        self.common_core = CommonCodingCore()

    @abstractmethod
    def parse(self, source: str, path: str = "<memory>") -> SourceDocument:
        raise NotImplementedError

    @abstractmethod
    def to_units(self, document: SourceDocument) -> Iterable[ContextualUnit]:
        raise NotImplementedError

    @abstractmethod
    def render(self, node: IRNode) -> str:
        raise NotImplementedError
