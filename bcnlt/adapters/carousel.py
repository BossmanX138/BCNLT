from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bcnlt.core import SourceDocument
from bcnlt.languages import CANONICAL_LANGUAGE_IDS, LANGUAGE_DEFINITIONS, LanguageDefinition

from .coverage import SemanticExtraction
from .spec import LanguageSemanticSpec


@dataclass(frozen=True, slots=True)
class LanguageCartridge:
    definition: LanguageDefinition
    spec: LanguageSemanticSpec

    def __post_init__(self) -> None:
        if self.definition.semantic_spec_version != self.spec.semantic_spec_version:
            raise ValueError("Language definition and semantic spec versions differ")


class SemanticEngine(Protocol):
    def parse(
        self,
        source: str,
        *,
        cartridge: LanguageCartridge,
        path: str,
        dialect: str | None,
    ) -> SourceDocument: ...

    def extract(
        self,
        document: SourceDocument,
        *,
        cartridge: LanguageCartridge,
    ) -> SemanticExtraction: ...


@dataclass(frozen=True, slots=True)
class LanguageCarousel:
    cartridges: tuple[LanguageCartridge, ...]

    def __post_init__(self) -> None:
        ids = tuple(item.definition.canonical_id for item in self.cartridges)
        if ids != CANONICAL_LANGUAGE_IDS:
            raise ValueError("Language carousel must contain all canonical languages in order")

    @classmethod
    def default(cls) -> "LanguageCarousel":
        from .python import build_python_spec

        return cls(
            tuple(
                LanguageCartridge(
                    definition=definition,
                    spec=(
                        build_python_spec()
                        if definition.canonical_id == "python"
                        else LanguageSemanticSpec(definition.semantic_spec_version)
                    ),
                )
                for definition in LANGUAGE_DEFINITIONS
            )
        )

    @property
    def language_ids(self) -> tuple[str, ...]:
        return tuple(item.definition.canonical_id for item in self.cartridges)

    def for_language(self, language: str) -> LanguageCartridge:
        for cartridge in self.cartridges:
            if cartridge.definition.canonical_id == language:
                return cartridge
        raise ValueError(f"Use an exact registered canonical language id, got: {language}")


class BCNLTMultilingualAdapter:
    def __init__(
        self,
        *,
        engine: SemanticEngine,
        carousel: LanguageCarousel | None = None,
    ) -> None:
        self.engine = engine
        self.carousel = carousel or LanguageCarousel.default()

    def parse(
        self,
        source: str,
        *,
        language: str,
        path: str,
        dialect: str | None = None,
    ) -> SourceDocument:
        cartridge = self.carousel.for_language(language)
        if dialect is not None and dialect not in cartridge.definition.dialects:
            raise ValueError(f"Unsupported {language} dialect: {dialect}")
        document = self.engine.parse(
            source,
            cartridge=cartridge,
            path=path,
            dialect=dialect,
        )
        if document.language != language or document.dialect != dialect:
            raise ValueError("Semantic engine changed explicit language metadata")
        return document

    def extract(self, document: SourceDocument) -> SemanticExtraction:
        cartridge = self.carousel.for_language(document.language)
        extraction = self.engine.extract(document, cartridge=cartridge)
        if extraction.language != document.language or extraction.dialect != document.dialect:
            raise ValueError("Semantic extraction language metadata mismatch")
        return extraction
