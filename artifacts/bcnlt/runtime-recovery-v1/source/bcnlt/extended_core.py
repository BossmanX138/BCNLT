from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Iterator

from .events import SemanticEvent
from .ir import COMMON_CODING_CONCEPTS, IRKind


@dataclass(frozen=True, slots=True)
class CommonCoreProjection:
    """Optional decomposition of an Extended Core meaning into universal primitives."""

    event: str
    ir_kind: str

    def __post_init__(self) -> None:
        event = SemanticEvent(self.event).value
        ir_kind = IRKind(self.ir_kind).value
        allowed = {(item.event, item.ir_kind.value) for item in COMMON_CODING_CONCEPTS}
        if (event, ir_kind) not in allowed:
            raise ValueError("Extended Core projection must name one Common Core concept")
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "ir_kind", ir_kind)

    def to_dict(self) -> dict[str, str]:
        return {"event": self.event, "ir_kind": self.ir_kind}


@dataclass(frozen=True, slots=True)
class ExtendedCoreEntry:
    """A validated learned meaning kept outside the immutable Common Core."""

    extended_core_id: str
    signature: str
    revision: int
    language: str
    semantic_spec_sha256: str
    surface: str
    relation: str
    category: str
    hypotheses: tuple[str, ...]
    evidence: tuple[str, ...]
    confidence: float
    validated_domain: str
    common_core_projection: CommonCoreProjection | None
    evidence_sha256: tuple[str, ...]
    extension_name: str | None = None


class ExtendedCoreLibrary:
    """Immutable, indexed view of currently validated Extended Core meanings."""

    def __init__(self, entries: Iterable[ExtendedCoreEntry] = ()) -> None:
        ordered = tuple(sorted(entries, key=lambda item: item.extended_core_id))
        by_id: dict[str, ExtendedCoreEntry] = {}
        by_signature: dict[str, ExtendedCoreEntry] = {}
        for entry in ordered:
            if entry.extended_core_id in by_id:
                raise ValueError(f"Duplicate Extended Core ID: {entry.extended_core_id}")
            if entry.signature in by_signature:
                raise ValueError(f"Duplicate Extended Core signature: {entry.signature}")
            by_id[entry.extended_core_id] = entry
            by_signature[entry.signature] = entry
        self._entries = ordered
        self._by_id = by_id
        self._by_signature = by_signature

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ExtendedCoreEntry]:
        return iter(self._entries)

    def by_id(self, extended_core_id: str) -> ExtendedCoreEntry | None:
        return self._by_id.get(extended_core_id)

    def by_signature(self, signature: str) -> ExtendedCoreEntry | None:
        return self._by_signature.get(signature)


def extended_core_id(*, language: str, signature: str, revision: int) -> str:
    if revision < 1:
        raise ValueError("Extended Core revision must be positive")
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24]
    normalized_language = language.strip().lower().replace("_", "-")
    if not normalized_language:
        raise ValueError("Extended Core language is required")
    return f"bcnlt.extended.{normalized_language}.{digest}.r{revision}"


def optional_common_core_projection(
    *, event: str, ir_kind: str
) -> CommonCoreProjection | None:
    """Build a decomposition when one exists; native extensions need none."""

    normalized_event = SemanticEvent(event).value
    normalized_ir_kind = IRKind(ir_kind).value
    if (
        normalized_event == SemanticEvent.UNKNOWN_SEMANTIC.value
        and normalized_ir_kind == IRKind.EXTENSION.value
    ):
        return None
    return CommonCoreProjection(event=normalized_event, ir_kind=normalized_ir_kind)
