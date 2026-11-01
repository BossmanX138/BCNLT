from __future__ import annotations

from dataclasses import dataclass, field

from .cache import SemanticCacheIdentity


@dataclass
class RegionInvalidator:
    """Tracks file regions that need reprocessing after edits."""

    _ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    _identities: dict[str, SemanticCacheIdentity] = field(default_factory=dict)

    def mark_changed(
        self,
        path: str,
        start_line: int,
        end_line: int,
        *,
        identity: SemanticCacheIdentity | None = None,
    ) -> None:
        full_range = (0, 2**31 - 1)
        ranges = self._ranges.setdefault(path, [])
        if identity is not None:
            prior = self._identities.get(path)
            if prior is not None and prior != identity:
                self._ranges[path] = [full_range]
                ranges = self._ranges[path]
            self._identities[path] = identity
        if ranges != [full_range]:
            ranges.append((start_line, end_line))

    def changed_files(self) -> set[str]:
        return set(self._ranges)

    def changed_ranges(self, path: str) -> list[tuple[int, int]]:
        return list(self._ranges.get(path, []))

    def identity(self, path: str) -> SemanticCacheIdentity | None:
        return self._identities.get(path)

    def clear(self, path: str | None = None) -> None:
        if path is None:
            self._ranges.clear()
            self._identities.clear()
        else:
            self._ranges.pop(path, None)
            self._identities.pop(path, None)
