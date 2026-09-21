from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegionInvalidator:
    """Tracks file regions that need reprocessing after edits."""

    _ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def mark_changed(self, path: str, start_line: int, end_line: int) -> None:
        self._ranges.setdefault(path, []).append((start_line, end_line))

    def changed_files(self) -> set[str]:
        return set(self._ranges)

    def changed_ranges(self, path: str) -> list[tuple[int, int]]:
        return list(self._ranges.get(path, []))

    def clear(self, path: str | None = None) -> None:
        if path is None:
            self._ranges.clear()
        else:
            self._ranges.pop(path, None)
