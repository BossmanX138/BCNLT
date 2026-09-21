from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .adapters.python import PythonAdapter
from .core import ContextualUnit


def process_python_source(source: str, path: str = "<memory>") -> list[ContextualUnit]:
    adapter = PythonAdapter()
    document = adapter.parse(source, path=path)
    return list(adapter.to_units(document))


def process_python_files(paths: Iterable[str], workers: int = 1) -> dict[str, list[ContextualUnit]]:
    path_list = list(paths)

    def process(path: str) -> tuple[str, list[ContextualUnit]]:
        source = Path(path).read_text(encoding="utf-8")
        return path, process_python_source(source, path=path)

    if workers <= 1:
        return dict(process(path) for path in path_list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(process, path_list))
