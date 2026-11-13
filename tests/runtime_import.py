"""Select one contained BCNLT runtime for test-module collection."""
from __future__ import annotations

from pathlib import Path
import sys


def select_bcnlt_runtime(source: Path) -> None:
    resolved = str(source.resolve())
    for name in tuple(sys.modules):
        if name == "bcnlt" or name.startswith("bcnlt."):
            sys.modules.pop(name, None)
    sys.path[:] = [
        entry
        for entry in sys.path
        if "bcnlt_runtime_v" not in str(entry).replace("\\", "/").lower()
    ]
    sys.path.insert(0, resolved)
