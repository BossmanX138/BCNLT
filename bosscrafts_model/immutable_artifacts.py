"""No-clobber publication for small immutable artifact generations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def publish_new_files(
    files: tuple[tuple[Path, bytes], ...],
    *,
    artifact_label: str,
) -> None:
    """Stage, sync, and hard-link a complete new generation in caller order."""
    if not files:
        raise ValueError("immutable artifact publication requires at least one file")
    destinations = [path.resolve() for path, _ in files]
    if len(set(destinations)) != len(destinations):
        raise ValueError("immutable artifact destinations must be unique")
    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for destination, (_, content) in zip(destinations, files, strict=True):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_handle = tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            )
            temporary = Path(temporary_handle.name)
            try:
                temporary_handle.write(content)
                temporary_handle.flush()
                os.fsync(temporary_handle.fileno())
            finally:
                temporary_handle.close()
            staged.append((destination, temporary))
        for destination, temporary in staged:
            if destination.exists():
                raise ValueError(f"{artifact_label} output file already exists")
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise ValueError(
                    f"{artifact_label} output file already exists"
                ) from error
            published.append((destination, temporary))
    except BaseException:
        for destination, temporary in reversed(published):
            try:
                if destination.exists() and destination.samefile(temporary):
                    destination.unlink()
            except OSError:
                pass
        raise
    finally:
        for _, temporary in staged:
            if temporary.exists():
                temporary.unlink()
