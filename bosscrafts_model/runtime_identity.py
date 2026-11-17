"""Deterministic identities for the contained BCNLT runtime."""
from __future__ import annotations

import hashlib
from pathlib import Path

PYTHON_SOURCE_IDENTITY_POLICY = "python-source-bytes-crlf-cr-to-lf-v1"


def identity_bytes(path: Path) -> bytes:
    """Return checkout-independent bytes for identity-bearing runtime files."""

    payload = path.read_bytes()
    if path.suffix.casefold() == ".py":
        return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(identity_bytes(path)).hexdigest()


def is_generated_python_bytecode(path: Path, root: Path) -> bool:
    """Return whether *path* is an ordinary interpreter cache below *root*."""

    relative = path.relative_to(root)
    return (
        "__pycache__" in relative.parts
        and path.suffix.casefold() in {".pyc", ".pyo"}
    )


def aggregate_sha256(
    root: Path,
    pattern: str,
    *,
    ignore_generated_python_bytecode: bool = False,
) -> tuple[str, int, int]:
    paths = sorted(
        (
            path
            for path in root.rglob(pattern)
            if path.is_file()
            and not (
                ignore_generated_python_bytecode
                and is_generated_python_bytecode(path, root)
            )
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(root).as_posix()
        payload = identity_bytes(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).hexdigest().encode("ascii"))
        digest.update(b"\n")
        total_bytes += len(payload)
    return digest.hexdigest(), len(paths), total_bytes
