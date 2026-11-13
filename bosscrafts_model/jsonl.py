"""Strict JSONL record boundaries that preserve Unicode inside JSON strings."""
from __future__ import annotations


def split_jsonl_records(text: str) -> tuple[str, ...]:
    """Split only on the JSONL LF delimiter and reject empty interior records."""
    if type(text) is not str:
        raise TypeError("JSONL payload must be a string")
    if not text:
        return ()
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if any(line == "" for line in lines):
        raise ValueError("JSONL payload contains an empty record")
    return tuple(lines)
