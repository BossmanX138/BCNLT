"""Bounded decompression helpers for audited SQLite artifacts."""
from __future__ import annotations

import zlib

MAX_DECOMPRESSED_ARTIFACT_BYTES = 64 * 1024 * 1024


def decompress_bounded(payload: bytes, *, maximum_output_bytes: int) -> bytes:
    """Decompress one exact zlib stream without exceeding the declared budget."""
    if type(payload) is not bytes:
        raise TypeError("compressed payload must be bytes")
    if (
        type(maximum_output_bytes) is not int
        or not 0 <= maximum_output_bytes <= MAX_DECOMPRESSED_ARTIFACT_BYTES
    ):
        raise ValueError("maximum_output_bytes exceeds the audited artifact limit")
    decompressor = zlib.decompressobj()
    try:
        output = decompressor.decompress(payload, maximum_output_bytes + 1)
        if len(output) > maximum_output_bytes or decompressor.unconsumed_tail:
            raise ValueError("compressed payload exceeds its output budget")
        output += decompressor.flush(maximum_output_bytes + 1 - len(output))
    except zlib.error as error:
        raise ValueError("compressed payload is not a valid zlib stream") from error
    if len(output) > maximum_output_bytes:
        raise ValueError("compressed payload exceeds its output budget")
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("compressed payload is truncated or contains trailing data")
    return output
