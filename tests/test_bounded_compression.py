from __future__ import annotations

import unittest
import zlib

from bosscrafts_model.compression import decompress_bounded


class BoundedCompressionTests(unittest.TestCase):
    def test_exact_stream_roundtrips_within_budget(self):
        raw = b"semantic-token-bytes" * 100
        self.assertEqual(
            decompress_bounded(
                zlib.compress(raw),
                maximum_output_bytes=len(raw),
            ),
            raw,
        )

    def test_output_bomb_truncation_and_trailing_streams_fail_closed(self):
        raw = b"x" * 100_000
        with self.assertRaisesRegex(ValueError, "budget"):
            decompress_bounded(
                zlib.compress(raw),
                maximum_output_bytes=1024,
            )
        with self.assertRaisesRegex(ValueError, "truncated"):
            decompress_bounded(
                zlib.compress(b"safe")[:-1],
                maximum_output_bytes=1024,
            )
        with self.assertRaisesRegex(ValueError, "trailing"):
            decompress_bounded(
                zlib.compress(b"safe") + zlib.compress(b"second"),
                maximum_output_bytes=1024,
            )
        with self.assertRaisesRegex(ValueError, "artifact limit"):
            decompress_bounded(
                zlib.compress(b"safe"),
                maximum_output_bytes=65 * 1024 * 1024,
            )


if __name__ == "__main__":
    unittest.main()
