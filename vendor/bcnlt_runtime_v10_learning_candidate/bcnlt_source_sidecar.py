"""Isolated BCNLT multilingual source-to-canonical-record sidecar.

The default mode accepts one bounded JSON request on stdin. ``--worker`` mode
accepts newline-framed bounded JSON requests and emits one newline-framed JSON
response per request, allowing the verified parser runtime to be reused without
changing semantic output. Source text is parsed, never executed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "source"
SITE_PACKAGES = ROOT / "site-packages"
GRAMMARS = ROOT / "grammars"
sys.path.insert(0, str(SITE_PACKAGES))
sys.path.insert(0, str(SOURCE_ROOT))
os.environ["BCNLT_PARSER_CACHE"] = str(GRAMMARS)

MAX_SOURCE_BYTES = 200 * 1024
MAX_REQUEST_BYTES = MAX_SOURCE_BYTES * 6 + 16 * 1024
MAX_CANONICAL_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_CANONICAL_RECORDS = 16_384
MAX_RESPONSE_BYTES = MAX_CANONICAL_OUTPUT_BYTES * 6 + 64 * 1024
PROTOCOL = "bosscrafts-bcnlt-source-sidecar-v3"
SIDECAR_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__)
    .read_bytes()
    .replace(b"\r\n", b"\n")
    .replace(b"\r", b"\n")
).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_request(raw: bytes) -> dict[str, object]:
    try:
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("sidecar request exceeds the bounded transport limit")
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("sidecar request must be one valid JSON object") from error
    if type(payload) is not dict or set(payload) != {
        "protocol",
        "source",
        "language",
        "path",
        "dialect",
        "learning_mode",
        "learned_memory_root",
    }:
        raise ValueError("sidecar request keys mismatch")
    if payload["protocol"] != PROTOCOL:
        raise ValueError("sidecar protocol mismatch")
    if type(payload["source"]) is not str:
        raise ValueError("source must be a string")
    if len(payload["source"].encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source exceeds the BCNLT parser limit")
    if (
        type(payload["language"]) is not str
        or not payload["language"]
        or len(payload["language"]) > 64
    ):
        raise ValueError("language must be a bounded non-empty string")
    if (
        type(payload["path"]) is not str
        or not payload["path"]
        or len(payload["path"]) > 4096
        or len(payload["path"].encode("utf-8")) > 16 * 1024
    ):
        raise ValueError("path must be a bounded non-empty logical path")
    if payload["dialect"] is not None and (
        type(payload["dialect"]) is not str
        or len(payload["dialect"]) > 256
        or len(payload["dialect"].encode("utf-8")) > 1024
    ):
        raise ValueError("dialect must be null or a bounded string")
    learning_mode = payload["learning_mode"]
    learned_memory_root = payload["learned_memory_root"]
    if learning_mode not in {"ephemeral", "observe", "read_only"}:
        raise ValueError("learning mode is invalid")
    if learning_mode == "ephemeral":
        if learned_memory_root is not None:
            raise ValueError("ephemeral learning cannot receive a memory root")
    else:
        if (
            type(learned_memory_root) is not str
            or not learned_memory_root
            or len(learned_memory_root) > 4096
            or len(learned_memory_root.encode("utf-8")) > 16 * 1024
        ):
            raise ValueError("persistent learning requires a bounded memory root")
        memory_root = Path(learned_memory_root)
        if not memory_root.is_absolute() or memory_root == Path(memory_root.anchor):
            raise ValueError("learned-memory root must be absolute and non-root")
        if memory_root.exists() and not memory_root.is_dir():
            raise ValueError("learned-memory root must be a directory")
    return payload


def _read_request() -> dict[str, object]:
    return _parse_request(sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1))


def _run(payload: dict[str, object]) -> dict[str, object]:
    from bcnlt.adapters import BCNLTMultilingualAdapter, LanguageCarousel
    from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine
    from bcnlt.canonical import canonical_record_bytes
    from bcnlt.semantic_record import records_from_assessments
    from bcnlt.wildcard import WildcardMemory

    language = str(payload["language"])
    source = str(payload["source"])
    path = str(payload["path"])
    dialect = payload["dialect"]
    learning_mode = str(payload["learning_mode"])
    memory_root_value = payload["learned_memory_root"]
    memory_root = (
        Path(memory_root_value)
        if isinstance(memory_root_value, str)
        else None
    )
    carousel = LanguageCarousel.default()
    cartridge = carousel.for_language(language)
    adapter = BCNLTMultilingualAdapter(
        engine=TreeSitterSemanticEngine(
            wildcard_memory_root=memory_root,
            wildcard_memory_read_only=learning_mode == "read_only",
        ),
        carousel=carousel,
    )
    document = adapter.parse(
        source,
        language=language,
        path=path,
        dialect=dialect if isinstance(dialect, str) else None,
    )
    extraction = adapter.extract(document)
    if len(extraction.assessments) > MAX_CANONICAL_RECORDS:
        raise ValueError("semantic record count exceeds the bounded transport limit")
    records = records_from_assessments(extraction.assessments)
    canonical_bytes = bytearray()
    for record in records:
        encoded = canonical_record_bytes(record)
        if len(canonical_bytes) + len(encoded) > MAX_CANONICAL_OUTPUT_BYTES:
            raise ValueError("canonical semantic output exceeds the bounded transport limit")
        canonical_bytes.extend(encoded)
    canonical_jsonl = canonical_bytes.decode("utf-8")
    learned_memory_sha256 = None
    learned_definition_count = 0
    extended_core_entry_count = 0
    if memory_root is not None:
        memory_path = memory_root / f"{language}.jsonl"
        if memory_path.is_file():
            learned_memory_sha256 = hashlib.sha256(memory_path.read_bytes()).hexdigest()
            memory = WildcardMemory(
                memory_path,
                language=language,
                semantic_spec_sha256=cartridge.spec.canonical_hash,
                read_only=True,
            )
            learned_definition_count = len(memory.records)
            extended_core_entry_count = len(memory.validated_extended_core())
    return {
        "protocol": PROTOCOL,
        "sidecar_source_sha256": SIDECAR_SOURCE_SHA256,
        "language": language,
        "path": path,
        "dialect": dialect,
        "semantic_schema_version": "bcnlt-semantic-record-v3",
        "semantic_spec_sha256": cartridge.spec.canonical_hash,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "canonical_sha256": hashlib.sha256(canonical_jsonl.encode("utf-8")).hexdigest(),
        "record_count": len(records),
        "canonical_jsonl": canonical_jsonl,
        "learning_mode": learning_mode,
        "learned_memory_sha256": learned_memory_sha256,
        "learned_definition_count": learned_definition_count,
        "extended_core_entry_count": extended_core_entry_count,
    }


def _response_bytes(response: dict[str, object]) -> bytes:
    response_bytes = (
        json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if len(response_bytes) > MAX_RESPONSE_BYTES:
        raise ValueError("response exceeds transport limit")
    return response_bytes


def main() -> int:
    try:
        response_bytes = _response_bytes(_run(_read_request()))
    except Exception as error:
        if os.environ.get("BCNLT_DIAGNOSTIC_TRACEBACK") == "1":
            traceback.print_exc()
        print(f"BCNLT sidecar rejected input: {type(error).__name__}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(response_bytes)
    return 0


def worker_main() -> int:
    """Serve bounded newline-framed requests until clean stdin EOF."""

    while True:
        raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 2)
        if not raw:
            return 0
        if len(raw) > MAX_REQUEST_BYTES + 1 or not raw.endswith(b"\n"):
            error = {
                "protocol": PROTOCOL,
                "ok": False,
                "error_type": "FrameError",
            }
            sys.stdout.buffer.write(_response_bytes(error))
            sys.stdout.buffer.flush()
            return 2
        try:
            response = _run(_parse_request(raw[:-1]))
        except Exception as error:
            if os.environ.get("BCNLT_DIAGNOSTIC_TRACEBACK") == "1":
                traceback.print_exc()
            response = {
                "protocol": PROTOCOL,
                "ok": False,
                "error_type": type(error).__name__,
            }
        sys.stdout.buffer.write(_response_bytes(response))
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit(main())
    if sys.argv[1:] == ["--worker"]:
        raise SystemExit(worker_main())
    print("BCNLT sidecar rejected command-line arguments", file=sys.stderr)
    raise SystemExit(2)
