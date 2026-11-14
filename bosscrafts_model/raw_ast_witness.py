"""Host client for the raw-AST witness sidecar."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from pathlib import Path

from .bcnlt_source_pipeline import (
    PARSER_RUNTIME_IDENTITY_SHA256,
    BCNLTSourcePipeline,
    _sidecar_environment,
)

PROTOCOL = "bosscrafts-raw-ast-witness-v2"
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_SIDECAR = Path(__file__).with_name("raw_ast_witness_sidecar.py")
_PYTHON_313 = Path(sys.executable).resolve()
SIDECAR_SOURCE_SHA256 = hashlib.sha256(_SIDECAR.read_bytes()).hexdigest()
_RESPONSE_KEYS = {
    "protocol",
    "sidecar_source_sha256",
    "parser_runtime_identity_sha256",
    "source_sha256",
    "language",
    "path",
    "dialect",
    "root_node_type",
    "witnesses",
}
_WITNESS_KEYS = {
    "query_id",
    "span",
    "ast_relation",
    "exact_named_node_types",
    "boundary_start_named_node_types",
    "boundary_end_named_node_types",
    "relation_span_match",
    "claimed_scope",
    "derived_scope",
    "scope_match",
}


def _validate_witness_rows(
    value: object, queries: tuple[dict[str, object], ...]
) -> None:
    if type(value) is not list or len(value) != len(queries):
        raise ValueError("raw AST witness response coverage mismatch")
    for row, query in zip(value, queries, strict=True):
        claimed_scope = query.get("scope")
        if (
            type(row) is not dict
            or set(row) != _WITNESS_KEYS
            or row.get("query_id") != query.get("query_id")
            or row.get("span") != query.get("span")
            or row.get("ast_relation") != query.get("ast_relation")
            or row.get("claimed_scope") != claimed_scope
            or type(row.get("relation_span_match")) is not bool
            or (
                row.get("derived_scope") is not None
                and (
                    type(row.get("derived_scope")) is not str
                    or not row["derived_scope"]
                )
            )
            or (
                claimed_scope is None
                and row.get("scope_match") is not None
            )
            or (
                claimed_scope is not None
                and type(row.get("scope_match")) is not bool
            )
        ):
            raise ValueError("raw AST witness response row mismatch")
        for field in (
            "exact_named_node_types",
            "boundary_start_named_node_types",
            "boundary_end_named_node_types",
        ):
            node_types = row.get(field)
            if (
                type(node_types) is not list
                or any(type(item) is not str or not item for item in node_types)
                or node_types != sorted(set(node_types))
            ):
                raise ValueError("raw AST witness response node types are invalid")
        if row["relation_span_match"] is False and (
            row["derived_scope"] is not None
            or row["scope_match"] not in {False, None}
        ):
            raise ValueError("raw AST witness response relation state is inconsistent")


def _close_process_streams(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            with suppress(OSError):
                stream.close()


class RawAstWitnessClient:
    """Obtain raw syntax evidence without invoking semantic extraction."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        expected_sidecar_sha256: str | None = None,
    ) -> None:
        if type(timeout_seconds) not in {int, float} or not 1 <= timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be between 1 and 300")
        expected = (
            SIDECAR_SOURCE_SHA256
            if expected_sidecar_sha256 is None
            else expected_sidecar_sha256
        )
        if (
            type(expected) is not str
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("expected_sidecar_sha256 must be a lowercase SHA-256")
        if hashlib.sha256(_SIDECAR.read_bytes()).hexdigest() != expected:
            raise ValueError("raw AST witness sidecar source SHA-256 mismatch")
        BCNLTSourcePipeline.validate_contained_runtime(_PYTHON_313)
        self._timeout_seconds = float(timeout_seconds)
        self._expected_sidecar_sha256 = expected
        self._bytecode_cache = tempfile.TemporaryDirectory(
            prefix="bosscrafts-raw-ast-pycache-"
        )
        self._process: subprocess.Popen[bytes] | None = None
        self._reader = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="raw-ast-witness-reader"
        )
        self._lock = threading.Lock()
        self._closed = False

    def _start(self) -> subprocess.Popen[bytes]:
        if self._closed:
            raise RuntimeError("raw AST witness client is closed")
        process = self._process
        if process is not None and process.poll() is None:
            return process
        if process is not None:
            _close_process_streams(process)
            self._process = None
        if (
            hashlib.sha256(_SIDECAR.read_bytes()).hexdigest()
            != self._expected_sidecar_sha256
        ):
            raise ValueError("raw AST witness sidecar source SHA-256 mismatch")
        process = subprocess.Popen(
            [
                str(_PYTHON_313),
                "-I",
                "-B",
                "-X",
                f"pycache_prefix={self._bytecode_cache.name}",
                "-X",
                "utf8",
                str(_SIDECAR),
                "--serve",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_sidecar_environment(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            _close_process_streams(process)
            raise RuntimeError("raw AST witness sidecar pipes are unavailable")
        self._process = process
        return process

    def _stop_process(self) -> bytes:
        process = self._process
        self._process = None
        if process is None:
            return b""
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            return process.stderr.read() if process.stderr is not None else b""
        finally:
            _close_process_streams(process)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            self._process = None
            if process is not None:
                try:
                    if process.stdin is not None:
                        with suppress(OSError):
                            process.stdin.close()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                finally:
                    _close_process_streams(process)
            self._reader.shutdown(wait=True, cancel_futures=True)
            self._bytecode_cache.cleanup()

    def __enter__(self) -> "RawAstWitnessClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def inspect(
        self,
        source: str,
        *,
        language: str,
        path: str,
        dialect: str | None,
        queries: tuple[dict[str, object], ...],
    ) -> dict[str, object]:
        if type(source) is not str:
            raise TypeError("source must be a string")
        if type(queries) is not tuple or not queries:
            raise ValueError("queries must be a non-empty tuple")
        request = {
            "protocol": PROTOCOL,
            "source": source,
            "language": language,
            "path": path,
            "dialect": dialect,
            "queries": list(queries),
        }
        encoded = (
            json.dumps(
                request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with self._lock:
            process = self._start()
            if process.stdin is None or process.stdout is None:
                self._stop_process()
                raise RuntimeError("raw AST witness sidecar pipes were not initialized")
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                detail = self._stop_process().decode(
                    "utf-8", errors="replace"
                ).strip()
                raise ValueError(
                    f"raw AST witness sidecar failed: {detail or type(exc).__name__}"
                ) from exc
            pending = self._reader.submit(
                process.stdout.readline, MAX_RESPONSE_BYTES + 1
            )
            try:
                raw_response = pending.result(timeout=self._timeout_seconds)
            except FutureTimeoutError as exc:
                self._stop_process()
                with suppress(Exception):
                    pending.result(timeout=5)
                raise subprocess.TimeoutExpired(
                    str(_SIDECAR), self._timeout_seconds
                ) from exc
            if not raw_response:
                return_code = process.poll()
                detail = self._stop_process().decode(
                    "utf-8", errors="replace"
                ).strip()
                raise ValueError(
                    "raw AST witness sidecar failed: "
                    f"{detail or f'exit {return_code}'}"
                )
            if len(raw_response) > MAX_RESPONSE_BYTES:
                self._stop_process()
                raise ValueError("raw AST witness response exceeds bounded transport")
        try:
            response = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("raw AST witness response is invalid") from exc
        if type(response) is not dict or set(response) != _RESPONSE_KEYS:
            raise ValueError("raw AST witness response keys mismatch")
        if (
            response["protocol"] != PROTOCOL
            or response["sidecar_source_sha256"]
            != self._expected_sidecar_sha256
            or response["parser_runtime_identity_sha256"]
            != PARSER_RUNTIME_IDENTITY_SHA256
            or response["source_sha256"]
            != hashlib.sha256(source.encode("utf-8")).hexdigest()
            or response["language"] != language
            or response["path"] != path
            or response["dialect"] != dialect
            or type(response["root_node_type"]) is not str
            or not response["root_node_type"]
        ):
            raise ValueError("raw AST witness response provenance mismatch")
        _validate_witness_rows(response["witnesses"], queries)
        return response
