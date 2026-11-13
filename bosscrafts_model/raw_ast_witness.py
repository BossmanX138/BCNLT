"""Host client for the raw-AST witness sidecar."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import hashlib
import json
from pathlib import Path
import subprocess
import threading

from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256


PROTOCOL = "bosscrafts-raw-ast-witness-v1"
_SIDECAR = Path(__file__).with_name("raw_ast_witness_sidecar.py")
_PYTHON_313 = Path(r"C:\ProgramData\anaconda3\python.exe")
_RESPONSE_KEYS = {
    "protocol",
    "parser_runtime_identity_sha256",
    "source_sha256",
    "language",
    "path",
    "dialect",
    "root_node_type",
    "witnesses",
}


class RawAstWitnessClient:
    """Obtain raw syntax evidence without invoking semantic extraction."""

    def __init__(self, *, timeout_seconds: float = 60.0) -> None:
        if type(timeout_seconds) not in {int, float} or not 1 <= timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be between 1 and 300")
        self._timeout_seconds = float(timeout_seconds)
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
            self._process = None
        process = subprocess.Popen(
            [str(_PYTHON_313), "-I", str(_SIDECAR), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            raise RuntimeError("raw AST witness sidecar pipes are unavailable")
        self._process = process
        return process

    def _stop_process(self) -> bytes:
        process = self._process
        self._process = None
        if process is None:
            return b""
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if process.stderr is None:
            return b""
        return process.stderr.read()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            self._process = None
            if process is not None:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            self._reader.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "RawAstWitnessClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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
            assert process.stdin is not None
            assert process.stdout is not None
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
            pending = self._reader.submit(process.stdout.readline)
            try:
                raw_response = pending.result(timeout=self._timeout_seconds)
            except FutureTimeoutError as exc:
                self._stop_process()
                try:
                    pending.result(timeout=5)
                except Exception:
                    pass
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
        try:
            response = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("raw AST witness response is invalid") from exc
        if type(response) is not dict or set(response) != _RESPONSE_KEYS:
            raise ValueError("raw AST witness response keys mismatch")
        if (
            response["protocol"] != PROTOCOL
            or response["parser_runtime_identity_sha256"]
            != PARSER_RUNTIME_IDENTITY_SHA256
            or response["source_sha256"]
            != hashlib.sha256(source.encode("utf-8")).hexdigest()
            or response["language"] != language
            or response["path"] != path
            or response["dialect"] != dialect
            or type(response["root_node_type"]) is not str
            or type(response["witnesses"]) is not list
            or len(response["witnesses"]) != len(queries)
        ):
            raise ValueError("raw AST witness response provenance mismatch")
        expected_ids = [query.get("query_id") for query in queries]
        actual_ids = [row.get("query_id") for row in response["witnesses"]]
        if actual_ids != expected_ids:
            raise ValueError("raw AST witness query identity mismatch")
        return response
