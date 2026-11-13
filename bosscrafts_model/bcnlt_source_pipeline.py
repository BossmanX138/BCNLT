"""Live BCNLT semantic preprocessing bound to the frozen native tokenizer.

Parsing runs in an isolated copy of the active CPython 3.13 interpreter so its
pinned native Tree-sitter modules remain separate from the model process. The
sidecar only parses source and returns canonical JSONL. This module validates
that handoff, applies the frozen structural-token protocol, and preserves an
explicit raw-code path.

Learned meanings remain outside the immutable tokenizer vocabulary.  The
default is ephemeral and therefore byte-for-byte compatible with the frozen
foundation behavior.  A caller may explicitly select ``observe`` to append
bounded wildcard evidence, or ``read_only`` to consume a previously validated
Extended Core without mutating it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .bcnlt_adapter import (
    EXTRACTION_RULE_VERSION,
    GRAMMAR_PACKAGE_VERSION,
    PARSER_BUNDLE_SHA256,
    BCNLTCanonicalRecord,
    BCNLTMultilingualOutputAdapter,
    BCNLTParserOutputProvenance,
    validate_canonical_record_payload,
)
from .runtime_identity import (
    PYTHON_SOURCE_IDENTITY_POLICY,
)
from .runtime_identity import (
    aggregate_sha256 as _aggregate_sha256,
)
from .runtime_identity import (
    identity_bytes as _identity_bytes,
)
from .runtime_identity import (
    is_generated_python_bytecode as _is_generated_python_bytecode,
)
from .runtime_identity import (
    sha256_file as _sha256,
)
from .semantic import NativeSemanticRecordCodec
from .tokenizer import NativeTokenizerBinding

SIDECAR_PROTOCOL = "bosscrafts-bcnlt-source-sidecar-v4"
CURRENT_RUNTIME_SPEC_SHA256 = {
    "bash": "da1bb0468fb23c5a6eaa8af1eb0e9eeea4b9e295e7ffbc0d4aeb073c671810ca",
    "c": "f3765a6dd8687810a0d51f0ea9067af9bbee6642b98f23419168cd34d75e37f5",
    "c_sharp": "f11b7bec87537d6d7acab18156bc92b6fe262e935f13df421b805b6a5ebef105",
    "cpp": "c499bd19f76c84e522d6a8370a5d5e6ae26a96c7ca6bddf54b8f01e177182378",
    "go": "23f3a60f689ae22f34e5b819066d8e29ce823d4535cfca915e945ba2f2f3e6c2",
    "java": "c20f4e36ab239b8ed7f75428587f72d56f7e85c2f0a46a5a38a853f0ba033e4b",
    "javascript": "b300189c01eb2607a1d6597e9fef5b5d74795d7283f5194ca1b11712673c96c3",
    "kotlin": "02568b22e74dc1a05d3c4cf84c23808eec19f818ee312f497107782bbf3be9a5",
    "php": "6f5aaff7714b1893ff7966d671700d6fc55a641b7665dcdd98701e11904e57ed",
    "python": "9dec2e498ca82f936d4d454a650de48f55dcea094000f27507dbbaf0cd408381",
    "ruby": "59ecd18636cc23219ce4e0a50bebe5b5904db420dca47b2666e108387d77ae57",
    "rust": "d0df302624a6374508c715ea95e4351a5479e7fbced804784b0c19df4831d66b",
    "swift": "d553d2c3df05cc7448e833ecf54266fb73a297d74b57db1bea0be1a099e3dd5a",
    "typescript": "7124ea196695ceb20d9657f9cd4f144c97a872632ec07a7c55a90de23dbf6e2b",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_ROOT = _PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate"
_SIDECAR = _RUNTIME_ROOT / "bcnlt_source_sidecar.py"
_DEFAULT_PYTHON_313 = Path(sys.executable).resolve()
_RUNTIME_MANIFEST = _RUNTIME_ROOT / "runtime-manifest.json"
_EXPECTED_RUNTIME_IDENTITY = {
    "source": "c5623e97f3f1e1e884e1d54a316d174292fbcbf61da3ddfdbf3b432418e0450a",
    "sidecar": "5eb8df0226ada9b5ba9d7d5b845d86abb6b4dbd22423c35a7ba6bff1f31be1f3",
    "tree_sitter": "d8efa5533a1c79cc404df3a0ff03d76626ad50d54b74d97e4cf6e30ab85bebe4",
    "language_pack": "18fda57fbe2f52f85cf09fd1fc5af6509bfe0a8c6d3d425cafe3d7b63f8dca6e",
    "grammars": "19fc42827e8ab3b37282c0666caa1ddd8abcd3708298fb1665c10df440c3d090",
    "site_packages": "3daa4575bdd757c649febf2c8514c80ab1311641c0205463b1c89b1e928b262c",
}
_EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "cb7b0df010bbe162b0b9c0877addded2f5b06f3bc0fbbd9b39b58a7bdad6ed20"
)
PARSER_RUNTIME_IDENTITY_SHA256 = (
    "c1ca753bf4d1dbabf841fdcd120e5c3eec026b7211ca5c0f08d066074acee938"
)
PARSER_RUNTIME_NOT_APPLICABLE = "not_applicable"
_FIELD_ORDER = (
    "surface",
    "scope",
    "symbol_id",
    "defines",
    "span",
    "ast_relation",
    "ir",
    "recent_history",
    "metadata",
    "confidence",
    "uncertainty",
)
_JSON_FIELDS = frozenset(("span", "ir", "recent_history", "metadata"))
_SCORE_FIELDS = frozenset(("confidence", "uncertainty"))
_SEMANTIC_RECORD_BATCH_SIZE = 128
_RESPONSE_KEYS = frozenset(
    (
        "protocol",
        "sidecar_source_sha256",
        "parser_runtime_identity_sha256",
        "language",
        "path",
        "dialect",
        "semantic_schema_version",
        "semantic_spec_sha256",
        "source_sha256",
        "canonical_sha256",
        "record_count",
        "canonical_jsonl",
        "learning_mode",
        "learned_memory_sha256",
        "learned_definition_count",
        "extended_core_entry_count",
    )
)
_WORKER_ERROR_KEYS = frozenset(("protocol", "ok", "error_type"))
_MAX_SIDECAR_RESPONSE_BYTES = 16 * 1024 * 1024 * 6 + 64 * 1024


class BCNLTPreprocessingError(RuntimeError):
    """A fail-closed live parser or semantic handoff failure."""


@dataclass(frozen=True)
class PreparedBCNLTInput:
    mode: str
    language: str
    path: str
    token_ids: tuple[int, ...]
    binding_sha256: str
    tokenizer_sha256: str
    source_sha256: str
    canonical_jsonl: str | None
    canonical_sha256: str | None
    semantic_spec_sha256: str | None
    parser_runtime_identity_sha256: str | None
    record_count: int
    learning_mode: str
    learned_memory_sha256: str | None
    learned_definition_count: int
    extended_core_entry_count: int
    used_raw_fallback: bool = False
    fallback_reason: str | None = None


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate sidecar response key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _sidecar_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "USERPROFILE": os.environ.get("USERPROFILE", str(Path.home())),
        "LOCALAPPDATA": os.environ.get(
            "LOCALAPPDATA", str(Path.home() / "AppData" / "Local")
        ),
        "TEMP": os.environ.get("TEMP", str(_PROJECT_ROOT / ".tmp")),
        "TMP": os.environ.get("TMP", str(_PROJECT_ROOT / ".tmp")),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _unexpected_runtime_cache_entries(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" in path.relative_to(root).parts
        and not _is_generated_python_bytecode(path, root)
    )


class _PersistentSidecarWorker:
    """One serial, restartable parser process with bounded framed I/O."""

    def __init__(self, python: Path, *, timeout_seconds: float) -> None:
        self._python = python
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._bytecode_cache = tempfile.TemporaryDirectory(
            prefix="bosscrafts-bcnlt-pycache-"
        )
        self._reader = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bcnlt-sidecar-reader",
        )

    @property
    def pid(self) -> int | None:
        with self._lock:
            process = self._process
            return (
                process.pid if process is not None and process.poll() is None else None
            )

    def _start_locked(self) -> subprocess.Popen[bytes]:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        if process is not None:
            self._close_streams(process)
        if _sha256(_SIDECAR) != _EXPECTED_RUNTIME_IDENTITY["sidecar"]:
            raise BCNLTPreprocessingError(
                "BCNLT persistent parser sidecar source identity changed"
            )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [
                    str(self._python),
                    "-I",
                    "-B",
                    "-X",
                    f"pycache_prefix={self._bytecode_cache.name}",
                    "-X",
                    "utf8",
                    str(_SIDECAR),
                    "--worker",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=_sidecar_environment(),
                creationflags=creationflags,
                bufsize=0,
            )
        except OSError as error:
            raise BCNLTPreprocessingError(
                "BCNLT persistent parser sidecar could not start"
            ) from error
        if process.stdin is None or process.stdout is None:
            process.kill()
            process.wait(timeout=5)
            raise BCNLTPreprocessingError(
                "BCNLT persistent sidecar pipes are unavailable"
            )
        self._process = process
        return process

    @staticmethod
    def _close_streams(process: subprocess.Popen[bytes]) -> None:
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _stop_locked(self, *, force: bool) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None and not force and process.stdin is not None:
            try:
                process.stdin.close()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process.poll() is None:
            try:
                process.kill() if force else process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        self._close_streams(process)

    def request(self, request_bytes: bytes) -> bytes:
        if b"\n" in request_bytes:
            raise BCNLTPreprocessingError("BCNLT worker request framing mismatch")
        with self._lock:
            process = self._start_locked()
            if process.stdin is None or process.stdout is None:
                raise BCNLTPreprocessingError(
                    "BCNLT persistent sidecar pipes are unavailable"
                )
            try:
                process.stdin.write(request_bytes + b"\n")
                process.stdin.flush()
            except OSError as error:
                self._stop_locked(force=True)
                raise BCNLTPreprocessingError(
                    "BCNLT persistent parser sidecar request failed"
                ) from error
            future = self._reader.submit(
                process.stdout.readline,
                _MAX_SIDECAR_RESPONSE_BYTES + 2,
            )
            try:
                response = future.result(timeout=self._timeout_seconds)
            except FutureTimeoutError as error:
                self._stop_locked(force=True)
                try:
                    future.result(timeout=2)
                except BaseException:
                    pass
                raise BCNLTPreprocessingError(
                    "BCNLT persistent parser sidecar timed out"
                ) from error
            except OSError as error:
                self._stop_locked(force=True)
                raise BCNLTPreprocessingError(
                    "BCNLT persistent parser sidecar response failed"
                ) from error
            if (
                not response
                or len(response) > _MAX_SIDECAR_RESPONSE_BYTES + 1
                or not response.endswith(b"\n")
            ):
                self._stop_locked(force=True)
                raise BCNLTPreprocessingError(
                    "BCNLT persistent parser sidecar returned an invalid frame"
                )
            return response

    def close(self) -> None:
        with self._lock:
            self._stop_locked(force=False)
        self._reader.shutdown(wait=True, cancel_futures=True)
        self._bytecode_cache.cleanup()


class _NativeSemanticStreamCodec:
    """Full BCNLT v3 record stream over the immutable 65K vocabulary."""

    def __init__(self, binding: NativeTokenizerBinding):
        self._binding = binding
        self._reserved_ids = frozenset(range(77))
        self._event_by_id = {
            token_id: token.removeprefix("<|bcnlt:event:").removesuffix("|>").upper()
            for token, token_id in binding.structural_token_ids.items()
            if token.startswith("<|bcnlt:event:")
        }

    def _id(self, name: str) -> int:
        token = f"<|bcnlt:{name}|>"
        try:
            return self._binding.structural_token_ids[token]
        except KeyError as error:
            raise ValueError(f"frozen structural token is missing: {token}") from error

    def _encode_payload_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return self._binding.encode_payload_batch(texts)

    def encode(self, records: Sequence[BCNLTCanonicalRecord]) -> tuple[int, ...]:
        ids = [self._id("stream:begin"), self._id("mode:semantic_code")]
        for start in range(0, len(records), _SEMANTIC_RECORD_BATCH_SIZE):
            payload_texts: list[str] = []
            prepared: list[
                tuple[BCNLTCanonicalRecord, list[tuple[str, object, int | None]]]
            ] = []
            for record in records[start : start + _SEMANTIC_RECORD_BATCH_SIZE]:
                payload = json.loads(record.canonical_json)
                fields: list[tuple[str, object, int | None]] = []
                for field in _FIELD_ORDER:
                    value = (
                        payload[f"{field}_band"]
                        if field in _SCORE_FIELDS
                        else payload[field]
                    )
                    payload_index: int | None = None
                    if value is not None and field not in _SCORE_FIELDS:
                        text = (
                            json.dumps(
                                value,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            if field in _JSON_FIELDS
                            else value
                        )
                        if type(text) is not str:
                            raise ValueError(f"{field} payload must serialize to text")
                        payload_index = len(payload_texts)
                        payload_texts.append(text)
                    fields.append((field, value, payload_index))
                prepared.append((record, fields))
            encoded_payloads = self._encode_payload_batch(payload_texts)
            for record, fields in prepared:
                ids.extend(
                    (
                        self._id("record:begin"),
                        self._id(f"event:{record.event.lower()}"),
                    )
                )
                for field, value, payload_index in fields:
                    ids.append(self._id(f"field:{field}:begin"))
                    if value is None:
                        ids.append(self._id("value:null"))
                    elif field in _SCORE_FIELDS:
                        ids.append(self._id(f"value:{value}"))
                    elif payload_index is not None:
                        ids.extend(encoded_payloads[payload_index])
                    else:  # pragma: no cover - guarded by the field preparation above
                        raise RuntimeError("semantic payload batch index is missing")
                    ids.append(self._id(f"field:{field}:end"))
                ids.append(self._id("record:end"))
        ids.append(self._id("stream:end"))
        return tuple(ids)

    def decode(self, ids: Sequence[int]) -> str:
        values = tuple(ids)
        if len(values) < 3:
            raise ValueError("semantic token stream is truncated")
        if values[:2] != (self._id("stream:begin"), self._id("mode:semantic_code")):
            raise ValueError("semantic token stream header mismatch")
        if values[-1] != self._id("stream:end"):
            raise ValueError("semantic token stream end mismatch")
        cursor = 2
        lines: list[str] = []
        while cursor < len(values) - 1:
            if values[cursor] != self._id("record:begin"):
                raise ValueError("semantic record begin mismatch")
            cursor += 1
            try:
                event = self._event_by_id[values[cursor]]
            except (KeyError, IndexError) as error:
                raise ValueError("semantic event token mismatch") from error
            cursor += 1
            payload: dict[str, object] = {"event": event}
            for field in _FIELD_ORDER:
                if cursor >= len(values) - 1 or values[cursor] != self._id(
                    f"field:{field}:begin"
                ):
                    raise ValueError(f"semantic {field} field begin mismatch")
                cursor += 1
                end = self._id(f"field:{field}:end")
                body: list[int] = []
                while cursor < len(values) - 1 and values[cursor] != end:
                    body.append(values[cursor])
                    cursor += 1
                if cursor >= len(values) - 1:
                    raise ValueError(f"semantic {field} field is truncated")
                cursor += 1
                if body == [self._id("value:null")]:
                    value: object = None
                elif field in _SCORE_FIELDS and len(body) == 1:
                    bands = {
                        self._id("value:low"): "low",
                        self._id("value:medium"): "medium",
                        self._id("value:high"): "high",
                    }
                    try:
                        value = bands[body[0]]
                    except KeyError as error:
                        raise ValueError(f"semantic {field} band mismatch") from error
                else:
                    if any(token_id in self._reserved_ids for token_id in body):
                        raise ValueError(f"unexpected reserved token inside {field}")
                    text = self._binding.decode(body)
                    if field in _JSON_FIELDS:
                        value = json.loads(
                            text,
                            object_pairs_hook=_reject_duplicate_keys,
                            parse_constant=_reject_nonfinite,
                        )
                        canonical = json.dumps(
                            value,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if text != canonical:
                            raise ValueError(f"semantic {field} JSON is not canonical")
                    else:
                        value = text
                payload[f"{field}_band" if field in _SCORE_FIELDS else field] = value
            if cursor >= len(values) - 1 or values[cursor] != self._id("record:end"):
                raise ValueError("semantic record end mismatch")
            cursor += 1
            validate_canonical_record_payload(
                payload,
                allowed_events=frozenset(self._event_by_id.values()),
            )
            lines.append(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
        return "".join(f"{line}\n" for line in lines)


class BCNLTSourcePipeline:
    """Parse source to canonical BCNLT records and frozen model token IDs."""

    def __init__(
        self,
        binding: NativeTokenizerBinding,
        *,
        sidecar_python: str | Path | None = None,
        timeout_seconds: float = 30.0,
        persistent_sidecars: bool = False,
        learning_mode: str = "ephemeral",
        learned_memory_root: str | Path | None = None,
    ) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        if type(timeout_seconds) not in (int, float):
            raise TypeError("sidecar timeout must be a finite number")
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > 60
        ):
            raise ValueError("sidecar timeout must be in (0, 60] seconds")
        if type(persistent_sidecars) is not bool:
            raise TypeError("persistent_sidecars must be a boolean")
        if type(learning_mode) is not str:
            raise TypeError("learning_mode must be a string")
        if learning_mode not in {"ephemeral", "observe", "read_only"}:
            raise ValueError("learning_mode must be ephemeral, observe, or read_only")
        if learning_mode == "ephemeral":
            if learned_memory_root is not None:
                raise ValueError(
                    "ephemeral learning must not receive a learned-memory root"
                )
            resolved_memory_root = None
        else:
            if learned_memory_root is None:
                raise ValueError(
                    "persistent learning requires an explicit learned-memory root"
                )
            resolved_memory_root = Path(learned_memory_root).resolve()
            if resolved_memory_root == Path(resolved_memory_root.anchor):
                raise ValueError("learned-memory root cannot be a filesystem root")
            if resolved_memory_root.exists() and not resolved_memory_root.is_dir():
                raise ValueError("learned-memory root must be a directory")
        configured = sidecar_python or os.environ.get("BCNLT_SIDECAR_PYTHON")
        self._python = Path(configured) if configured else _DEFAULT_PYTHON_313
        self._timeout_seconds = float(timeout_seconds)
        self._persistent_sidecars = persistent_sidecars
        self._learning_mode = learning_mode
        self._learned_memory_root = resolved_memory_root
        self._worker_local = threading.local()
        self._workers: set[_PersistentSidecarWorker] = set()
        self._workers_lock = threading.Lock()
        self._binding = binding
        self._stream_codec = _NativeSemanticStreamCodec(binding)
        self._adapter = BCNLTMultilingualOutputAdapter(
            NativeSemanticRecordCodec(binding),
            runtime_spec_sha256=CURRENT_RUNTIME_SPEC_SHA256,
        )
        self.validate_contained_runtime(self._python)

    @property
    def parser_execution_integrated(self) -> bool:
        return True

    @property
    def parser_runtime_identity_sha256(self) -> str:
        return PARSER_RUNTIME_IDENTITY_SHA256

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset(CURRENT_RUNTIME_SPEC_SHA256)

    @property
    def persistent_sidecars_enabled(self) -> bool:
        return self._persistent_sidecars

    @property
    def sidecar_timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def learning_mode(self) -> str:
        return self._learning_mode

    @property
    def learned_memory_root(self) -> Path | None:
        return self._learned_memory_root

    @property
    def persistent_worker_pids(self) -> tuple[int, ...]:
        with self._workers_lock:
            return tuple(
                sorted(
                    pid for worker in self._workers if (pid := worker.pid) is not None
                )
            )

    def close(self) -> None:
        with self._workers_lock:
            workers = tuple(self._workers)
            self._workers.clear()
        for worker in workers:
            worker.close()
        if hasattr(self._worker_local, "worker"):
            del self._worker_local.worker

    def __enter__(self) -> "BCNLTSourcePipeline":
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        self.close()

    def _persistent_worker(self) -> _PersistentSidecarWorker:
        worker = getattr(self._worker_local, "worker", None)
        if isinstance(worker, _PersistentSidecarWorker):
            return worker
        worker = _PersistentSidecarWorker(
            self._python,
            timeout_seconds=self._timeout_seconds,
        )
        self._worker_local.worker = worker
        with self._workers_lock:
            self._workers.add(worker)
        return worker

    @staticmethod
    def validate_contained_runtime(python: Path) -> None:
        """Fail closed unless the shipped parser runtime and interpreter match."""

        if not python.is_file():
            raise ValueError("pinned CPython 3.13 sidecar interpreter is unavailable")
        required = (
            _SIDECAR,
            _RUNTIME_ROOT / "source" / "bcnlt" / "__init__.py",
            _RUNTIME_ROOT
            / "site-packages"
            / "tree_sitter"
            / "_binding.cp313-win_amd64.pyd",
            _RUNTIME_ROOT
            / "site-packages"
            / "tree_sitter_language_pack"
            / "_native.pyd",
        )
        if any(not path.is_file() for path in required):
            raise ValueError("project-contained BCNLT sidecar runtime is incomplete")
        grammar_names = (
            "bash",
            "c",
            "c_sharp",
            "cpp",
            "go",
            "java",
            "javascript",
            "kotlin",
            "php",
            "python",
            "ruby",
            "rust",
            "swift",
            "typescript",
        )
        if any(
            not (_RUNTIME_ROOT / "grammars" / f"tree_sitter_{name}.dll").is_file()
            for name in grammar_names
        ):
            raise ValueError("project-contained BCNLT grammar set is incomplete")
        if _sha256(_RUNTIME_MANIFEST) != _EXPECTED_RUNTIME_MANIFEST_SHA256:
            raise ValueError("project-contained BCNLT runtime manifest identity mismatch")
        try:
            manifest = json.loads(
                _RUNTIME_MANIFEST.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "project-contained BCNLT runtime manifest is invalid"
            ) from error
        if (
            manifest.get("schema_version")
            != "bosscrafts-bcnlt-contained-runtime-v10-learning-candidate"
        ):
            raise ValueError("project-contained BCNLT runtime manifest schema mismatch")
        source_hash, source_count, source_bytes = _aggregate_sha256(
            _RUNTIME_ROOT / "source" / "bcnlt", "*.py"
        )
        grammar_hash, grammar_count, grammar_bytes = _aggregate_sha256(
            _RUNTIME_ROOT / "grammars", "*.dll"
        )
        site_packages_root = _RUNTIME_ROOT / "site-packages"
        site_packages_hash, site_packages_count, site_packages_bytes = (
            _aggregate_sha256(
                site_packages_root,
                "*",
                ignore_generated_python_bytecode=True,
            )
        )
        unexpected_cache_entries = _unexpected_runtime_cache_entries(
            site_packages_root
        )
        if unexpected_cache_entries:
            raise ValueError("project-contained BCNLT runtime cache contains non-bytecode")
        source_manifest = manifest.get("source", {})
        runtime_manifest = manifest.get("runtime", {})
        actual = {
            "source": source_hash,
            "sidecar": _sha256(_SIDECAR),
            "tree_sitter": _sha256(required[2]),
            "language_pack": _sha256(required[3]),
            "grammars": grammar_hash,
            "site_packages": site_packages_hash,
        }
        if actual != _EXPECTED_RUNTIME_IDENTITY:
            raise ValueError("project-contained BCNLT runtime identity mismatch")
        runtime_identity = hashlib.sha256(
            json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if runtime_identity != PARSER_RUNTIME_IDENTITY_SHA256:
            raise ValueError(
                "project-contained BCNLT runtime aggregate identity mismatch"
            )
        if (
            source_manifest.get("aggregate_sha256") != source_hash
            or source_manifest.get("file_count") != source_count
            or source_manifest.get("bytes") != source_bytes
            or source_manifest.get("identity_policy") != PYTHON_SOURCE_IDENTITY_POLICY
            or runtime_manifest.get("sidecar_sha256") != actual["sidecar"]
            or runtime_manifest.get("sidecar_bytes") != len(_identity_bytes(_SIDECAR))
            or runtime_manifest.get("tree_sitter_binding_sha256")
            != actual["tree_sitter"]
            or runtime_manifest.get("language_pack_binding_sha256")
            != actual["language_pack"]
            or runtime_manifest.get("site_packages_aggregate_sha256")
            != site_packages_hash
            or runtime_manifest.get("site_packages_file_count")
            != site_packages_count
            or runtime_manifest.get("site_packages_bytes") != site_packages_bytes
            or runtime_manifest.get("grammar_aggregate_sha256") != grammar_hash
            or runtime_manifest.get("grammar_count") != grammar_count
            or runtime_manifest.get("grammar_bytes") != grammar_bytes
            or runtime_manifest.get("python_source_identity_policy")
            != PYTHON_SOURCE_IDENTITY_POLICY
            or runtime_manifest.get("parser_runtime_identity_sha256")
            != PARSER_RUNTIME_IDENTITY_SHA256
        ):
            raise ValueError(
                "project-contained BCNLT runtime manifest identity mismatch"
            )
        version_check = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import platform,struct,sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{platform.python_implementation()}|{struct.calcsize(\"P\") * 8}')",
            ],
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if (
            version_check.returncode != 0
            or version_check.stdout.strip() != "3.13|CPython|64"
        ):
            raise ValueError("BCNLT sidecar requires 64-bit CPython 3.13")

    def _provenance(self, language: str) -> BCNLTParserOutputProvenance:
        return BCNLTParserOutputProvenance(
            language=language,
            parser_bundle_sha256=PARSER_BUNDLE_SHA256,
            semantic_spec_sha256=CURRENT_RUNTIME_SPEC_SHA256[language],
            grammar_revision=f"{GRAMMAR_PACKAGE_VERSION}/{language}",
            extraction_rule_version=EXTRACTION_RULE_VERSION,
        )

    def _invoke_sidecar(
        self,
        source: str,
        *,
        language: str,
        path: str,
        dialect: str | None,
    ) -> dict[str, object]:
        request = json.dumps(
            {
                "protocol": SIDECAR_PROTOCOL,
                "source": source,
                "language": language,
                "path": path,
                "dialect": dialect,
                "learning_mode": self._learning_mode,
                "learned_memory_root": (
                    str(self._learned_memory_root)
                    if self._learned_memory_root is not None
                    else None
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            request_bytes = request.encode("utf-8")
        except UnicodeEncodeError as error:
            raise BCNLTPreprocessingError(
                "BCNLT parser sidecar request is not UTF-8"
            ) from error
        if self._persistent_sidecars:
            stdout_bytes = self._persistent_worker().request(request_bytes)
        else:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if _sha256(_SIDECAR) != _EXPECTED_RUNTIME_IDENTITY["sidecar"]:
                raise BCNLTPreprocessingError(
                    "BCNLT parser sidecar source identity changed"
                )
            try:
                with tempfile.TemporaryDirectory(
                    prefix="bosscrafts-bcnlt-pycache-"
                ) as bytecode_cache:
                    completed = subprocess.run(
                        [
                            str(self._python),
                            "-I",
                            "-B",
                            "-X",
                            f"pycache_prefix={bytecode_cache}",
                            "-X",
                            "utf8",
                            str(_SIDECAR),
                        ],
                        input=request_bytes,
                        capture_output=True,
                        timeout=self._timeout_seconds,
                        check=False,
                        env=_sidecar_environment(),
                        creationflags=creationflags,
                    )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise BCNLTPreprocessingError(
                    "BCNLT parser sidecar could not complete"
                ) from error
            if completed.returncode != 0:
                stderr = completed.stderr.decode("utf-8", errors="replace")
                detail = (
                    stderr.strip().splitlines()[-1] if stderr.strip() else "rejected"
                )
                raise BCNLTPreprocessingError(f"BCNLT parser sidecar {detail}")
            stdout_bytes = completed.stdout
        try:
            stdout = stdout_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BCNLTPreprocessingError(
                "BCNLT parser sidecar returned non-UTF-8 output"
            ) from error
        try:
            response = json.loads(stdout, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise BCNLTPreprocessingError(
                "BCNLT parser sidecar returned invalid JSON"
            ) from error
        if type(response) is dict and set(response) == _WORKER_ERROR_KEYS:
            if (
                response.get("protocol") != SIDECAR_PROTOCOL
                or response.get("ok") is not False
                or type(response.get("error_type")) is not str
                or not response["error_type"]
                or len(response["error_type"]) > 128
            ):
                raise BCNLTPreprocessingError(
                    "BCNLT persistent sidecar error response mismatch"
                )
            raise BCNLTPreprocessingError(
                f"BCNLT parser sidecar rejected input: {response['error_type']}"
            )
        if type(response) is not dict or set(response) != _RESPONSE_KEYS:
            raise BCNLTPreprocessingError("BCNLT parser sidecar response keys mismatch")
        return response

    def prepare_semantic(
        self,
        source: str,
        *,
        language: str,
        path: str = "<memory>",
        dialect: str | None = None,
        allow_raw_fallback: bool = False,
    ) -> PreparedBCNLTInput:
        if type(allow_raw_fallback) is not bool:
            raise TypeError("allow_raw_fallback must be a boolean")
        if type(source) is not str:
            raise TypeError("source must be a string")
        if language not in CURRENT_RUNTIME_SPEC_SHA256:
            raise ValueError(f"unsupported BCNLT language: {language}")
        if type(path) is not str or not path:
            raise ValueError("path must be a non-empty logical path")
        try:
            response = self._invoke_sidecar(
                source,
                language=language,
                path=path,
                dialect=dialect,
            )
            canonical_jsonl = response["canonical_jsonl"]
            if type(canonical_jsonl) is not str or (
                canonical_jsonl and not canonical_jsonl.endswith("\n")
            ):
                raise BCNLTPreprocessingError("BCNLT canonical JSONL framing mismatch")
            expected_source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            expected_canonical_hash = hashlib.sha256(
                canonical_jsonl.encode("utf-8")
            ).hexdigest()
            if (
                response["protocol"] != SIDECAR_PROTOCOL
                or response["sidecar_source_sha256"]
                != _EXPECTED_RUNTIME_IDENTITY["sidecar"]
                or response["parser_runtime_identity_sha256"]
                != PARSER_RUNTIME_IDENTITY_SHA256
                or response["language"] != language
                or response["path"] != path
                or response["dialect"] != dialect
                or response["semantic_schema_version"] != "bcnlt-semantic-record-v3"
                or response["semantic_spec_sha256"]
                != CURRENT_RUNTIME_SPEC_SHA256[language]
                or response["source_sha256"] != expected_source_hash
                or response["canonical_sha256"] != expected_canonical_hash
            ):
                raise BCNLTPreprocessingError(
                    "BCNLT parser sidecar provenance mismatch"
                )
            if not canonical_jsonl:
                records: tuple[BCNLTCanonicalRecord, ...] = ()
            else:
                records = self._adapter.parse_jsonl(
                    canonical_jsonl, self._provenance(language)
                )
            if type(response["record_count"]) is not int or response[
                "record_count"
            ] != len(records):
                raise BCNLTPreprocessingError(
                    "BCNLT parser sidecar record count mismatch"
                )
            learned_memory_sha256 = response["learned_memory_sha256"]
            learned_definition_count = response["learned_definition_count"]
            extended_core_entry_count = response["extended_core_entry_count"]
            if (
                response["learning_mode"] != self._learning_mode
                or (
                    learned_memory_sha256 is not None
                    and (
                        type(learned_memory_sha256) is not str
                        or len(learned_memory_sha256) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in learned_memory_sha256
                        )
                    )
                )
                or type(learned_definition_count) is not int
                or learned_definition_count < 0
                or type(extended_core_entry_count) is not int
                or extended_core_entry_count < 0
                or extended_core_entry_count > learned_definition_count
                or (
                    self._learning_mode == "ephemeral"
                    and (
                        learned_memory_sha256 is not None
                        or learned_definition_count != 0
                        or extended_core_entry_count != 0
                    )
                )
            ):
                raise BCNLTPreprocessingError(
                    "BCNLT learned-memory provenance mismatch"
                )
            token_ids = self._stream_codec.encode(records)
            if self._stream_codec.decode(token_ids) != canonical_jsonl:
                raise BCNLTPreprocessingError(
                    "BCNLT semantic token stream failed round-trip"
                )
            return PreparedBCNLTInput(
                mode="semantic_code",
                language=language,
                path=path,
                token_ids=token_ids,
                binding_sha256=self._binding.binding_sha256,
                tokenizer_sha256=self._binding.tokenizer_sha256,
                source_sha256=expected_source_hash,
                canonical_jsonl=canonical_jsonl,
                canonical_sha256=expected_canonical_hash,
                semantic_spec_sha256=CURRENT_RUNTIME_SPEC_SHA256[language],
                parser_runtime_identity_sha256=PARSER_RUNTIME_IDENTITY_SHA256,
                record_count=len(records),
                learning_mode=self._learning_mode,
                learned_memory_sha256=learned_memory_sha256,
                learned_definition_count=learned_definition_count,
                extended_core_entry_count=extended_core_entry_count,
            )
        except (TypeError, ValueError, UnicodeError) as error:
            normalized = BCNLTPreprocessingError(
                "BCNLT semantic handoff validation failed"
            )
            if not allow_raw_fallback:
                raise normalized from error
            return self.prepare_raw(
                source,
                language=language,
                path=path,
                fallback_reason=str(normalized),
            )
        except BCNLTPreprocessingError as error:
            if not allow_raw_fallback:
                raise
            return self.prepare_raw(
                source,
                language=language,
                path=path,
                fallback_reason=str(error),
            )

    def prepare_raw(
        self,
        source: str,
        *,
        language: str,
        path: str = "<memory>",
        fallback_reason: str | None = None,
    ) -> PreparedBCNLTInput:
        if type(source) is not str:
            raise TypeError("source must be a string")
        if language not in CURRENT_RUNTIME_SPEC_SHA256:
            raise ValueError(f"unsupported BCNLT language: {language}")
        if type(path) is not str or not path:
            raise ValueError("path must be a non-empty logical path")
        ids = (
            self._binding.structural_token_ids["<|bcnlt:stream:begin|>"],
            self._binding.structural_token_ids["<|bcnlt:mode:raw_code|>"],
            *self._binding.encode_payload(source),
            self._binding.structural_token_ids["<|bcnlt:stream:end|>"],
        )
        return PreparedBCNLTInput(
            mode="raw_code",
            language=language,
            path=path,
            token_ids=ids,
            binding_sha256=self._binding.binding_sha256,
            tokenizer_sha256=self._binding.tokenizer_sha256,
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            canonical_jsonl=None,
            canonical_sha256=None,
            semantic_spec_sha256=None,
            parser_runtime_identity_sha256=None,
            record_count=0,
            learning_mode=self._learning_mode,
            learned_memory_sha256=None,
            learned_definition_count=0,
            extended_core_entry_count=0,
            used_raw_fallback=fallback_reason is not None,
            fallback_reason=fallback_reason,
        )

    def decode_semantic(self, ids: Sequence[int]) -> str:
        return self._stream_codec.decode(ids)

    def decode_raw(self, ids: Sequence[int]) -> str:
        values = tuple(ids)
        if (
            len(values) < 3
            or values[0] != self._binding.structural_token_ids["<|bcnlt:stream:begin|>"]
        ):
            raise ValueError("raw token stream header mismatch")
        if values[1] != self._binding.structural_token_ids["<|bcnlt:mode:raw_code|>"]:
            raise ValueError("raw token stream mode mismatch")
        if values[-1] != self._binding.structural_token_ids["<|bcnlt:stream:end|>"]:
            raise ValueError("raw token stream end mismatch")
        if any(0 <= token_id <= 76 for token_id in values[2:-1]):
            raise ValueError("raw token stream contains a reserved control token")
        return self._binding.decode(values[2:-1])
