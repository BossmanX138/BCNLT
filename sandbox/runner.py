#!/usr/bin/env python3
"""Container-internal compiler and executor for BCNLT scientific fixtures.

The host supplies one bounded JSON request.  Commands are selected exclusively
from this file; source text is never interpolated into a shell command.  The
container has no host mounts or network and is discarded after one request.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from typing import Any


PROTOCOL = "bosscrafts-bcnlt-sandbox-v1"
MAX_REQUEST_BYTES = 128 * 1024
MAX_SOURCE_BYTES = 64 * 1024
MAX_STDIN_BYTES = 16 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
COMPILE_TIMEOUT_SECONDS = 20.0
RUN_TIMEOUT_SECONDS = 4.0
WORK = Path("/work")


LANGUAGES: dict[str, dict[str, Any]] = {
    "python": {
        "filename": "main.py",
        "compile": None,
        "run": ["python3", "-I", "-B", "/work/main.py"],
    },
    "javascript": {
        "filename": "main.js",
        "compile": None,
        "run": ["node", "--no-warnings", "/work/main.js"],
    },
    "typescript": {
        "filename": "main.ts",
        "compile": [
            "tsc", "--pretty", "false", "--target", "ES2022", "--module",
            "commonjs", "--outDir", "/work/out", "/work/main.ts",
        ],
        "run": ["node", "--no-warnings", "/work/out/main.js"],
    },
    "c": {
        "filename": "main.c",
        "compile": [
            "gcc", "-std=c17", "-O0", "-pipe", "-fno-ident", "-Wall",
            "-Wextra", "-o", "/work/program", "/work/main.c",
        ],
        "run": ["/work/program"],
    },
    "cpp": {
        "filename": "main.cpp",
        "compile": [
            "g++", "-std=c++20", "-O0", "-pipe", "-fno-ident", "-Wall",
            "-Wextra", "-o", "/work/program", "/work/main.cpp",
        ],
        "run": ["/work/program"],
    },
    "c_sharp": {
        "filename": "Main.cs",
        "compile": ["mcs", "-nologo", "-out:/work/program.exe", "/work/Main.cs"],
        "run": ["mono", "/work/program.exe"],
    },
    "java": {
        "filename": "Main.java",
        "compile": ["javac", "-encoding", "UTF-8", "-d", "/work", "/work/Main.java"],
        "run": ["java", "-Xms8m", "-Xmx96m", "-cp", "/work", "Main"],
    },
    "go": {
        "filename": "main.go",
        "compile": [
            "go", "build", "-p=1", "-trimpath", "-o", "/work/program", "/work/main.go",
        ],
        "run": ["/work/program"],
    },
    "rust": {
        "filename": "main.rs",
        "compile": [
            "rustc", "--edition=2021", "-C", "opt-level=0", "-C",
            "debuginfo=0", "-o", "/work/program", "/work/main.rs",
        ],
        "run": ["/work/program"],
    },
    "php": {
        "filename": "main.php",
        "compile": None,
        "run": ["php", "-n", "/work/main.php"],
    },
    "ruby": {
        "filename": "main.rb",
        "compile": None,
        "run": ["ruby", "--disable-gems", "/work/main.rb"],
    },
    "kotlin": {
        "filename": "Main.kt",
        "compile": [
            "kotlinc", "/work/Main.kt", "-include-runtime", "-d", "/work/program.jar",
        ],
        "run": ["java", "-Xms8m", "-Xmx96m", "-jar", "/work/program.jar"],
    },
    "swift": {
        "filename": "main.swift",
        "compile": [
            "swiftc", "-O", "-o", "/work/program", "/work/main.swift",
        ],
        "run": ["/work/program"],
    },
    "bash": {
        "filename": "main.sh",
        "compile": None,
        "run": ["bash", "--noprofile", "--norc", "/work/main.sh"],
    },
}


def _read_request() -> dict[str, object]:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("request exceeds sandbox transport limit")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("request must be one UTF-8 JSON object") from error
    if type(payload) is not dict or set(payload) != {
        "protocol", "language", "source", "stdin"
    }:
        raise ValueError("request keys mismatch")
    if payload["protocol"] != PROTOCOL:
        raise ValueError("request protocol mismatch")
    if payload["language"] not in LANGUAGES:
        raise ValueError("language is not in the sandbox allowlist")
    source = payload["source"]
    stdin_text = payload["stdin"]
    if type(source) is not str or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source is missing or exceeds the sandbox limit")
    if type(stdin_text) is not str or len(stdin_text.encode("utf-8")) > MAX_STDIN_BYTES:
        raise ValueError("stdin is missing or exceeds the sandbox limit")
    return payload


def _environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/work",
        "TMPDIR": "/work/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GOCACHE": "/work/.gocache",
        "GOPATH": "/work/.gopath",
        "GOMODCACHE": "/work/.gomodcache",
        "GOTOOLCHAIN": "local",
        "GOMAXPROCS": "2",
    }


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _run_bounded(
    command: list[str],
    *,
    stdin_bytes: bytes,
    timeout_seconds: float,
) -> dict[str, object]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=WORK,
        env=_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _terminate(process)
        raise RuntimeError("sandbox process pipes were not initialized")
    try:
        process.stdin.write(stdin_bytes)
        process.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    output = {"stdout": bytearray(), "stderr": bytearray()}
    timed_out = False
    output_limited = False
    while selector.get_map():
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            timed_out = True
            _terminate(process)
            break
        events = selector.select(min(remaining, 0.05))
        if not events and process.poll() is not None:
            events = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
        for key, _ in events:
            chunk = os.read(key.fd, 4096)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            target = output[str(key.data)]
            target.extend(chunk)
            if len(target) > MAX_OUTPUT_BYTES:
                del target[MAX_OUTPUT_BYTES:]
                output_limited = True
                _terminate(process)
                break
        if output_limited:
            break

    if timed_out or output_limited:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            _terminate(process)
    else:
        process.wait(timeout=1)
    for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
        try:
            remainder = stream.read(MAX_OUTPUT_BYTES - len(output[name]))
        except OSError:
            remainder = b""
        output[name].extend(remainder)
    duration_ms = int((time.monotonic() - started) * 1000)
    return {
        "exit_code": process.returncode,
        "stdout_b64": base64.b64encode(bytes(output["stdout"])).decode("ascii"),
        "stderr_b64": base64.b64encode(bytes(output["stderr"])).decode("ascii"),
        "timed_out": timed_out,
        "output_limited": output_limited,
        "duration_ms": duration_ms,
    }


def _response(payload: dict[str, object]) -> bytes:
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES * 3:
        raise ValueError("sandbox response exceeds transport limit")
    return encoded


def main() -> int:
    try:
        request = _read_request()
        language = str(request["language"])
        config = LANGUAGES[language]
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "tmp").mkdir(exist_ok=True)
        source_path = WORK / str(config["filename"])
        source_path.write_text(str(request["source"]), encoding="utf-8", newline="\n")

        compile_command = config["compile"]
        if compile_command is not None:
            compile_result = _run_bounded(
                list(compile_command),
                stdin_bytes=b"",
                timeout_seconds=COMPILE_TIMEOUT_SECONDS,
            )
            if (
                compile_result["exit_code"] != 0
                or compile_result["timed_out"]
                or compile_result["output_limited"]
            ):
                sys.stdout.buffer.write(
                    _response(
                        {
                            "protocol": PROTOCOL,
                            "language": language,
                            "status": "compile_failed",
                            "phase": "compile",
                            **compile_result,
                        }
                    )
                )
                return 0

        run_result = _run_bounded(
            list(config["run"]),
            stdin_bytes=str(request["stdin"]).encode("utf-8"),
            timeout_seconds=RUN_TIMEOUT_SECONDS,
        )
        status = "passed"
        if run_result["timed_out"]:
            status = "timed_out"
        elif run_result["output_limited"]:
            status = "output_limited"
        elif run_result["exit_code"] != 0:
            status = "runtime_failed"
        sys.stdout.buffer.write(
            _response(
                {
                    "protocol": PROTOCOL,
                    "language": language,
                    "status": status,
                    "phase": "run",
                    **run_result,
                }
            )
        )
        return 0
    except Exception as error:
        sys.stdout.buffer.write(
            _response(
                {
                    "protocol": PROTOCOL,
                    "language": None,
                    "status": "invalid_request",
                    "phase": "request",
                    "exit_code": None,
                    "stdout_b64": "",
                    "stderr_b64": base64.b64encode(type(error).__name__.encode()).decode("ascii"),
                    "timed_out": False,
                    "output_limited": False,
                    "duration_ms": 0,
                }
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
