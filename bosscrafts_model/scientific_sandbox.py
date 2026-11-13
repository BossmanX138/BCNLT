"""Disposable, attestable Docker execution for BCNLT scientific fixtures."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

PROTOCOL = "bosscrafts-bcnlt-sandbox-v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _PROJECT_ROOT / "sandbox" / "images.json"
_RUNNER = _PROJECT_ROOT / "sandbox" / "runner.py"
_MAX_SOURCE_BYTES = 64 * 1024
_MAX_STDIN_BYTES = 16 * 1024
_MAX_DOCKER_STDOUT_BYTES = 256 * 1024
_MAX_DOCKER_STDERR_BYTES = 64 * 1024
_MEMORY_BYTES = 512 * 1024 * 1024
_NANO_CPUS = 500_000_000
_PIDS_LIMIT = 64
_TMPFS = "rw,exec,nosuid,nodev,size=134217728,mode=1777"


class SandboxExecutionError(RuntimeError):
    """The host could not establish or complete an isolated execution."""


class _BoundedProcessResult(TypedDict):
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    output_limited: bool


@dataclass(frozen=True)
class SandboxAttestation:
    image_id: str
    image_manifest_sha256: str
    policy_sha256: str
    container_id: str
    network_mode: str
    read_only_rootfs: bool
    user: str
    cap_drop: tuple[str, ...]
    no_new_privileges: bool
    privileged: bool
    pids_limit: int
    memory_bytes: int
    memory_swap_bytes: int
    nano_cpus: int
    ipc_mode: str
    cgroupns_mode: str
    mount_count: int
    tmpfs: str
    log_driver: str

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "image_manifest_sha256": self.image_manifest_sha256,
            "policy_sha256": self.policy_sha256,
            "container_id": self.container_id,
            "network_mode": self.network_mode,
            "read_only_rootfs": self.read_only_rootfs,
            "user": self.user,
            "cap_drop": list(self.cap_drop),
            "no_new_privileges": self.no_new_privileges,
            "privileged": self.privileged,
            "pids_limit": self.pids_limit,
            "memory_bytes": self.memory_bytes,
            "memory_swap_bytes": self.memory_swap_bytes,
            "nano_cpus": self.nano_cpus,
            "ipc_mode": self.ipc_mode,
            "cgroupns_mode": self.cgroupns_mode,
            "mount_count": self.mount_count,
            "tmpfs": self.tmpfs,
            "log_driver": self.log_driver,
        }


@dataclass(frozen=True)
class SandboxExecutionResult:
    protocol: str
    language: str
    status: str
    phase: str
    exit_code: int | None
    stdout_bytes: bytes
    stderr_bytes: bytes
    timed_out: bool
    output_limited: bool
    duration_ms: int
    container_name: str
    request_sha256: str
    response_sha256: str
    attestation: SandboxAttestation

    @property
    def stdout_text(self) -> str:
        return self.stdout_bytes.decode("utf-8")

    @property
    def stderr_text(self) -> str:
        return self.stderr_bytes.decode("utf-8")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "language": self.language,
            "status": self.status,
            "phase": self.phase,
            "exit_code": self.exit_code,
            "stdout_b64": base64.b64encode(self.stdout_bytes).decode("ascii"),
            "stderr_b64": base64.b64encode(self.stderr_bytes).decode("ascii"),
            "timed_out": self.timed_out,
            "output_limited": self.output_limited,
            "duration_ms": self.duration_ms,
            "container_name": self.container_name,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "attestation": self.attestation.to_dict(),
        }


@dataclass(frozen=True)
class _ImageIdentity:
    name: str
    image_id: str
    repo_digest: str
    runner_sha256: str
    languages: frozenset[str]


class DockerScientificSandbox:
    """Run one request in one verified container and then destroy it."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        docker: str = "docker",
        host_timeout_seconds: float = 35.0,
    ) -> None:
        if host_timeout_seconds <= 0 or host_timeout_seconds > 60:
            raise ValueError("host timeout must be in (0, 60] seconds")
        self._docker = docker
        self._host_timeout_seconds = host_timeout_seconds
        self._manifest_path = Path(manifest_path).resolve()
        self._manifest_bytes = self._manifest_path.read_bytes()
        self._manifest_sha256 = hashlib.sha256(self._manifest_bytes).hexdigest()
        self._images = self._load_manifest(self._manifest_bytes)

    @classmethod
    def load_default(cls) -> "DockerScientificSandbox":
        return cls(_DEFAULT_MANIFEST)

    def execute(
        self,
        *,
        language: str,
        source: str,
        stdin_text: str,
    ) -> SandboxExecutionResult:
        image = self._image_for_language(language)
        if type(source) is not str or len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
            raise ValueError("source must be a string within the sandbox limit")
        if type(stdin_text) is not str or len(stdin_text.encode("utf-8")) > _MAX_STDIN_BYTES:
            raise ValueError("stdin must be a string within the sandbox limit")
        request_bytes = _canonical_json(
            {
                "protocol": PROTOCOL,
                "language": language,
                "source": source,
                "stdin": stdin_text,
            }
        )
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        container_name = f"bcnlt-science-{uuid.uuid4().hex}"
        container_id: str | None = None
        cleanup_error: str | None = None
        try:
            container_id = self._create_container(
                image=image,
                name=container_name,
                request_sha256=request_sha256,
            )
            inspection = self._inspect_container(container_id)
            attestation = self._attest(
                inspection,
                image=image,
                container_id=container_id,
                request_sha256=request_sha256,
            )
            completed = _run_bounded_process(
                [self._docker, "start", "--attach", "--interactive", container_id],
                input_bytes=request_bytes,
                timeout_seconds=self._host_timeout_seconds,
                max_stdout_bytes=_MAX_DOCKER_STDOUT_BYTES,
                max_stderr_bytes=_MAX_DOCKER_STDERR_BYTES,
            )
            if completed["timed_out"]:
                raise SandboxExecutionError("sandbox host transport timed out")
            if completed["output_limited"]:
                raise SandboxExecutionError("sandbox host transport exceeded output limit")
            if completed["returncode"] != 0:
                detail = completed["stderr"].decode("utf-8", errors="replace").strip()
                raise SandboxExecutionError(
                    "sandbox container could not complete" + (f": {detail}" if detail else "")
                )
            response_bytes = completed["stdout"]
            response = self._parse_response(response_bytes, language=language)
            return SandboxExecutionResult(
                protocol=PROTOCOL,
                language=language,
                status=response["status"],
                phase=response["phase"],
                exit_code=response["exit_code"],
                stdout_bytes=response["stdout_bytes"],
                stderr_bytes=response["stderr_bytes"],
                timed_out=response["timed_out"],
                output_limited=response["output_limited"],
                duration_ms=response["duration_ms"],
                container_name=container_name,
                request_sha256=request_sha256,
                response_sha256=hashlib.sha256(response_bytes).hexdigest(),
                attestation=attestation,
            )
        finally:
            if container_id is not None:
                try:
                    removed = subprocess.run(
                        [self._docker, "rm", "--force", container_id],
                        capture_output=True,
                        timeout=10,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except subprocess.TimeoutExpired:
                    cleanup_error = "docker rm timed out"
                except OSError as exc:
                    cleanup_error = f"docker rm could not start: {exc}"
                else:
                    if removed.returncode != 0:
                        cleanup_error = removed.stderr.decode(
                            "utf-8", errors="replace"
                        ).strip()
            if cleanup_error is not None:
                raise SandboxExecutionError(
                    "sandbox container cleanup failed" +
                    (f": {cleanup_error}" if cleanup_error else "")
                )

    def _load_manifest(self, raw: bytes) -> tuple[_ImageIdentity, ...]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("sandbox image manifest is not valid JSON") from error
        if type(payload) is not dict or set(payload) != {"schema", "protocol", "images"}:
            raise ValueError("sandbox image manifest keys mismatch")
        if payload["schema"] != "bosscrafts-bcnlt-sandbox-images-v1" or payload["protocol"] != PROTOCOL:
            raise ValueError("sandbox image manifest identity mismatch")
        if not isinstance(payload["images"], list) or not payload["images"]:
            raise ValueError("sandbox image manifest requires at least one image")
        runner_sha256 = hashlib.sha256(_RUNNER.read_bytes()).hexdigest()
        identities: list[_ImageIdentity] = []
        seen_languages: set[str] = set()
        for item in payload["images"]:
            if type(item) is not dict or set(item) != {
                "name", "image_id", "repo_digest", "runner_sha256", "languages"
            }:
                raise ValueError("sandbox image entry keys mismatch")
            image_id = item["image_id"]
            languages = item["languages"]
            if (
                not isinstance(image_id, str)
                or not image_id.startswith("sha256:")
                or len(image_id) != 71
                or any(character not in "0123456789abcdef" for character in image_id[7:])
            ):
                raise ValueError("sandbox image ID is invalid")
            if item["runner_sha256"] != runner_sha256:
                raise ValueError("sandbox runner bytes do not match the image manifest")
            if (
                not isinstance(languages, list)
                or not languages
                or not all(isinstance(value, str) and value for value in languages)
                or len(set(languages)) != len(languages)
                or seen_languages.intersection(languages)
            ):
                raise ValueError("sandbox image language routing is invalid")
            seen_languages.update(languages)
            observed = self._docker_control(
                ["image", "inspect", image_id, "--format", "{{.Id}}"]
            ).strip()
            if observed != image_id:
                raise ValueError("installed sandbox image does not match the manifest")
            identities.append(
                _ImageIdentity(
                    name=_required_string(item, "name"),
                    image_id=image_id,
                    repo_digest=_required_string(item, "repo_digest"),
                    runner_sha256=item["runner_sha256"],
                    languages=frozenset(languages),
                )
            )
        return tuple(identities)

    def _image_for_language(self, language: str) -> _ImageIdentity:
        if type(language) is not str:
            raise TypeError("language must be a string")
        for image in self._images:
            if language in image.languages:
                return image
        raise ValueError(f"language has no pinned sandbox image: {language}")

    def _create_container(
        self,
        *,
        image: _ImageIdentity,
        name: str,
        request_sha256: str,
    ) -> str:
        output = self._docker_control(
            [
                "create", "--interactive", "--name", name,
                "--label", "bosscrafts.bcnlt.sandbox=true",
                "--label", f"bosscrafts.bcnlt.request_sha256={request_sha256}",
                "--network", "none",
                "--read-only",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges=true",
                "--pids-limit", str(_PIDS_LIMIT),
                "--memory", str(_MEMORY_BYTES),
                "--memory-swap", str(_MEMORY_BYTES),
                "--cpus", "0.5",
                "--ulimit", "nofile=128:128",
                "--ulimit", "nproc=64:64",
                "--ulimit", "fsize=67108864:67108864",
                "--ulimit", "cpu=30:30",
                "--ipc", "none",
                "--cgroupns", "private",
                "--user", "65532:65532",
                "--tmpfs", f"/work:{_TMPFS}",
                "--log-driver", "none",
                "--restart", "no",
                "--stop-timeout", "1",
                image.image_id,
            ]
        ).strip()
        if len(output) != 64 or any(character not in "0123456789abcdef" for character in output):
            raise SandboxExecutionError("docker create returned an invalid container ID")
        return output

    def _inspect_container(self, container_id: str) -> dict[str, Any]:
        output = self._docker_control(["inspect", container_id])
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as error:
            raise SandboxExecutionError("docker inspect returned invalid JSON") from error
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise SandboxExecutionError("docker inspect response shape mismatch")
        return payload[0]

    def _attest(
        self,
        inspection: dict[str, Any],
        *,
        image: _ImageIdentity,
        container_id: str,
        request_sha256: str,
    ) -> SandboxAttestation:
        host = inspection.get("HostConfig")
        config = inspection.get("Config")
        mounts = inspection.get("Mounts")
        if not isinstance(host, dict) or not isinstance(config, dict) or not isinstance(mounts, list):
            raise SandboxExecutionError("docker inspection is missing policy fields")
        cap_drop = tuple(sorted(host.get("CapDrop") or ()))
        security = tuple(host.get("SecurityOpt") or ())
        no_new_privileges = any(
            isinstance(item, str) and item.split("=", 1)[0] == "no-new-privileges"
            for item in security
        )
        binds = host.get("Binds") or ()
        configured_mounts = host.get("Mounts") or ()
        mount_count = len(mounts) + len(binds) + len(configured_mounts)
        tmpfs = (host.get("Tmpfs") or {}).get("/work")
        log_driver = (host.get("LogConfig") or {}).get("Type")
        labels = config.get("Labels") or {}
        policy = {
            "image_id": inspection.get("Image"),
            "network_mode": host.get("NetworkMode"),
            "read_only_rootfs": host.get("ReadonlyRootfs"),
            "user": config.get("User"),
            "cap_drop": cap_drop,
            "no_new_privileges": no_new_privileges,
            "privileged": host.get("Privileged"),
            "pids_limit": host.get("PidsLimit"),
            "memory_bytes": host.get("Memory"),
            "memory_swap_bytes": host.get("MemorySwap"),
            "nano_cpus": host.get("NanoCpus"),
            "ipc_mode": host.get("IpcMode"),
            "cgroupns_mode": host.get("CgroupnsMode"),
            "mount_count": mount_count,
            "tmpfs": tmpfs,
            "log_driver": log_driver,
            "attach_stdin": config.get("AttachStdin"),
            "open_stdin": config.get("OpenStdin"),
            "device_count": len(host.get("Devices") or ()),
            "device_request_count": len(host.get("DeviceRequests") or ()),
            "port_binding_count": len(host.get("PortBindings") or {}),
            "restart_policy": (host.get("RestartPolicy") or {}).get("Name"),
            "sandbox_label": labels.get("bosscrafts.bcnlt.sandbox"),
            "request_label": labels.get("bosscrafts.bcnlt.request_sha256"),
        }
        expected = {
            "image_id": image.image_id,
            "network_mode": "none",
            "read_only_rootfs": True,
            "user": "65532:65532",
            "cap_drop": ("ALL",),
            "no_new_privileges": True,
            "privileged": False,
            "pids_limit": _PIDS_LIMIT,
            "memory_bytes": _MEMORY_BYTES,
            "memory_swap_bytes": _MEMORY_BYTES,
            "nano_cpus": _NANO_CPUS,
            "ipc_mode": "none",
            "cgroupns_mode": "private",
            "mount_count": 0,
            "tmpfs": _TMPFS,
            "log_driver": "none",
            "attach_stdin": True,
            "open_stdin": True,
            "device_count": 0,
            "device_request_count": 0,
            "port_binding_count": 0,
            "restart_policy": "no",
            "sandbox_label": "true",
            "request_label": request_sha256,
        }
        if policy != expected:
            differences = sorted(
                key for key in expected if policy.get(key) != expected[key]
            )
            raise SandboxExecutionError(
                "sandbox isolation policy mismatch: " + ", ".join(differences)
            )
        return SandboxAttestation(
            image_id=image.image_id,
            image_manifest_sha256=self._manifest_sha256,
            policy_sha256=hashlib.sha256(_canonical_json(policy)).hexdigest(),
            container_id=container_id,
            network_mode="none",
            read_only_rootfs=True,
            user="65532:65532",
            cap_drop=("ALL",),
            no_new_privileges=True,
            privileged=False,
            pids_limit=_PIDS_LIMIT,
            memory_bytes=_MEMORY_BYTES,
            memory_swap_bytes=_MEMORY_BYTES,
            nano_cpus=_NANO_CPUS,
            ipc_mode="none",
            cgroupns_mode="private",
            mount_count=0,
            tmpfs=_TMPFS,
            log_driver="none",
        )

    def _parse_response(self, raw: bytes, *, language: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SandboxExecutionError("sandbox response is not valid UTF-8 JSON") from error
        keys = {
            "protocol", "language", "status", "phase", "exit_code",
            "stdout_b64", "stderr_b64", "timed_out", "output_limited", "duration_ms",
        }
        if type(payload) is not dict or set(payload) != keys:
            raise SandboxExecutionError("sandbox response keys mismatch")
        if payload["protocol"] != PROTOCOL or payload["language"] != language:
            raise SandboxExecutionError("sandbox response identity mismatch")
        if payload["status"] not in {
            "passed", "compile_failed", "runtime_failed", "timed_out", "output_limited"
        } or payload["phase"] not in {"compile", "run"}:
            raise SandboxExecutionError("sandbox response status mismatch")
        if payload["exit_code"] is not None and type(payload["exit_code"]) is not int:
            raise SandboxExecutionError("sandbox response exit code is invalid")
        if type(payload["timed_out"]) is not bool or type(payload["output_limited"]) is not bool:
            raise SandboxExecutionError("sandbox response flags are invalid")
        if type(payload["duration_ms"]) is not int or payload["duration_ms"] < 0:
            raise SandboxExecutionError("sandbox response duration is invalid")
        try:
            stdout_bytes = base64.b64decode(payload["stdout_b64"], validate=True)
            stderr_bytes = base64.b64decode(payload["stderr_b64"], validate=True)
        except (TypeError, ValueError) as error:
            raise SandboxExecutionError("sandbox response output encoding is invalid") from error
        if len(stdout_bytes) > 64 * 1024 or len(stderr_bytes) > 64 * 1024:
            raise SandboxExecutionError("sandbox response output exceeds the limit")
        payload["stdout_bytes"] = stdout_bytes
        payload["stderr_bytes"] = stderr_bytes
        return payload

    def _docker_control(self, arguments: list[str]) -> str:
        try:
            completed = subprocess.run(
                [self._docker, *arguments],
                capture_output=True,
                timeout=20,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SandboxExecutionError("docker control command could not complete") from error
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise SandboxExecutionError(
                "docker control command failed" + (f": {detail}" if detail else "")
            )
        if len(completed.stdout) > _MAX_DOCKER_STDOUT_BYTES or len(completed.stderr) > _MAX_DOCKER_STDERR_BYTES:
            raise SandboxExecutionError("docker control output exceeds the host limit")
        try:
            return completed.stdout.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SandboxExecutionError("docker control output is not UTF-8") from error


def _run_bounded_process(
    command: list[str],
    *,
    input_bytes: bytes,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
) -> _BoundedProcessResult:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                with suppress(OSError):
                    stream.close()
        raise SandboxExecutionError("docker transport pipes are unavailable")
    stdout = bytearray()
    stderr = bytearray()
    limited = threading.Event()

    def drain(stream: Any, target: bytearray, limit: int) -> None:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            remaining = limit - len(target)
            target.extend(chunk[:max(0, remaining)])
            if len(chunk) > remaining:
                limited.set()
                return

    readers = (
        threading.Thread(target=drain, args=(process.stdout, stdout, max_stdout_bytes), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr, max_stderr_bytes), daemon=True),
    )
    try:
        for reader in readers:
            reader.start()
        try:
            process.stdin.write(input_bytes)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        deadline = time.monotonic() + timeout_seconds
        timed_out = False
        while process.poll() is None:
            if limited.is_set():
                process.kill()
                break
            if time.monotonic() >= deadline:
                timed_out = True
                process.kill()
                break
            time.sleep(0.01)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        for reader in readers:
            reader.join(timeout=2)
        if any(reader.is_alive() for reader in readers):
            raise SandboxExecutionError("docker transport readers did not terminate")
        return {
            "returncode": process.returncode,
            "stdout": bytes(stdout),
            "stderr": bytes(stderr),
            "timed_out": timed_out,
            "output_limited": limited.is_set(),
        }
    finally:
        if process.poll() is None:
            with suppress(OSError):
                process.kill()
            with suppress(OSError, subprocess.TimeoutExpired):
                process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                with suppress(OSError):
                    stream.close()
        for reader in readers:
            reader.join(timeout=2)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"sandbox image {key} must be a non-empty string")
    return value
