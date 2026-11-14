"""Seal and verify the exact local NVIDIA safety-judge checkpoint."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .immutable_artifacts import publish_new_files

SAFETY_JUDGE_RUNTIME_SCHEMA = "bosscrafts-safety-judge-runtime-v1"
SAFETY_JUDGE_REPOSITORY = "nvidia/Nemotron-3-Content-Safety"
SAFETY_JUDGE_REVISION = "b47f6f6c18dc7d11a786abbbca2510cba58ede20"
SAFETY_JUDGE_LICENSE_NAME = "nvidia-nemotron-open-model-license"
SAFETY_JUDGE_LICENSE_URL = (
    "https://www.nvidia.com/en-us/agreements/enterprise-software/"
    "nvidia-nemotron-open-model-license/"
)
SAFETY_JUDGE_RUNTIME_VERSIONS = {
    "torch": "2.8.0",
    "transformers": "4.57.1",
}
EXPECTED_SAFETY_JUDGE_RUNTIME_MANIFEST_SHA256 = (
    "435dfcc22bfbb580275f8240d1de0e167f18e8ffa3c3f9dc11a94543b4f8031d"
)
MAX_SAFETY_JUDGE_MANIFEST_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SafetyJudgeFilePin:
    size: int
    sha256: str


@dataclass(frozen=True)
class SafetyJudgeRuntime:
    root: Path
    repository: str
    revision: str
    manifest_sha256: str
    total_bytes: int


SAFETY_JUDGE_FILE_PINS: dict[str, SafetyJudgeFilePin] = {
    "README.md": SafetyJudgeFilePin(
        17_879,
        "559967dc72348c714c3177ea3f803d47dfff7571d0bd9cf939ff2f2ea0bdef54",
    ),
    "added_tokens.json": SafetyJudgeFilePin(
        35,
        "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
    ),
    "bias.md": SafetyJudgeFilePin(
        891,
        "06c079b0869c2c4e9ba2d1657a3c46bf1b03b8d08e4bb939814554ed223464df",
    ),
    "chat_template.jinja": SafetyJudgeFilePin(
        3_736,
        "ca936922569db4cb430d90da426b7c40e442b866622eec9b0eb9d85a269a7e48",
    ),
    "config.json": SafetyJudgeFilePin(
        2_524,
        "0e99733a7ea77b2477492986380e249f8329991a71761f6420d7c48c371530df",
    ),
    "explainability.md": SafetyJudgeFilePin(
        5_320,
        "933615aa69da44547dfc1eaac892ef7a701d90b76271da8dd9e8183a57ed75ea",
    ),
    "generation_config.json": SafetyJudgeFilePin(
        210,
        "3f6c815fb4d217ffc4690d2898e24b2cee99dab9b878203b628371af4dca1b36",
    ),
    "model-00001-of-00005.safetensors": SafetyJudgeFilePin(
        839_692_288,
        "8e8d52233767a77543cdbd3eddb871b932596b7dbd6e4bbb159aec0287ce2734",
    ),
    "model-00002-of-00005.safetensors": SafetyJudgeFilePin(
        1_992_693_784,
        "6ec50e70eb58a27655419361a1869b4c1e35e4dad1aee74f97e6b87179bc2942",
    ),
    "model-00003-of-00005.safetensors": SafetyJudgeFilePin(
        1_992_547_688,
        "790973e71a83ab2340ecbcf57f1f7073987e9121212809ce0903ce781588cbda",
    ),
    "model-00004-of-00005.safetensors": SafetyJudgeFilePin(
        1_971_556_896,
        "37456b6c164cc48193ab80c8486e566db7df4b141c36fa2d454657c88a157257",
    ),
    "model-00005-of-00005.safetensors": SafetyJudgeFilePin(
        1_803_786_096,
        "195fc063a927e282ff988635ad1c561d6deba0318879821b4bef66952e54ba7d",
    ),
    "model.safetensors.index.json": SafetyJudgeFilePin(
        90_594,
        "07ad14190b2cfd4c9347b6e501ec783aaf42780c21393f0d71140a61157217d9",
    ),
    "preprocessor_config.json": SafetyJudgeFilePin(
        570,
        "f688d6bb20c5017601c4011de7ca656da8485b540b05013efdaf986c0fcc918d",
    ),
    "privacy.md": SafetyJudgeFilePin(
        4_057,
        "95a66031fa9be6e16d8298a2e9285f74e275a1929668a68c526924b3bad9fe9c",
    ),
    "processor_config.json": SafetyJudgeFilePin(
        70,
        "3ffd5f11778dc73e2b69b3c00535e4121e1badf7018136263cd17b5b34fbaa53",
    ),
    "safety.md": SafetyJudgeFilePin(
        2_361,
        "6a3e6ba32255a23e6365d015b2d35f5e9b5601df3a68fbd81ec30a303b91d106",
    ),
    "special_tokens_map.json": SafetyJudgeFilePin(
        662,
        "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
    ),
    "tokenizer.json": SafetyJudgeFilePin(
        33_384_568,
        "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
    ),
    "tokenizer.model": SafetyJudgeFilePin(
        4_689_074,
        "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
    ),
    "tokenizer_config.json": SafetyJudgeFilePin(
        1_155_387,
        "983c80895e7b188911f5ccfbb4f780a4e73fb3131ed2080a3e847ef8623cfdca",
    ),
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_linklike(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (is_junction is not None and bool(is_junction()))


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safety judge manifest key: {key}")
        result[key] = value
    return result


def _validate_contract(
    *,
    repository: str,
    revision: str,
    license_name: str,
    license_url: str,
    runtime_versions: Mapping[str, str],
    file_pins: Mapping[str, SafetyJudgeFilePin],
) -> None:
    if (
        not repository
        or _REVISION.fullmatch(revision) is None
        or not license_name
        or not license_url.startswith("https://")
        or set(runtime_versions) != {"torch", "transformers"}
        or any(
            type(value) is not str or not value
            for value in runtime_versions.values()
        )
        or not file_pins
        or len(file_pins) > 128
    ):
        raise ValueError("safety judge runtime contract is invalid")
    for relative, pin in file_pins.items():
        path = PurePosixPath(relative)
        if (
            type(relative) is not str
            or not relative
            or relative == "manifest.json"
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or type(pin) is not SafetyJudgeFilePin
            or type(pin.size) is not int
            or pin.size < 1
            or _SHA256.fullmatch(pin.sha256) is None
        ):
            raise ValueError("safety judge file pin is invalid")


def _manifest_body(
    *,
    repository: str,
    revision: str,
    license_name: str,
    license_url: str,
    runtime_versions: Mapping[str, str],
    file_pins: Mapping[str, SafetyJudgeFilePin],
) -> dict[str, object]:
    _validate_contract(
        repository=repository,
        revision=revision,
        license_name=license_name,
        license_url=license_url,
        runtime_versions=runtime_versions,
        file_pins=file_pins,
    )
    return {
        "schema_version": SAFETY_JUDGE_RUNTIME_SCHEMA,
        "repository": repository,
        "revision": revision,
        "license_name": license_name,
        "license_url": license_url,
        "runtime_versions": dict(sorted(runtime_versions.items())),
        "files": {
            relative: {"size": pin.size, "sha256": pin.sha256}
            for relative, pin in sorted(file_pins.items())
        },
        "total_bytes": sum(pin.size for pin in file_pins.values()),
    }


def _expected_manifest(
    *,
    repository: str,
    revision: str,
    license_name: str,
    license_url: str,
    runtime_versions: Mapping[str, str],
    file_pins: Mapping[str, SafetyJudgeFilePin],
) -> dict[str, object]:
    body = _manifest_body(
        repository=repository,
        revision=revision,
        license_name=license_name,
        license_url=license_url,
        runtime_versions=runtime_versions,
        file_pins=file_pins,
    )
    return {**body, "manifest_sha256": _sha256_bytes(_canonical(body))}


def _verify_files(root: Path, file_pins: Mapping[str, SafetyJudgeFilePin]) -> None:
    actual: set[str] = set()
    expected_directories = {
        PurePosixPath(*parts[:index]).as_posix()
        for relative in file_pins
        for parts in (PurePosixPath(relative).parts,)
        for index in range(1, len(parts))
    }
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if _is_linklike(path):
            raise ValueError("safety judge runtime contains a symbolic link")
        if path.is_dir():
            if relative not in expected_directories:
                raise ValueError("safety judge runtime file inventory is invalid")
        elif path.is_file() and relative != "manifest.json":
            actual.add(relative)
        elif not path.is_file():
            raise ValueError("safety judge runtime file inventory is invalid")
    if actual != set(file_pins):
        raise ValueError("safety judge runtime file inventory is invalid")
    for relative, pin in file_pins.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        try:
            if path.is_symlink() or path.stat().st_size != pin.size:
                raise ValueError("safety judge runtime file integrity mismatch")
            digest = _sha256_file(path)
        except OSError as error:
            raise ValueError("safety judge runtime file is unreadable") from error
        if digest != pin.sha256:
            raise ValueError("safety judge runtime file integrity mismatch")


def _runtime_root(output_root: str | Path) -> Path:
    candidate = Path(output_root)
    if _is_linklike(candidate):
        raise ValueError("safety judge runtime root is invalid")
    root = candidate.resolve()
    if not root.is_dir():
        raise ValueError("safety judge runtime root is invalid")
    return root


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        content = path.read_bytes()
        if not content or len(content) > MAX_SAFETY_JUDGE_MANIFEST_BYTES:
            raise ValueError("safety judge runtime manifest size is invalid")
        value = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("safety judge runtime manifest is unreadable") from error
    if type(value) is not dict:
        raise ValueError("safety judge runtime manifest is invalid")
    return value


def _seal_pinned_safety_judge_runtime(
    output_root: str | Path,
    *,
    repository: str,
    revision: str,
    license_name: str,
    license_url: str,
    runtime_versions: Mapping[str, str],
    file_pins: Mapping[str, SafetyJudgeFilePin],
) -> SafetyJudgeRuntime:
    root = _runtime_root(output_root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise ValueError("safety judge runtime manifest already exists")
    expected = _expected_manifest(
        repository=repository,
        revision=revision,
        license_name=license_name,
        license_url=license_url,
        runtime_versions=runtime_versions,
        file_pins=file_pins,
    )
    _verify_files(root, file_pins)
    content = (
        json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    publish_new_files(
        ((manifest_path, content),), artifact_label="safety judge runtime manifest"
    )
    return _verify_pinned_safety_judge_runtime(
        root,
        repository=repository,
        revision=revision,
        license_name=license_name,
        license_url=license_url,
        runtime_versions=runtime_versions,
        file_pins=file_pins,
    )


def _verify_pinned_safety_judge_runtime(
    output_root: str | Path,
    *,
    repository: str,
    revision: str,
    license_name: str,
    license_url: str,
    runtime_versions: Mapping[str, str],
    file_pins: Mapping[str, SafetyJudgeFilePin],
) -> SafetyJudgeRuntime:
    root = _runtime_root(output_root)
    expected = _expected_manifest(
        repository=repository,
        revision=revision,
        license_name=license_name,
        license_url=license_url,
        runtime_versions=runtime_versions,
        file_pins=file_pins,
    )
    if _read_manifest(root / "manifest.json") != expected:
        raise ValueError("safety judge runtime manifest integrity mismatch")
    _verify_files(root, file_pins)
    return SafetyJudgeRuntime(
        root=root,
        repository=repository,
        revision=revision,
        manifest_sha256=str(expected["manifest_sha256"]),
        total_bytes=sum(pin.size for pin in file_pins.values()),
    )


def seal_safety_judge_runtime(output_root: str | Path) -> SafetyJudgeRuntime:
    """Seal an exact already-downloaded NVIDIA safety-judge snapshot."""
    runtime = _seal_pinned_safety_judge_runtime(
        output_root,
        repository=SAFETY_JUDGE_REPOSITORY,
        revision=SAFETY_JUDGE_REVISION,
        license_name=SAFETY_JUDGE_LICENSE_NAME,
        license_url=SAFETY_JUDGE_LICENSE_URL,
        runtime_versions=SAFETY_JUDGE_RUNTIME_VERSIONS,
        file_pins=SAFETY_JUDGE_FILE_PINS,
    )
    if runtime.manifest_sha256 != EXPECTED_SAFETY_JUDGE_RUNTIME_MANIFEST_SHA256:
        raise ValueError("safety judge runtime expected identity mismatch")
    return runtime


def verify_safety_judge_runtime(output_root: str | Path) -> SafetyJudgeRuntime:
    """Rehash every local safety-judge byte and verify exact provenance."""
    runtime = _verify_pinned_safety_judge_runtime(
        output_root,
        repository=SAFETY_JUDGE_REPOSITORY,
        revision=SAFETY_JUDGE_REVISION,
        license_name=SAFETY_JUDGE_LICENSE_NAME,
        license_url=SAFETY_JUDGE_LICENSE_URL,
        runtime_versions=SAFETY_JUDGE_RUNTIME_VERSIONS,
        file_pins=SAFETY_JUDGE_FILE_PINS,
    )
    if runtime.manifest_sha256 != EXPECTED_SAFETY_JUDGE_RUNTIME_MANIFEST_SHA256:
        raise ValueError("safety judge runtime expected identity mismatch")
    return runtime
