"""Install and verify the exact evaluator used for frozen IFEval cases."""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from .immutable_artifacts import publish_new_files

IFEVAL_RUNTIME_SCHEMA = "bosscrafts-ifeval-runtime-v1"
IFEVAL_EVALUATOR_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
NLTK_DATA_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
EXPECTED_IFEVAL_RUNTIME_MANIFEST_SHA256 = (
    "44f4d751b203a2c8330542e00d8abb418e270f00697c2a4ba5badfd18abd6712"
)
MAX_REMOTE_FILE_BYTES = 64 * 1024 * 1024
MAX_RUNTIME_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_FILES = 4_096
MAX_ARCHIVE_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_PACKAGE_INIT = (
    b'"""Pinned Google Research IFEval evaluator package."""\n'
)


@dataclass(frozen=True)
class RemoteFilePin:
    url: str
    sha256: str


@dataclass(frozen=True)
class RemoteArchivePin:
    url: str
    sha256: str
    required_root: str
    destination: str


@dataclass(frozen=True)
class IfevalRuntime:
    root: Path
    manifest_sha256: str
    evaluator_revision: str
    nltk_revision: str


_OFFICIAL_FILE_PINS: dict[str, RemoteFilePin] = {
    "instruction_following_eval/evaluation_lib.py": RemoteFilePin(
        url=(
            "https://raw.githubusercontent.com/google-research/google-research/"
            f"{IFEVAL_EVALUATOR_REVISION}/instruction_following_eval/"
            "evaluation_lib.py"
        ),
        sha256="35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
    ),
    "instruction_following_eval/instructions.py": RemoteFilePin(
        url=(
            "https://raw.githubusercontent.com/google-research/google-research/"
            f"{IFEVAL_EVALUATOR_REVISION}/instruction_following_eval/instructions.py"
        ),
        sha256="60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    ),
    "instruction_following_eval/instructions_registry.py": RemoteFilePin(
        url=(
            "https://raw.githubusercontent.com/google-research/google-research/"
            f"{IFEVAL_EVALUATOR_REVISION}/instruction_following_eval/"
            "instructions_registry.py"
        ),
        sha256="ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    ),
    "instruction_following_eval/instructions_util.py": RemoteFilePin(
        url=(
            "https://raw.githubusercontent.com/google-research/google-research/"
            f"{IFEVAL_EVALUATOR_REVISION}/instruction_following_eval/"
            "instructions_util.py"
        ),
        sha256="a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
    ),
    "instruction_following_eval/requirements.txt": RemoteFilePin(
        url=(
            "https://raw.githubusercontent.com/google-research/google-research/"
            f"{IFEVAL_EVALUATOR_REVISION}/instruction_following_eval/requirements.txt"
        ),
        sha256="1b1716c9ac21b9cc2bd0d7354ed8f7b6982ccf35598529601a7c08508acec94a",
    ),
    "instruction_following_eval/README.md": RemoteFilePin(
        url=(
            "https://raw.githubusercontent.com/google-research/google-research/"
            f"{IFEVAL_EVALUATOR_REVISION}/instruction_following_eval/README.md"
        ),
        sha256="fef3c44408e4b39837af345095ccce7da798853fd60ea94a1de63afa9d3d67b3",
    ),
    "LICENSES/google-research-Apache-2.0.txt": RemoteFilePin(
        url=(
            "https://raw.githubusercontent.com/google-research/google-research/"
            f"{IFEVAL_EVALUATOR_REVISION}/LICENSE"
        ),
        sha256="cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
}

_OFFICIAL_ARCHIVE_PINS: dict[str, RemoteArchivePin] = {
    "punkt": RemoteArchivePin(
        url=(
            "https://raw.githubusercontent.com/nltk/nltk_data/"
            f"{NLTK_DATA_REVISION}/packages/tokenizers/punkt.zip"
        ),
        sha256="51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec",
        required_root="punkt",
        destination="nltk_data/tokenizers",
    ),
    "punkt_tab": RemoteArchivePin(
        url=(
            "https://raw.githubusercontent.com/nltk/nltk_data/"
            f"{NLTK_DATA_REVISION}/packages/tokenizers/punkt_tab.zip"
        ),
        sha256="e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
        required_root="punkt_tab",
        destination="nltk_data/tokenizers",
    ),
}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate IFEval runtime manifest key: {key}")
        result[key] = value
    return result


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise ValueError(f"{label} path is unsafe")
    return path


def _validate_pins(
    file_pins: Mapping[str, RemoteFilePin],
    archive_pins: Mapping[str, RemoteArchivePin],
) -> None:
    if not file_pins or not archive_pins:
        raise ValueError("IFEval runtime source pins are incomplete")
    destinations: set[str] = {"instruction_following_eval/__init__.py"}
    for destination, file_pin in file_pins.items():
        _safe_relative_path(destination, "remote file")
        if destination in destinations:
            raise ValueError("IFEval runtime source destination is duplicated")
        destinations.add(destination)
        if not file_pin.url or _SHA256.fullmatch(file_pin.sha256) is None:
            raise ValueError("IFEval runtime remote file pin is invalid")
    for name, archive_pin in archive_pins.items():
        if _SOURCE_NAME.fullmatch(name) is None:
            raise ValueError("IFEval runtime archive name is invalid")
        _safe_relative_path(archive_pin.destination, "archive destination")
        _safe_relative_path(archive_pin.required_root, "archive root")
        if len(PurePosixPath(archive_pin.required_root).parts) != 1:
            raise ValueError("IFEval runtime archive root is invalid")
        if not archive_pin.url or _SHA256.fullmatch(archive_pin.sha256) is None:
            raise ValueError("IFEval runtime archive pin is invalid")


def _extract_archive(
    content: bytes,
    *,
    pin: RemoteArchivePin,
) -> dict[str, bytes]:
    extracted: dict[str, bytes] = {}
    total_size = 0
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_ARCHIVE_FILES:
                raise ValueError("IFEval runtime archive file count is invalid")
            for info in infos:
                member = _safe_relative_path(info.filename, "archive")
                if member.parts[0] != pin.required_root:
                    raise ValueError("IFEval runtime archive root is invalid")
                if info.is_dir():
                    continue
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if info.flag_bits & 1 or file_type not in {0, stat.S_IFREG}:
                    raise ValueError("IFEval runtime archive member type is unsafe")
                if info.file_size < 0 or info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise ValueError("IFEval runtime archive member size is invalid")
                total_size += info.file_size
                if total_size > MAX_ARCHIVE_EXPANDED_BYTES:
                    raise ValueError("IFEval runtime archive expansion is too large")
                value = archive.read(info)
                if len(value) != info.file_size:
                    raise ValueError("IFEval runtime archive member is truncated")
                destination = PurePosixPath(pin.destination, *member.parts).as_posix()
                if destination in extracted:
                    raise ValueError("IFEval runtime archive path is duplicated")
                extracted[destination] = value
    except (zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError("IFEval runtime archive is invalid") from error
    if not extracted:
        raise ValueError("IFEval runtime archive contains no files")
    return extracted


def _source_contracts(
    file_pins: Mapping[str, RemoteFilePin],
    archive_pins: Mapping[str, RemoteArchivePin],
) -> dict[str, object]:
    return {
        "files": {
            destination: {"url": pin.url, "sha256": pin.sha256}
            for destination, pin in file_pins.items()
        },
        "archives": {
            name: {
                "url": pin.url,
                "sha256": pin.sha256,
                "required_root": pin.required_root,
                "destination": pin.destination,
                "archive_file": f"sources/{name}.zip",
            }
            for name, pin in archive_pins.items()
        },
    }


def _runtime_contracts(contents: Mapping[str, bytes]) -> dict[str, object]:
    return {
        path: {"bytes": len(content), "sha256": _sha256(content)}
        for path, content in sorted(contents.items())
    }


def _fetch_official_bytes(url: str) -> bytes:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "raw.githubusercontent.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("IFEval runtime source URL is not an allowed pinned URL")
    # The complete URL was matched against the immutable HTTPS pin inventory
    # above; redirects are disabled by the dedicated opener below.
    request = urllib.request.Request(  # noqa: S310
        url,
        headers={"User-Agent": "BossCrafts-IFEval-Runtime/1"},
    )
    opener = urllib.request.build_opener(_NoRedirectHandler())
    try:
        with opener.open(request, timeout=60) as response:
            if response.geturl() != url:
                raise ValueError("IFEval runtime source redirected")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_REMOTE_FILE_BYTES:
                raise ValueError("IFEval runtime source is too large")
            content = bytes(response.read(MAX_REMOTE_FILE_BYTES + 1))
    except (OSError, urllib.error.URLError) as error:
        raise ValueError("IFEval runtime source download failed") from error
    if not content or len(content) > MAX_REMOTE_FILE_BYTES:
        raise ValueError("IFEval runtime source size is invalid")
    return content


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        content = path.read_bytes()
        if not content or len(content) > MAX_RUNTIME_MANIFEST_BYTES:
            raise ValueError("IFEval runtime manifest size is invalid")
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("IFEval runtime manifest is unreadable") from error
    if type(payload) is not dict:
        raise ValueError("IFEval runtime manifest must be an object")
    return payload


def _install_pinned_ifeval_runtime(
    output_root: str | Path,
    *,
    evaluator_revision: str,
    nltk_revision: str,
    file_pins: Mapping[str, RemoteFilePin],
    archive_pins: Mapping[str, RemoteArchivePin],
    fetch_bytes: Callable[[str], bytes],
) -> IfevalRuntime:
    if (
        _GIT_REVISION.fullmatch(evaluator_revision) is None
        or _GIT_REVISION.fullmatch(nltk_revision) is None
    ):
        raise ValueError("IFEval runtime revision is invalid")
    _validate_pins(file_pins, archive_pins)
    root = Path(output_root).resolve()
    if root.exists():
        raise ValueError("IFEval runtime output root already exists")
    contents: dict[str, bytes] = {
        "instruction_following_eval/__init__.py": _PACKAGE_INIT
    }
    for destination, file_pin in file_pins.items():
        content = fetch_bytes(file_pin.url)
        if not content or len(content) > MAX_REMOTE_FILE_BYTES:
            raise ValueError("IFEval runtime source size is invalid")
        if _sha256(content) != file_pin.sha256:
            raise ValueError("IFEval runtime source SHA-256 mismatch")
        contents[destination] = content
    for name, archive_pin in archive_pins.items():
        content = fetch_bytes(archive_pin.url)
        if not content or len(content) > MAX_REMOTE_FILE_BYTES:
            raise ValueError("IFEval runtime archive size is invalid")
        if _sha256(content) != archive_pin.sha256:
            raise ValueError("IFEval runtime archive SHA-256 mismatch")
        archive_path = f"sources/{name}.zip"
        contents[archive_path] = content
        for path, value in _extract_archive(content, pin=archive_pin).items():
            if path in contents:
                raise ValueError("IFEval runtime archive output is duplicated")
            contents[path] = value
    unsigned: dict[str, object] = {
        "schema_version": IFEVAL_RUNTIME_SCHEMA,
        "evaluator_revision": evaluator_revision,
        "nltk_revision": nltk_revision,
        "sources": _source_contracts(file_pins, archive_pins),
        "runtime_files": _runtime_contracts(contents),
    }
    manifest = {**unsigned, "manifest_sha256": _sha256(_canonical(unsigned))}
    manifest_content = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    publication = tuple(
        (root.joinpath(*PurePosixPath(path).parts), content)
        for path, content in sorted(contents.items())
    ) + ((root / "manifest.json", manifest_content),)
    publish_new_files(publication, artifact_label="IFEval runtime")
    return _verify_pinned_ifeval_runtime(
        root,
        evaluator_revision=evaluator_revision,
        nltk_revision=nltk_revision,
        file_pins=file_pins,
        archive_pins=archive_pins,
    )


def _verify_pinned_ifeval_runtime(
    output_root: str | Path,
    *,
    evaluator_revision: str,
    nltk_revision: str,
    file_pins: Mapping[str, RemoteFilePin],
    archive_pins: Mapping[str, RemoteArchivePin],
) -> IfevalRuntime:
    _validate_pins(file_pins, archive_pins)
    root = Path(output_root).resolve()
    manifest = _read_manifest(root / "manifest.json")
    required_fields = {
        "schema_version",
        "evaluator_revision",
        "nltk_revision",
        "sources",
        "runtime_files",
        "manifest_sha256",
    }
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        set(manifest) != required_fields
        or manifest.get("schema_version") != IFEVAL_RUNTIME_SCHEMA
        or manifest.get("evaluator_revision") != evaluator_revision
        or manifest.get("nltk_revision") != nltk_revision
        or manifest.get("sources") != _source_contracts(file_pins, archive_pins)
        or manifest.get("manifest_sha256") != _sha256(_canonical(unsigned))
    ):
        raise ValueError("IFEval runtime manifest integrity mismatch")
    expected_contents: dict[str, bytes | None] = {
        "instruction_following_eval/__init__.py": _PACKAGE_INIT,
        **{destination: None for destination in file_pins},
    }
    for name, pin in archive_pins.items():
        archive_path = root / "sources" / f"{name}.zip"
        try:
            archive_content = archive_path.read_bytes()
        except OSError as error:
            raise ValueError("IFEval runtime archive is unreadable") from error
        if _sha256(archive_content) != pin.sha256:
            raise ValueError("IFEval runtime archive SHA-256 mismatch")
        expected_contents[f"sources/{name}.zip"] = archive_content
        for path, content in _extract_archive(archive_content, pin=pin).items():
            if path in expected_contents:
                raise ValueError("IFEval runtime expected path is duplicated")
            expected_contents[path] = content
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != {*expected_contents, "manifest.json"}:
        raise ValueError("IFEval runtime file inventory is invalid")
    observed_contents: dict[str, bytes] = {}
    for relative, expected in expected_contents.items():
        file_path = root.joinpath(*PurePosixPath(relative).parts)
        if file_path.is_symlink():
            raise ValueError("IFEval runtime file type is unsafe")
        try:
            content = file_path.read_bytes()
        except OSError as error:
            raise ValueError("IFEval runtime file is unreadable") from error
        if expected is not None and content != expected:
            raise ValueError("IFEval runtime file integrity mismatch")
        file_pin = file_pins.get(relative)
        if file_pin is not None and _sha256(content) != file_pin.sha256:
            raise ValueError("IFEval runtime source SHA-256 mismatch")
        observed_contents[relative] = content
    if manifest.get("runtime_files") != _runtime_contracts(observed_contents):
        raise ValueError("IFEval runtime file SHA-256 contract mismatch")
    return IfevalRuntime(
        root=root,
        manifest_sha256=str(manifest["manifest_sha256"]),
        evaluator_revision=evaluator_revision,
        nltk_revision=nltk_revision,
    )


def install_ifeval_runtime(output_root: str | Path) -> IfevalRuntime:
    """Download and immutably install the exact audited evaluator runtime."""
    return _install_pinned_ifeval_runtime(
        output_root,
        evaluator_revision=IFEVAL_EVALUATOR_REVISION,
        nltk_revision=NLTK_DATA_REVISION,
        file_pins=_OFFICIAL_FILE_PINS,
        archive_pins=_OFFICIAL_ARCHIVE_PINS,
        fetch_bytes=_fetch_official_bytes,
    )


def verify_ifeval_runtime(output_root: str | Path) -> IfevalRuntime:
    """Revalidate every installed evaluator and NLTK data byte."""
    return _verify_pinned_ifeval_runtime(
        output_root,
        evaluator_revision=IFEVAL_EVALUATOR_REVISION,
        nltk_revision=NLTK_DATA_REVISION,
        file_pins=_OFFICIAL_FILE_PINS,
        archive_pins=_OFFICIAL_ARCHIVE_PINS,
    )
