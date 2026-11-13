from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform as platform_module
import re
import shutil
import sys
import urllib.request
from urllib.parse import urlsplit
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol


REQUIRED_GRAMMARS = (
    "python",
    "javascript",
    "typescript",
    "c",
    "cpp",
    "c_sharp",
    "java",
    "go",
    "rust",
    "php",
    "ruby",
    "kotlin",
    "swift",
    "bash",
)
SCHEMA_VERSION = "bcnlt-parser-bundle-v1"
MAX_SOURCE_BYTES = 200 * 1024
PARSE_TIMEOUT_MS = 500
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_PROVENANCE_BYTES = 4 * 1024 * 1024
_PROVENANCE_PATH_PREFIX = "/xberg-io/tree-sitter-language-pack/releases/download/"


class ParseNode(Protocol):
    type: str
    children: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ParserBundleArtifact:
    platform: str
    relative_path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.platform or not self.relative_path:
            raise ValueError("Parser artifact platform and path are required")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Parser artifact path must remain inside the bundle")
        if self.size < 1:
            raise ValueError("Parser artifact size must be positive")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("Parser artifact SHA-256 must contain 64 lowercase hex characters")

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "relative_path": self.relative_path,
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ParserBundleEntry:
    grammar_id: str
    repository: str
    revision: str
    abi_version: int
    artifacts: tuple[ParserBundleArtifact, ...]

    def __post_init__(self) -> None:
        if self.grammar_id not in REQUIRED_GRAMMARS:
            raise ValueError(f"Unexpected grammar id: {self.grammar_id}")
        if not self.repository.startswith("https://github.com/"):
            raise ValueError(f"Grammar repository must be a GitHub HTTPS URL: {self.grammar_id}")
        if not _SHA40.fullmatch(self.revision):
            raise ValueError("Grammar revisions must be 40-character lowercase source SHAs")
        if self.abi_version < 13:
            raise ValueError(f"Unsupported Tree-sitter ABI for {self.grammar_id}")
        platforms = tuple(artifact.platform for artifact in self.artifacts)
        if len(platforms) != len(set(platforms)):
            raise ValueError(f"Duplicate platform artifact for {self.grammar_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "grammar_id": self.grammar_id,
            "repository": self.repository,
            "revision": self.revision,
            "abi_version": self.abi_version,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class ParserBundleManifest:
    schema_version: str
    package_versions: tuple[tuple[str, str], ...]
    entries: tuple[ParserBundleEntry, ...]
    max_source_bytes: int
    parse_timeout_ms: int
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported parser bundle schema")
        ids = tuple(entry.grammar_id for entry in self.entries)
        if ids != REQUIRED_GRAMMARS:
            raise ValueError("Parser bundle must contain exactly the required 14 grammars in order")
        if self.max_source_bytes != MAX_SOURCE_BYTES or self.parse_timeout_ms != PARSE_TIMEOUT_MS:
            raise ValueError("Parser resource limits do not match the BCNLT v1 contract")
        if not _SHA256.fullmatch(self.manifest_sha256):
            raise ValueError("Parser manifest SHA-256 must contain 64 lowercase hex characters")

    @classmethod
    def create(
        cls,
        *,
        package_versions: tuple[tuple[str, str], ...],
        entries: tuple[ParserBundleEntry, ...],
    ) -> "ParserBundleManifest":
        provisional = cls(
            schema_version=SCHEMA_VERSION,
            package_versions=tuple(sorted(package_versions)),
            entries=entries,
            max_source_bytes=MAX_SOURCE_BYTES,
            parse_timeout_ms=PARSE_TIMEOUT_MS,
            manifest_sha256="0" * 64,
        )
        return replace(provisional, manifest_sha256=provisional.compute_digest())

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_versions": {key: value for key, value in self.package_versions},
            "entries": [entry.to_dict() for entry in self.entries],
            "max_source_bytes": self.max_source_bytes,
            "parse_timeout_ms": self.parse_timeout_ms,
        }

    def compute_digest(self) -> str:
        return hashlib.sha256(_canonical_json(self._unsigned_dict())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned_dict(), "manifest_sha256": self.manifest_sha256}


def _manifest_from_dict(payload: dict[str, Any]) -> ParserBundleManifest:
    versions = payload.get("package_versions")
    if not isinstance(versions, dict):
        raise ValueError("Parser bundle package versions must be an object")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("Parser bundle entries must be a list")
    entries = []
    for raw in raw_entries:
        if not isinstance(raw, dict) or not isinstance(raw.get("artifacts"), list):
            raise ValueError("Malformed parser bundle entry")
        artifacts = tuple(ParserBundleArtifact(**item) for item in raw["artifacts"])
        entries.append(
            ParserBundleEntry(
                grammar_id=str(raw["grammar_id"]),
                repository=str(raw["repository"]),
                revision=str(raw["revision"]),
                abi_version=int(raw["abi_version"]),
                artifacts=artifacts,
            )
        )
    manifest = ParserBundleManifest(
        schema_version=str(payload.get("schema_version", "")),
        package_versions=tuple(sorted((str(k), str(v)) for k, v in versions.items())),
        entries=tuple(entries),
        max_source_bytes=int(payload.get("max_source_bytes", 0)),
        parse_timeout_ms=int(payload.get("parse_timeout_ms", 0)),
        manifest_sha256=str(payload.get("manifest_sha256", "")),
    )
    if manifest.manifest_sha256 != manifest.compute_digest():
        raise ValueError("Parser bundle manifest digest mismatch")
    return manifest


def load_parser_bundle(path: str | Path) -> ParserBundleManifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load parser bundle manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Parser bundle manifest must be a JSON object")
    return _manifest_from_dict(payload)


def verify_parser_bundle(
    manifest: ParserBundleManifest,
    root: str | Path,
    platform: str,
) -> None:
    if manifest.manifest_sha256 != manifest.compute_digest():
        raise ValueError("Parser bundle manifest digest mismatch")
    root_path = Path(root).resolve()
    for entry in manifest.entries:
        artifact = next((item for item in entry.artifacts if item.platform == platform), None)
        if artifact is None:
            raise ValueError(f"Parser bundle has no {platform} artifact for {entry.grammar_id}")
        path = (root_path / artifact.relative_path).resolve()
        try:
            path.relative_to(root_path)
        except ValueError as exc:
            raise ValueError("Parser artifact escapes the bundle root") from exc
        if not path.is_file():
            raise ValueError(f"Parser artifact is missing for {entry.grammar_id}: {path}")
        if path.stat().st_size != artifact.size:
            raise ValueError(f"Parser artifact size mismatch for {entry.grammar_id}")
        if _sha256(path) != artifact.sha256:
            raise ValueError(f"Parser artifact SHA-256 mismatch for {entry.grammar_id}")


def assert_clean_parse_tree(root: ParseNode) -> None:
    stack = [root]
    while stack:
        node = stack.pop()
        node_type = str(getattr(node, "type", ""))
        is_missing = bool(getattr(node, "is_missing", False))
        if node_type in {"ERROR", "MISSING"} or is_missing:
            diagnostic = "MISSING" if is_missing else node_type
            raise ValueError(f"Tree-sitter parse contains {diagnostic} diagnostic")
        stack.extend(reversed(tuple(getattr(node, "children", ()))))


def _platform_id() -> str:
    machine = platform_module.machine().lower()
    arch = "x86_64" if machine in {"amd64", "x86_64"} else machine
    if sys.platform == "win32":
        return f"windows-{arch}"
    if sys.platform.startswith("linux"):
        return f"linux-{arch}"
    raise ValueError(f"Unsupported parser bundle platform: {sys.platform}/{machine}")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load parser bundle configuration: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported parser bundle configuration schema")
    return payload


def _load_provenance(config: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    url = str(config.get("provenance_url", ""))
    expected = str(config.get("provenance_sha256", ""))
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or not parsed.path.startswith(_PROVENANCE_PATH_PREFIX)
        or not Path(parsed.path).name
    ):
        raise ValueError("Parser provenance URL is outside the pinned HTTPS release origin")
    if not _SHA256.fullmatch(expected):
        raise ValueError("Parser provenance SHA-256 is invalid")
    target = cache_root / Path(parsed.path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        too_large = False
        with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as output:
            remaining = MAX_PROVENANCE_BYTES + 1
            while remaining:
                block = response.read(min(1024 * 1024, remaining))
                if not block:
                    break
                output.write(block)
                remaining -= len(block)
            too_large = remaining == 0
        if too_large:
            target.unlink(missing_ok=True)
            raise ValueError("Parser provenance response exceeds the bounded download limit")
    if _sha256(target) != expected:
        target.unlink(missing_ok=True)
        raise ValueError("Parser provenance SHA-256 mismatch")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("languages"), dict):
        raise ValueError("Parser provenance payload is malformed")
    return payload


def _copy_content_addressed(source: Path, output_root: Path, runtime: Path) -> None:
    digest = _sha256(source)
    object_path = output_root / "objects" / "sha256" / digest / source.name
    object_path.parent.mkdir(parents=True, exist_ok=True)
    if not object_path.exists():
        shutil.copy2(source, object_path)
    runtime.parent.mkdir(parents=True, exist_ok=True)
    if runtime.exists():
        if _sha256(runtime) != digest:
            raise ValueError(f"Existing frozen parser artifact differs: {runtime}")
        return
    try:
        os.link(object_path, runtime)
    except OSError:
        shutil.copy2(object_path, runtime)


def freeze_parser_bundle(
    config: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
) -> ParserBundleManifest:
    config_payload = _load_config(Path(config))
    configured = config_payload.get("grammars")
    if not isinstance(configured, list):
        raise ValueError("Parser grammar configuration must be a list")
    grammar_ids = tuple(str(item.get("id")) for item in configured if isinstance(item, dict))
    if grammar_ids != REQUIRED_GRAMMARS:
        raise ValueError("Parser configuration must contain exactly the required 14 grammars")

    cache = Path(cache_root)
    output = Path(output_root)
    provenance = _load_provenance(config_payload, cache)
    provenance_languages = provenance["languages"]

    try:
        from tree_sitter_language_pack import (
            PackConfig,
            cache_dir as active_cache_dir,
            configure,
            download,
            get_language,
        )
    except ImportError as exc:
        raise ValueError("Install the pinned BCNLT multilingual parser dependencies") from exc

    configure(PackConfig(cache_dir=str(cache)))
    download_names = [str(item.get("pack_name", item["id"])) for item in configured]
    download(download_names)
    platform_id = _platform_id()
    package_version = importlib.metadata.version("tree-sitter-language-pack")
    library_root = Path(active_cache_dir())
    extension = ".dll" if sys.platform == "win32" else ".so"
    prefix = "" if sys.platform == "win32" else "lib"

    old_entries: dict[str, ParserBundleEntry] = {}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old_entries = {item.grammar_id: item for item in load_parser_bundle(manifest_path).entries}

    entries = []
    for item in configured:
        grammar_id = str(item["id"])
        provenance_id = str(item.get("provenance_id", grammar_id))
        source = provenance_languages.get(provenance_id)
        if not isinstance(source, dict) or not source.get("repo") or not source.get("rev"):
            raise ValueError(f"Grammar {grammar_id} has no source revision")
        revision = str(source["rev"])
        if not _SHA40.fullmatch(revision):
            raise ValueError(f"Grammar {grammar_id} has no 40-character source revision")
        cache_stem = str(item.get("cache_stem", grammar_id))
        library = library_root / f"{prefix}tree_sitter_{cache_stem}{extension}"
        if not library.is_file():
            raise ValueError(f"Downloaded parser library is missing: {library}")
        runtime = (
            output
            / "platforms"
            / platform_id
            / "tree-sitter-language-pack"
            / f"v{package_version}"
            / "libs"
            / library.name
        )
        _copy_content_addressed(library, output, runtime)
        artifact = ParserBundleArtifact(
            platform=platform_id,
            relative_path=runtime.relative_to(output).as_posix(),
            size=runtime.stat().st_size,
            sha256=_sha256(runtime),
        )
        previous = old_entries.get(grammar_id)
        artifacts = () if previous is None else tuple(
            existing for existing in previous.artifacts if existing.platform != platform_id
        )
        artifacts = tuple(sorted((*artifacts, artifact), key=lambda value: value.platform))
        language = get_language(str(item.get("pack_name", grammar_id)))
        entries.append(
            ParserBundleEntry(
                grammar_id=grammar_id,
                repository=str(source["repo"]),
                revision=revision,
                abi_version=int(language.abi_version),
                artifacts=artifacts,
            )
        )

    manifest = ParserBundleManifest.create(
        package_versions=(
            ("tree-sitter", importlib.metadata.version("tree-sitter")),
            ("tree-sitter-language-pack", package_version),
        ),
        entries=tuple(entries),
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_parser_bundle(manifest, output, platform_id)
    return manifest
