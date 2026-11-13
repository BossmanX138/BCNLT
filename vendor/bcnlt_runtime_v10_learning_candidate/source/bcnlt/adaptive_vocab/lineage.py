from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LINEAGE_SCHEMA_VERSION = "bcnlt-adaptive-vocab-lineage-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LineageArtifact:
    name: str
    path: str
    kind: str
    sha256: str

    @classmethod
    def capture(cls, name: str, path: str | Path) -> LineageArtifact:
        resolved = Path(path).expanduser().resolve(strict=True)
        if resolved.is_file():
            kind = "file"
            digest = _file_sha256(resolved)
        elif resolved.is_dir():
            kind = "directory"
            digest = _directory_sha256(resolved)
        else:
            raise ValueError(f"lineage input is neither a file nor directory: {resolved}")
        return cls(name=name, path=str(resolved), kind=kind, sha256=digest)


@dataclass(frozen=True, slots=True)
class ExperimentLineage:
    schema_version: str
    code_revision: str
    artifacts: tuple[LineageArtifact, ...]

    @classmethod
    def capture(
        cls,
        *,
        config_path: str | Path,
        source_config_path: str | Path,
        corpus_manifest_path: str | Path,
        v1_manifest_path: str | Path,
        v4_gate_path: str | Path,
        base_model_path: str | Path,
        parser_bundle_path: str | Path,
        code_revision: str,
    ) -> ExperimentLineage:
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", code_revision):
            raise ValueError("code_revision must be a lowercase immutable Git hash")
        paths = (
            ("config", config_path),
            ("source_config", source_config_path),
            ("corpus_manifest", corpus_manifest_path),
            ("v1_manifest", v1_manifest_path),
            ("v4_gate", v4_gate_path),
            ("base_model", base_model_path),
            ("parser_bundle", parser_bundle_path),
        )
        return cls(
            schema_version=LINEAGE_SCHEMA_VERSION,
            code_revision=code_revision,
            artifacts=tuple(LineageArtifact.capture(name, path) for name, path in paths),
        )

    def _identity_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "code_revision": self.code_revision,
            "artifacts": [
                {"name": item.name, "kind": item.kind, "sha256": item.sha256}
                for item in self.artifacts
            ],
        }

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self._identity_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "code_revision": self.code_revision,
            "identity_sha256": self.identity_sha256,
            "artifacts": [
                {
                    "name": item.name,
                    "path": item.path,
                    "kind": item.kind,
                    "sha256": item.sha256,
                }
                for item in self.artifacts
            ],
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")


def _resolve_platform_path(path: str | Path) -> Path:
    raw = str(path).replace("\\", "/")
    if os.name == "nt":
        match = re.fullmatch(r"/mnt/([A-Za-z])(?:/(.*))?", raw)
        if match:
            tail = match.group(2) or ""
            raw = f"{match.group(1).upper()}:/{tail}"
    else:
        match = re.fullmatch(r"([A-Za-z]):(?:/(.*))?", raw)
        if match:
            tail = match.group(2) or ""
            raw = f"/mnt/{match.group(1).lower()}/{tail}"
    return Path(raw).expanduser().resolve(strict=False)


def _paths_overlap(left: Path, right: Path) -> bool:
    left_value = os.path.normcase(str(left))
    right_value = os.path.normcase(str(right))
    try:
        common = os.path.commonpath((left_value, right_value))
    except ValueError:
        return False
    return common in {left_value, right_value}


def prepare_output_root(output: Path, protected: Sequence[Path]) -> Path:
    resolved_output = _resolve_platform_path(output)
    resolved_protected = tuple(_resolve_platform_path(path) for path in protected)
    for preserved in resolved_protected:
        if _paths_overlap(resolved_output, preserved):
            raise ValueError(
                f"output overlaps preserved artifact: {resolved_output} and {preserved}"
            )
    if resolved_output.exists():
        raise FileExistsError(f"output root must be a new path: {resolved_output}")

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_name(
        f".{resolved_output.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    try:
        temporary.rename(resolved_output)
    finally:
        if temporary.exists():
            temporary.rmdir()
    return resolved_output
