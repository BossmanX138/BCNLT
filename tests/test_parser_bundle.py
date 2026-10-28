from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bcnlt.parsers.bundle import (
    REQUIRED_GRAMMARS,
    ParserBundleArtifact,
    ParserBundleEntry,
    ParserBundleManifest,
    assert_clean_parse_tree,
    load_parser_bundle,
    verify_parser_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> ParserBundleManifest:
    artifacts = []
    for platform, payload in (("windows-x86_64", b"windows"), ("linux-x86_64", b"linux")):
        path = tmp_path / platform / "language-pack.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        artifacts.append(
            ParserBundleArtifact(
                platform=platform,
                relative_path=path.relative_to(tmp_path).as_posix(),
                size=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    entries = tuple(
        ParserBundleEntry(
            grammar_id=grammar_id,
            repository=f"https://github.com/tree-sitter/tree-sitter-{grammar_id}",
            revision=hashlib.sha1(grammar_id.encode()).hexdigest(),
            abi_version=15,
            artifacts=tuple(artifacts),
        )
        for grammar_id in REQUIRED_GRAMMARS
    )
    return ParserBundleManifest.create(
        package_versions=(
            ("tree-sitter", "0.26.0"),
            ("tree-sitter-language-pack", "1.22.1"),
        ),
        entries=entries,
    )


def test_manifest_requires_all_14_grammars_and_source_shas(tmp_path: Path):
    manifest = _manifest(tmp_path)

    assert tuple(entry.grammar_id for entry in manifest.entries) == REQUIRED_GRAMMARS
    assert all(len(entry.revision) == 40 for entry in manifest.entries)
    assert all(entry.abi_version > 0 for entry in manifest.entries)
    assert manifest.max_source_bytes == 200 * 1024
    assert manifest.parse_timeout_ms == 500

    with pytest.raises(ValueError, match="exactly the required 14 grammars"):
        ParserBundleManifest.create(
            package_versions=manifest.package_versions,
            entries=manifest.entries[:-1],
        )
    with pytest.raises(ValueError, match="40-character"):
        ParserBundleManifest.create(
            package_versions=manifest.package_versions,
            entries=(replace(manifest.entries[0], revision="main"), *manifest.entries[1:]),
        )


def test_manifest_records_distinct_windows_and_linux_artifacts(tmp_path: Path):
    manifest = _manifest(tmp_path)

    for entry in manifest.entries:
        by_platform = {artifact.platform: artifact for artifact in entry.artifacts}
        assert set(by_platform) == {"windows-x86_64", "linux-x86_64"}
        assert by_platform["windows-x86_64"].sha256 != by_platform["linux-x86_64"].sha256


def test_manifest_round_trip_and_digest_are_canonical(tmp_path: Path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    loaded = load_parser_bundle(path)

    assert loaded == manifest
    assert loaded.manifest_sha256 == manifest.compute_digest()


def test_verify_parser_bundle_detects_one_altered_byte(tmp_path: Path):
    manifest = _manifest(tmp_path)
    verify_parser_bundle(manifest, tmp_path, "windows-x86_64")
    target = tmp_path / manifest.entries[0].artifacts[0].relative_path
    target.write_bytes(target.read_bytes() + b"x")

    with pytest.raises(ValueError, match="size|SHA-256"):
        verify_parser_bundle(manifest, tmp_path, "windows-x86_64")


class _Node:
    def __init__(self, node_type: str, children: tuple["_Node", ...] = ()) -> None:
        self.type = node_type
        self.children = children


@pytest.mark.parametrize("bad_type", ["ERROR", "MISSING"])
def test_recursive_error_and_missing_diagnostics_fail(bad_type: str):
    root = _Node("module", (_Node("function", (_Node(bad_type),)),))

    with pytest.raises(ValueError, match=bad_type):
        assert_clean_parse_tree(root)


def test_clean_parse_tree_accepts_nested_valid_nodes():
    assert_clean_parse_tree(_Node("module", (_Node("function", (_Node("identifier"),)),)))
