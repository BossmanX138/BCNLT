from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from bcnlt.adaptive_vocab.lineage import ExperimentLineage, prepare_output_root


def _file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _capture(tmp_path: Path) -> ExperimentLineage:
    base_model = tmp_path / "base"
    _file(base_model / "config.json", b"base-config")
    _file(base_model / "weights" / "model.safetensors", b"weights")
    return ExperimentLineage.capture(
        config_path=_file(tmp_path / "adaptive.json", b"adaptive"),
        source_config_path=_file(tmp_path / "sources.json", b"sources"),
        corpus_manifest_path=_file(tmp_path / "corpus.json", b"corpus"),
        v1_manifest_path=_file(tmp_path / "v1.json", b"v1"),
        v4_gate_path=_file(tmp_path / "v4.json", b"v4"),
        base_model_path=base_model,
        parser_bundle_path=_file(tmp_path / "parser.json", b"parser"),
        code_revision="a" * 40,
    )


def test_lineage_captures_exact_lowercase_file_and_directory_hashes(tmp_path: Path):
    lineage = _capture(tmp_path)
    artifacts = {artifact.name: artifact for artifact in lineage.artifacts}

    assert artifacts["config"].sha256 == hashlib.sha256(b"adaptive").hexdigest()
    assert artifacts["config"].path == str((tmp_path / "adaptive.json").resolve())
    assert artifacts["base_model"].kind == "directory"
    assert all(
        artifact.sha256 == artifact.sha256.lower() and len(artifact.sha256) == 64
        for artifact in lineage.artifacts
    )
    assert lineage.identity_sha256 == lineage.identity_sha256.lower()
    assert lineage.canonical_bytes().endswith(b"\n")


def test_changed_input_changes_lineage_identity(tmp_path: Path):
    original = _capture(tmp_path)
    (tmp_path / "v4.json").write_bytes(b"changed-v4")
    changed = ExperimentLineage.capture(
        config_path=tmp_path / "adaptive.json",
        source_config_path=tmp_path / "sources.json",
        corpus_manifest_path=tmp_path / "corpus.json",
        v1_manifest_path=tmp_path / "v1.json",
        v4_gate_path=tmp_path / "v4.json",
        base_model_path=tmp_path / "base",
        parser_bundle_path=tmp_path / "parser.json",
        code_revision="a" * 40,
    )

    assert original.identity_sha256 != changed.identity_sha256
    assert {item.name: item.sha256 for item in original.artifacts}["v4_gate"] != {
        item.name: item.sha256 for item in changed.artifacts
    }["v4_gate"]


@pytest.mark.parametrize("relationship", ["equal", "inside", "contains"])
def test_output_root_rejects_every_protected_path_overlap(
    tmp_path: Path, relationship: str
):
    protected = tmp_path / "preserved"
    protected.mkdir()
    if relationship == "equal":
        output = protected
    elif relationship == "inside":
        output = protected / "candidate"
    else:
        output = tmp_path

    with pytest.raises(ValueError, match="overlaps preserved artifact"):
        prepare_output_root(output, [protected])


def test_output_root_resolves_symlink_aliases_before_overlap_check(tmp_path: Path):
    protected = tmp_path / "preserved"
    protected.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(protected, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ValueError, match="overlaps preserved artifact"):
        prepare_output_root(alias / "candidate", [protected])


@pytest.mark.skipif(os.name != "nt", reason="Windows/WSL alias syntax is Windows-specific")
def test_output_root_resolves_wsl_drive_alias_before_overlap_check(tmp_path: Path):
    protected = tmp_path / "preserved"
    protected.mkdir()
    drive = protected.drive.rstrip(":").lower()
    tail = protected.resolve().as_posix().split(":", 1)[1]
    wsl_alias = Path(f"/mnt/{drive}{tail}")

    with pytest.raises(ValueError, match="overlaps preserved artifact"):
        prepare_output_root(wsl_alias / "candidate", [protected])


def test_output_root_rejects_existing_nonempty_directory(tmp_path: Path):
    output = tmp_path / "candidate"
    _file(output / "partial.json", b"partial")

    with pytest.raises(FileExistsError, match="new path"):
        prepare_output_root(output, [])


def test_output_root_cleans_temporary_directory_when_atomic_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "candidate"

    def fail_rename(self: Path, target: Path):
        raise OSError("simulated interruption")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="simulated interruption"):
        prepare_output_root(output, [])

    assert not output.exists()
    assert not list(tmp_path.glob(".candidate.tmp-*"))


def test_output_root_is_new_empty_and_resolved(tmp_path: Path):
    output = tmp_path / "runs" / "candidate"

    prepared = prepare_output_root(output, [])

    assert prepared == output.resolve()
    assert prepared.is_dir()
    assert not any(prepared.iterdir())
