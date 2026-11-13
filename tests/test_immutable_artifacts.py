from __future__ import annotations

import os
from pathlib import Path

import pytest

from bosscrafts_model import immutable_artifacts


def test_immutable_publication_never_clobbers_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "evidence.json"
    destination.write_bytes(b"preserved")

    with pytest.raises(ValueError, match="already exists"):
        immutable_artifacts.publish_new_files(
            ((destination, b"replacement"),), artifact_label="fixture"
        )

    assert destination.read_bytes() == b"preserved"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_immutable_publication_rolls_back_owned_files_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.jsonl"
    manifest = tmp_path / "manifest.json"
    real_link = os.link
    calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("deliberate publication failure")
        real_link(source, destination)

    monkeypatch.setattr(immutable_artifacts.os, "link", fail_second_link)
    with pytest.raises(OSError, match="deliberate publication failure"):
        immutable_artifacts.publish_new_files(
            ((first, b"row\n"), (manifest, b"{}\n")), artifact_label="fixture"
        )

    assert list(tmp_path.iterdir()) == []
