from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from bcnlt import tokenizer_cli
from tests.tokenizer_fixtures import write_byte_level_tokenizer


def _run(*args: str, check: bool = True):
    return subprocess.run(
        [sys.executable, "-m", "bcnlt.tokenizer_cli", *args],
        text=True,
        capture_output=True,
        check=check,
    )


def test_cli_build_inspect_encode_and_verify_local_artifact(tmp_path: Path):
    base = write_byte_level_tokenizer(tmp_path / "base")
    artifact = tmp_path / "artifact"

    built = _run(
        "build",
        "--base",
        str(base),
        "--output",
        str(artifact),
        "--name",
        "fixture/base",
        "--revision",
        "local-v1",
    )
    inspected = _run("inspect", "--artifact", str(artifact))
    encoded = _run(
        "encode",
        "--artifact",
        str(artifact),
        "--base",
        str(base),
        "--mode",
        "text",
        "--text",
        "hello world",
    )
    verified = _run("verify", "--artifact", str(artifact), "--base", str(base))

    assert json.loads(built.stdout)["manifest_sha256"]
    assert json.loads(inspected.stdout)["schema_version"] == "bcnlt-tokenizer-v1"
    assert json.loads(inspected.stdout)["added_token_count"] > 20
    assert json.loads(encoded.stdout)["decoded"] == "hello world"
    assert json.loads(encoded.stdout)["ids"]
    assert json.loads(verified.stdout)["verified"] is True


def test_cli_semantic_encode_accepts_a_python_file(tmp_path: Path):
    base = write_byte_level_tokenizer(tmp_path / "base")
    artifact = tmp_path / "artifact"
    source = tmp_path / "sample.py"
    source.write_text("answer = 42\n", encoding="utf-8")
    _run(
        "build",
        "--base",
        str(base),
        "--output",
        str(artifact),
        "--name",
        "fixture/base",
        "--revision",
        "local-v1",
    )

    result = _run(
        "encode",
        "--artifact",
        str(artifact),
        "--base",
        str(base),
        "--mode",
        "semantic_code",
        "--language",
        "python",
        "--file",
        str(source),
    )
    payload = json.loads(result.stdout)

    assert payload["mode"] == "semantic_code"
    assert payload["record_count"] >= 2
    assert '"event":"DEFINE"' in payload["decoded"]


def test_cli_returns_nonzero_for_corrupted_artifact(tmp_path: Path):
    base = write_byte_level_tokenizer(tmp_path / "base")
    artifact = tmp_path / "artifact"
    _run(
        "build",
        "--base",
        str(base),
        "--output",
        str(artifact),
        "--name",
        "fixture/base",
        "--revision",
        "local-v1",
    )
    manifest = artifact / "bcnlt_tokenizer_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    result = _run(
        "verify",
        "--artifact",
        str(artifact),
        "--base",
        str(base),
        check=False,
    )

    assert result.returncode != 0
    assert "verification failed" in result.stderr.lower()


def test_cli_returns_nonzero_for_base_mismatch(tmp_path: Path):
    base = write_byte_level_tokenizer(tmp_path / "base")
    other = write_byte_level_tokenizer(tmp_path / "other", corpus="different base corpus")
    artifact = tmp_path / "artifact"
    _run(
        "build",
        "--base",
        str(base),
        "--output",
        str(artifact),
        "--name",
        "fixture/base",
        "--revision",
        "local-v1",
    )

    result = _run(
        "verify",
        "--artifact",
        str(artifact),
        "--base",
        str(other),
        check=False,
    )

    assert result.returncode != 0
    assert "base tokenizer" in result.stderr.lower() or "base vocabulary" in result.stderr.lower()


def test_cli_build_uses_resolved_hub_snapshot_and_commit(
    tmp_path: Path, monkeypatch, capsys
):
    snapshot = write_byte_level_tokenizer(tmp_path / "snapshot")
    artifact = tmp_path / "artifact"
    calls: list[tuple[str, str | None]] = []

    def fake_resolve(source: str, *, revision: str | None):
        calls.append((source, revision))
        return snapshot, "b" * 40

    monkeypatch.setattr(tokenizer_cli, "resolve_tokenizer_source", fake_resolve)

    result = tokenizer_cli._build(
        Namespace(
            base="organization/model",
            output=str(artifact),
            name="organization/model",
            revision="release-v1",
        )
    )
    output = json.loads(capsys.readouterr().out)
    manifest = json.loads(
        (artifact / "bcnlt_tokenizer_manifest.json").read_text(encoding="utf-8")
    )

    assert result == 0
    assert calls == [("organization/model", "release-v1")]
    assert output["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["base_tokenizer"]["revision"] == "b" * 40
