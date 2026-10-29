from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcnlt.hf_tokenizer import (
    HuggingFaceBaseTokenizer,
    identity_from_directory,
    resolve_tokenizer_source,
)
from bcnlt.tokenizer import BCNLTTokenizer
from bcnlt.tokenizer_manifest import build_manifest
from tests.tokenizer_fixtures import write_byte_level_tokenizer


def _build_tokenizer(tmp_path: Path) -> tuple[BCNLTTokenizer, Path]:
    base_path = write_byte_level_tokenizer(tmp_path / "base")
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    identity = identity_from_directory(base_path, name="fixture/base", revision="local-v1")
    return BCNLTTokenizer(base, build_manifest(base, identity)), base_path


def test_save_and_reload_preserve_ids_and_artifact_contract(tmp_path: Path):
    tokenizer, base_path = _build_tokenizer(tmp_path)
    artifact = tmp_path / "artifact"

    files = tokenizer.save_pretrained(artifact)
    reloaded = BCNLTTokenizer.from_pretrained(artifact, base_path)

    assert {Path(path).name for path in files} == {
        "bcnlt_tokenizer_manifest.json",
        "bcnlt-tokenizer-v1.tokens.json",
    }
    assert not list(artifact.glob("*.safetensors"))
    assert not list(artifact.glob("*.bin"))
    assert reloaded.manifest == tokenizer.manifest
    assert reloaded.encode_text("hello world") == tokenizer.encode_text("hello world")
    assert reloaded.decode(reloaded.encode_text("hello world")) == "hello world"


def test_fast_base_tokenizer_returns_exact_unicode_character_offsets(tmp_path: Path):
    _, base_path = _build_tokenizer(tmp_path)
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    text = "Aé🧙Z"

    ids, offsets = base.encode_with_offsets(text)

    assert len(ids) == len(offsets)
    assert all(text[start:end] for start, end in offsets if end > start)
    unique_spans = sorted({(start, end) for start, end in offsets if end > start})
    assert "".join(text[start:end] for start, end in unique_spans) == text


def test_reload_rejects_corrupted_artifact_manifest(tmp_path: Path):
    tokenizer, base_path = _build_tokenizer(tmp_path)
    artifact = tmp_path / "artifact"
    tokenizer.save_pretrained(artifact)
    manifest_path = artifact / "bcnlt_tokenizer_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["base_vocab_size"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        BCNLTTokenizer.from_pretrained(artifact, base_path)


def test_reload_rejects_a_different_base_tokenizer(tmp_path: Path):
    tokenizer, _ = _build_tokenizer(tmp_path)
    artifact = tmp_path / "artifact"
    tokenizer.save_pretrained(artifact)
    other_base = write_byte_level_tokenizer(
        tmp_path / "other-base",
        corpus="completely different tokens and training material",
    )

    with pytest.raises(ValueError, match="base tokenizer|base vocabulary"):
        BCNLTTokenizer.from_pretrained(artifact, other_base)


def test_save_refuses_to_mix_with_existing_files(tmp_path: Path):
    tokenizer, _ = _build_tokenizer(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        tokenizer.save_pretrained(artifact)

    assert (artifact / "unrelated.txt").read_text(encoding="utf-8") == "keep"


def test_hub_tokenizer_source_requires_an_explicit_revision():
    with pytest.raises(ValueError, match="revision"):
        resolve_tokenizer_source("organization/model", revision=None)


def test_hub_tokenizer_source_resolves_revision_to_immutable_commit(
    tmp_path: Path, monkeypatch
):
    import huggingface_hub

    snapshot = write_byte_level_tokenizer(tmp_path / "snapshot")
    calls: dict[str, object] = {}

    class FakeApi:
        def model_info(self, repo_id: str, revision: str):
            calls["model_info"] = (repo_id, revision)
            return type("ModelInfo", (), {"sha": "a" * 40})()

    def fake_snapshot_download(**kwargs):
        calls["snapshot_download"] = kwargs
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: FakeApi())
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    resolved, commit = resolve_tokenizer_source(
        "organization/model", revision="release-v1"
    )

    assert resolved == snapshot
    assert commit == "a" * 40
    assert calls["model_info"] == ("organization/model", "release-v1")
    assert calls["snapshot_download"]["revision"] == "a" * 40
    assert calls["snapshot_download"]["repo_id"] == "organization/model"
