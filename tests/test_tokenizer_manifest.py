from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bcnlt.events import CURRENT_EVENT_VALUES
from bcnlt.tokenizer_manifest import (
    BaseTokenizerIdentity,
    build_manifest,
    load_manifest,
    write_manifest,
)


class FakeBaseTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self._vocab = vocab or {"a": 0, "b": 1, "c": 2, "<unk>": 3}
        self.vocab_size = len(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._vocab.get(char, self._vocab["<unk>"]) for char in text]

    def decode(self, ids: list[int]) -> str:
        inverse = {value: key for key, value in self._vocab.items()}
        return "".join(inverse[item] for item in ids)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def _identity() -> BaseTokenizerIdentity:
    return BaseTokenizerIdentity.from_files(
        name="fixture/base",
        revision="abc123",
        files={"tokenizer.json": b'{"fixture":true}', "tokenizer_config.json": b"{}"},
    )


def _manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    write_manifest(path, build_manifest(FakeBaseTokenizer(), _identity()))
    return path


def _rewrite(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_manifest_appends_contiguous_unique_ids_after_the_base_vocabulary():
    base = FakeBaseTokenizer()
    manifest = build_manifest(base, _identity())
    ids = [item.id for item in manifest.added_tokens]
    tokens = [item.token for item in manifest.added_tokens]

    assert ids[0] == base.vocab_size
    assert ids == list(range(base.vocab_size, base.vocab_size + len(ids)))
    assert len(tokens) == len(set(tokens))
    assert len(ids) == len(set(ids))


def test_manifest_contains_exactly_one_token_for_every_semantic_event():
    manifest = build_manifest(FakeBaseTokenizer(), _identity())
    event_tokens = {
        item.token
        for item in manifest.added_tokens
        if item.token.startswith("<|bcnlt:event:")
    }

    assert event_tokens == {
        f"<|bcnlt:event:{event.lower()}|>" for event in CURRENT_EVENT_VALUES
    }


def test_manifest_and_file_identity_are_byte_stable(tmp_path: Path):
    first = build_manifest(FakeBaseTokenizer(), _identity())
    second = build_manifest(FakeBaseTokenizer(), _identity())
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_manifest(first_path, first)
    write_manifest(second_path, second)

    assert first == second
    assert first.manifest_sha256 == second.manifest_sha256
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")
    assert [item.path for item in first.base_tokenizer.files] == [
        "tokenizer.json",
        "tokenizer_config.json",
    ]


def test_build_rejects_structural_token_collision_with_base_vocabulary():
    base = FakeBaseTokenizer(
        {"a": 0, "b": 1, "<unk>": 2, "<|bcnlt:stream:begin|>": 3}
    )

    with pytest.raises(ValueError, match="collides"):
        build_manifest(base, _identity())


def test_load_rejects_wrong_base_vocabulary(tmp_path: Path):
    path = _manifest_path(tmp_path)
    other = FakeBaseTokenizer({"a": 0, "different": 1, "c": 2, "<unk>": 3})

    with pytest.raises(ValueError, match="base vocabulary"):
        load_manifest(path, other, _identity())


def test_load_requires_a_verified_base_source_identity(tmp_path: Path):
    path = _manifest_path(tmp_path)

    with pytest.raises(ValueError, match="identity"):
        load_manifest(path, FakeBaseTokenizer(), None)  # type: ignore[arg-type]


def test_load_rejects_changed_base_source_identity(tmp_path: Path):
    path = _manifest_path(tmp_path)
    changed = BaseTokenizerIdentity.from_files(
        name="fixture/base",
        revision="abc123",
        files={"tokenizer.json": b'{"fixture":false}', "tokenizer_config.json": b"{}"},
    )

    with pytest.raises(ValueError, match="source hash"):
        load_manifest(path, FakeBaseTokenizer(), changed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["added_tokens"].__setitem__(
                1,
                {
                    "token": payload["added_tokens"][1]["token"],
                    "id": payload["added_tokens"][0]["id"],
                },
            ),
            "duplicate",
        ),
        (lambda payload: payload["added_tokens"].pop(), "required structural tokens"),
        (lambda payload: payload.__setitem__("schema_version", "future-v9"), "schema"),
        (
            lambda payload: payload["added_tokens"][2].__setitem__(
                "id", payload["added_tokens"][2]["id"] + 1000
            ),
            "contiguous",
        ),
    ],
)
def test_load_rejects_invalid_manifest_structure(tmp_path: Path, mutation, message: str):
    path = _manifest_path(tmp_path)
    _rewrite(path, mutation)

    with pytest.raises(ValueError, match=message):
        load_manifest(path, FakeBaseTokenizer(), _identity())


def test_load_rejects_changed_digest(tmp_path: Path):
    path = _manifest_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["base_vocab_size"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        load_manifest(path, FakeBaseTokenizer(), _identity())
