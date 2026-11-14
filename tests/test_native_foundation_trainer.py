from __future__ import annotations

import json
from pathlib import Path

from bcnlt.native_foundation.trainer import _atomic_write_json, train_byte_bpe_tokenizer


def test_trainer_progress_write_retries_windows_sharing_violation(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "progress.json"
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(path: Path, target: Path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(5, "sharing violation")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    _atomic_write_json(destination, {"status": "running"})

    assert attempts == 2
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "running"}


def test_trainer_builds_exact_reloadable_lossless_vocabulary(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(
            f"function_{index}(value_{index}) {{ return value_{index} + {index}; }} café 東京 🚀"
            for index in range(800)
        ),
        encoding="utf-8",
    )
    structural = tmp_path / "structural.json"
    structural_token = "<|bcnlt:stream:begin|>"
    structural.write_text(
        json.dumps({"schema_version": "test", "tokens": [structural_token]}),
        encoding="utf-8",
    )
    output = tmp_path / "tokenizer"

    manifest = train_byte_bpe_tokenizer(
        corpus_files=(corpus,),
        structural_tokens_path=structural,
        output_dir=output,
        vocabulary_size=512,
        min_frequency=1,
        threads=2,
        label="test-control",
    )

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    assert tokenizer.get_vocab_size() == 512
    sample = "NUL:\x00 accents:café scripts:東京 emoji:🚀\nline two"
    encoded = tokenizer.encode(sample)
    assert tokenizer.decode(encoded.ids, skip_special_tokens=False) == sample
    structural_encoding = tokenizer.encode(structural_token)
    assert len(structural_encoding.ids) == 1
    assert manifest["status"] == "complete"
    assert manifest["vocabulary_size"] == 512
    assert manifest["tokenizer_sha256"]

