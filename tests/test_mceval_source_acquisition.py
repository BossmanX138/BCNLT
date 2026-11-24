from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bosscrafts_model.category_case_sources import MCEVAL_LANGUAGE_FILES
from bosscrafts_model.mceval_source_acquisition import (
    acquire_mceval_generation_sources,
)

REVISION = "1" * 40


def _row(language: str) -> bytes:
    value = {
        "task_id": f"{language}/1",
        "prompt": "function f() {",
        "canonical_solution": "return 1; }",
        "test": "check(f);",
        "entry_point": "f",
        "signature": "f()",
        "docstring": "return one",
        "instruction": "return one",
        "level": "easy",
    }
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def test_acquires_exact_evaluation_only_snapshot(tmp_path: Path) -> None:
    by_filename = {
        filename: _row(language) for language, filename in MCEVAL_LANGUAGE_FILES.items()
    }

    def fetch(url: str) -> bytes:
        return by_filename[url.rsplit("/", 1)[-1].replace("%23", "#")]

    output = tmp_path / "mceval"
    result = acquire_mceval_generation_sources(
        output,
        revision=REVISION,
        fetch=fetch,
    )
    manifest_bytes = (output / "source-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt = json.loads((output / "acquisition-receipt.json").read_bytes())

    assert result.source_file_count == 14
    assert result.source_row_count == 14
    assert result.source_manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert manifest["revision"] == REVISION
    assert manifest["license"] == "CC-BY-SA-4.0"
    assert set(manifest["languages"]) == set(MCEVAL_LANGUAGE_FILES)
    assert receipt["evaluation_only"] is True
    assert receipt["training_ready"] is False
    assert receipt["source_manifest_sha256"] == result.source_manifest_sha256
    assert all((output / filename).is_file() for filename in by_filename)


def test_rejects_invalid_or_mutable_acquisition_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="revision"):
        acquire_mceval_generation_sources(
            tmp_path / "bad-revision",
            revision="main",
            fetch=lambda _: b"ignored",
        )

    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "existing").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        acquire_mceval_generation_sources(
            output,
            revision=REVISION,
            fetch=lambda _: b"ignored",
        )


def test_rejects_malformed_source_without_publishing(tmp_path: Path) -> None:
    output = tmp_path / "malformed"
    with pytest.raises(ValueError, match="fields are invalid"):
        acquire_mceval_generation_sources(
            output,
            revision=REVISION,
            fetch=lambda _: b'{"task_id":"only"}\n',
        )
    assert not output.exists() or not any(output.iterdir())
