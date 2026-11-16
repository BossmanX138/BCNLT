from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from bosscrafts_model.resumable_task_packing import ResumableTaskDataset
from bosscrafts_model.reviewed_instruction_curriculum import (
    INSTRUCTION_RESPONSE_FAMILY,
    build_reviewed_instruction_curriculum,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _content_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _row(number: int, *, prompt: str, output: str, source: str = "fixture"):
    row = {
        "seed_id": f"seed:{number}",
        "instruction": prompt,
        "output": output,
        "constraints": [],
        "mutations": [],
        "source": source,
        "review_status": "approved",
        "provenance": {"dataset_id": source, "upstream_id": str(number)},
        "semantic_review": {"decision": "keep", "evidence": ["fixture"]},
    }
    row["id"] = _content_digest(row)
    return row


def _eval_row(
    start: int, *, split: str, prompt: str, output: str
) -> dict[str, object]:
    for number in range(start, start + 1000):
        row = _row(number, prompt=prompt, output=output)
        derived = "validation" if int(str(row["id"])[-16:], 16) % 2 == 0 else "test"
        if derived == split:
            return row
    raise AssertionError("unable to build deterministic eval fixture")


def _write_source(root: Path, train_rows: list[dict], eval_rows: list[dict]) -> None:
    root.mkdir()
    train = root / "train.json"
    evaluation = root / "eval.json"
    train.write_text(json.dumps(train_rows, indent=2) + "\n", encoding="utf-8")
    evaluation.write_text(
        json.dumps(eval_rows, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": "fixture.json",
        "source_sha256": "a" * 64,
        "accepted": len(train_rows) + len(eval_rows),
        "rejected": 0,
        "train": len(train_rows),
        "eval": len(eval_rows),
        "train_sha256": _content_digest(train_rows),
        "eval_sha256": _content_digest(eval_rows),
        "quality_review": "All rows carry prior semantic-review evidence.",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def test_cli_runs_from_project_root() -> None:
    project_root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_v10_reviewed_instruction_curriculum.py"),
            "--help",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-root" in result.stdout


def test_builds_text_mode_curriculum_and_rejects_cross_split_overlap(
    tmp_path: Path,
) -> None:
    binding = NativeTokenizerBinding.load()
    source = tmp_path / "source"
    train_rows = [
        _row(1, prompt="Train one", output="Answer one"),
        _row(2, prompt="Train two", output="Answer two"),
        _row(3, prompt="shared prompt", output="shared answer"),
    ]
    eval_rows = [
        _eval_row(
            4,
            split="validation",
            prompt="Validation only",
            output="Validation answer",
        ),
        _eval_row(1004, split="test", prompt="Test only", output="Test answer"),
        _row(6, prompt=" Shared   Prompt ", output="shared answer"),
    ]
    _write_source(source, train_rows, eval_rows)
    output = tmp_path / "curriculum.sqlite"

    result = build_reviewed_instruction_curriculum(
        binding,
        source_root=source,
        output_database=output,
        max_sequence=256,
    )

    assert result.example_count == 4
    assert result.split_counts == {"test": 1, "train": 2, "validation": 1}
    assert result.rejection_counts == {"cross_split_normalized_prompt": 2}
    dataset = ResumableTaskDataset(binding, output)
    rows = list(dataset.iter_examples())
    assert {row["language"] for row in rows} == {"english"}
    assert {row["family"] for row in rows} == {INSTRUCTION_RESPONSE_FAMILY}
    ids = binding.structural_token_ids
    for row in rows:
        input_ids = row["input_ids"]
        loss_start = row["loss_start_index"]
        assert input_ids[2] == ids["<|bcnlt:mode:text|>"]
        assert input_ids[loss_start + 1] == ids["<|bcnlt:mode:text|>"]
    connection = sqlite3.connect(output)
    try:
        assert connection.execute("SELECT COUNT(*) FROM rejections").fetchone()[0] == 2
        assert (
            connection.execute("SELECT COUNT(*) FROM instruction_provenance").fetchone()[0]
            == 4
        )
    finally:
        connection.close()


def test_rejects_missing_review_evidence_and_preserves_no_output(tmp_path: Path) -> None:
    binding = NativeTokenizerBinding.load()
    source = tmp_path / "source"
    train = _row(1, prompt="Train", output="Answer")
    train["semantic_review"] = {}
    _write_source(
        source,
        [train],
        [
            _eval_row(4, split="validation", prompt="Validation", output="Answer"),
            _eval_row(1004, split="test", prompt="Test", output="Answer"),
        ],
    )
    output = tmp_path / "curriculum.sqlite"

    with pytest.raises(ValueError, match="admission evidence"):
        build_reviewed_instruction_curriculum(
            binding,
            source_root=source,
            output_database=output,
            max_sequence=256,
        )
    assert not output.exists()


def test_preserves_original_requirements_rendering(tmp_path: Path) -> None:
    binding = NativeTokenizerBinding.load()
    source = tmp_path / "source"
    train = _row(1, prompt="Implement it", output="Done")
    train["constraints"] = ["Keep the API stable", "Return a value"]
    # IDs are content addresses and must change when fixture content changes.
    train["id"] = _content_digest({key: value for key, value in train.items() if key != "id"})
    _write_source(
        source,
        [train],
        [
            _eval_row(4, split="validation", prompt="Validation", output="Answer"),
            _eval_row(1004, split="test", prompt="Test", output="Answer"),
        ],
    )
    output = tmp_path / "curriculum.sqlite"
    build_reviewed_instruction_curriculum(
        binding,
        source_root=source,
        output_database=output,
        max_sequence=256,
    )
    dataset = ResumableTaskDataset(binding, output)
    row = next(item for item in dataset.iter_examples() if item["split"] == "train")
    prompt_ids = row["input_ids"][: row["loss_start_index"]]
    prompt = binding.decode(prompt_ids[3:-1])
    assert prompt == (
        "Implement it\n\nRequirements:\n"
        "- Keep the API stable\n"
        "- Return a value"
    )


def test_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    binding = NativeTokenizerBinding.load()
    source = tmp_path / "source"
    _write_source(
        source,
        [_row(1, prompt="Train", output="Answer")],
        [
            _eval_row(4, split="validation", prompt="Validation", output="Answer"),
            _eval_row(1004, split="test", prompt="Test", output="Answer"),
        ],
    )
    (source / "train.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source hash mismatch"):
        build_reviewed_instruction_curriculum(
            binding,
            source_root=source,
            output_database=tmp_path / "curriculum.sqlite",
            max_sequence=256,
        )
