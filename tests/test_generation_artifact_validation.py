from __future__ import annotations

import json
from pathlib import Path

import pytest

from bosscrafts_model.generation_artifacts import (
    load_generation_gate_records,
    load_humaneval_generation_records,
)

RUN_IDENTITY = "a" * 64


def _write_jsonl(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _humaneval_record() -> dict[str, object]:
    return {
        "schema_version": "bosscrafts-openai-humaneval-generation-v2",
        "run_identity_sha256": RUN_IDENTITY,
        "task_index": 0,
        "task_id": "HumanEval/0",
        "entry_point": "candidate",
        "prompt_sha256": "b" * 64,
        "prompt_token_count": 12,
        "generated_token_count": 3,
        "completion_token_count": 2,
        "generation_seconds": 0.5,
        "tokens_per_second": 6.0,
        "response_protocol_complete": True,
        "raw_model_response": " return 1",
        "raw_completion": " return 1",
        "clean_completion": " return 1\n",
    }


def _generation_gate_record() -> dict[str, object]:
    return {
        "schema_version": "bosscrafts-strict-generation-gate-v1",
        "run_identity_sha256": RUN_IDENTITY,
        "example_id": "example-1",
        "family": "instruction_to_code",
        "prompt_token_count": 2,
        "expected_token_count": 2,
        "generated_token_count": 2,
        "expected_ids": [10, 11],
        "generated_ids": [10, 11],
        "expected_protocol_text": "expected",
        "generated_protocol_text": "generated",
        "generated_payload_text": "payload",
        "protocol_complete": True,
        "exact_match": True,
        "target_token_accuracy": 1.0,
        "generation_seconds": 0.25,
        "tokens_per_second": 8.0,
    }


def test_humaneval_resume_record_requires_typed_metrics(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    valid = _humaneval_record()
    _write_jsonl(path, valid)
    assert load_humaneval_generation_records(path, RUN_IDENTITY)["HumanEval/0"][
        "generation_seconds"
    ] == 0.5

    valid["generation_seconds"] = "0.5"
    _write_jsonl(path, valid)
    with pytest.raises(ValueError, match="generation_seconds"):
        load_humaneval_generation_records(path, RUN_IDENTITY)


def test_generation_gate_resume_record_requires_integer_token_lists(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    valid = _generation_gate_record()
    _write_jsonl(path, valid)
    assert load_generation_gate_records(path, RUN_IDENTITY)["example-1"][
        "expected_ids"
    ] == [10, 11]

    valid["expected_ids"] = "10,11"
    _write_jsonl(path, valid)
    with pytest.raises(ValueError, match="expected_ids"):
        load_generation_gate_records(path, RUN_IDENTITY)


def test_generation_artifacts_reject_inconsistent_derived_metrics(tmp_path: Path) -> None:
    humaneval_path = tmp_path / "humaneval.jsonl"
    humaneval = _humaneval_record()
    humaneval["tokens_per_second"] = 5.0
    _write_jsonl(humaneval_path, humaneval)
    with pytest.raises(ValueError, match="tokens_per_second is inconsistent"):
        load_humaneval_generation_records(humaneval_path, RUN_IDENTITY)

    gate_path = tmp_path / "gate.jsonl"
    gate = _generation_gate_record()
    gate["exact_match"] = False
    _write_jsonl(gate_path, gate)
    with pytest.raises(ValueError, match="exact_match is inconsistent"):
        load_generation_gate_records(gate_path, RUN_IDENTITY)


def test_generation_artifacts_require_bounded_newline_framing(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(_humaneval_record()), encoding="utf-8")

    with pytest.raises(ValueError, match="bounded framing"):
        load_humaneval_generation_records(path, RUN_IDENTITY)


def test_generation_artifacts_reject_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    rendered = json.dumps(_generation_gate_record())
    rendered = rendered[:-1] + ', "example_id": "duplicate"}\n'
    path.write_text(rendered, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_generation_gate_records(path, RUN_IDENTITY)
