from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from bosscrafts_model.tokenizer import (
    EXPECTED_BINDING_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    NativeTokenizerBinding,
)
from bosscrafts_model.tokenizer_fidelity_acceptance import (
    TokenizerFidelityBatchResult,
    _raw_payload_ranges,
    _segmented_roundtrip,
    build_tokenizer_fidelity_receipt,
    publish_tokenizer_fidelity_receipt,
    validate_tokenizer_fidelity_receipt,
)


def _result(example_id: str = "e" * 64) -> TokenizerFidelityBatchResult:
    record = {
        "example_id": example_id,
        "exact_roundtrip": True,
        "id_stable": True,
        "reserved_control_violation": False,
    }
    detail = {
        **record,
        "task_example_id": "a" * 64,
        "language": "python",
        "family": "code_to_semantic",
        "input_ids_sha256": "b" * 64,
    }
    return TokenizerFidelityBatchResult(
        case_bundle_manifest_sha256="c" * 64,
        case_inventory_sha256="d" * 64,
        task_database_identity_sha256="f" * 64,
        binding_sha256=EXPECTED_BINDING_SHA256,
        tokenizer_sha256=EXPECTED_TOKENIZER_SHA256,
        records=(record,),
        details=(detail,),
    )


def test_builds_validates_and_publishes_checkpoint_bound_receipt(
    tmp_path: Path,
) -> None:
    receipt = build_tokenizer_fidelity_receipt(
        _result(),
        checkpoint_sha256="3" * 64,
        checkpoint_step=17,
    )
    assert (
        validate_tokenizer_fidelity_receipt(
            receipt,
            expected_checkpoint_sha256="3" * 64,
            expected_checkpoint_step=17,
            expected_example_ids=("e" * 64,),
            expected_records=[receipt["records"][0]],
            expected_task_database_identity_sha256="f" * 64,
        )
        == receipt
    )
    output = tmp_path / "tokenizer-fidelity.json"
    assert publish_tokenizer_fidelity_receipt(output, receipt) == output.resolve()
    with pytest.raises(ValueError, match="already exists"):
        publish_tokenizer_fidelity_receipt(output, receipt)

    forged = dict(receipt)
    forged["checkpoint_step"] = 18
    with pytest.raises(ValueError, match="integrity"):
        validate_tokenizer_fidelity_receipt(forged)


def test_segmented_roundtrip_preserves_escaped_payload_literals() -> None:
    binding = NativeTokenizerBinding.load()
    structural = binding.structural_token_ids
    raw_payload = binding.encode_payload("<|eos|>\nreturn 1\n")
    semantic_payload = [
        structural["<|bcnlt:record:begin|>"],
        structural["<|bcnlt:record:end|>"],
    ]
    input_ids = [
        structural["<|bcnlt:task:code_to_semantic|>"],
        structural["<|bcnlt:stream:begin|>"],
        structural["<|bcnlt:mode:raw_code|>"],
        *raw_payload,
        structural["<|bcnlt:stream:end|>"],
        structural["<|bcnlt:stream:begin|>"],
        structural["<|bcnlt:mode:semantic_code|>"],
        *semantic_payload,
        structural["<|bcnlt:stream:end|>"],
        structural["<|bcnlt:task:end|>"],
    ]
    assert _segmented_roundtrip(binding, input_ids) == (True, True)
    ranges = _raw_payload_ranges(
        binding,
        family="code_to_semantic",
        input_ids=input_ids,
    )
    assert len(ranges) == 1
    start, end = ranges[0]
    assert input_ids[start:end] == raw_payload
    assert all(token_id > 76 for token_id in raw_payload)


def test_receipt_rejects_detail_substitution() -> None:
    receipt = build_tokenizer_fidelity_receipt(
        _result(),
        checkpoint_sha256="3" * 64,
        checkpoint_step=17,
    )
    forged = dict(receipt)
    forged_details = [dict(receipt["details"][0])]
    forged_details[0]["task_example_id"] = hashlib.sha256(b"other").hexdigest()
    forged["details"] = forged_details
    with pytest.raises(ValueError, match="integrity"):
        validate_tokenizer_fidelity_receipt(forged)


def test_tokenizer_fidelity_script_exposes_frozen_inputs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_tokenizer_fidelity_acceptance.py"),
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--case-bundle" in completed.stdout
    assert "--test-database" in completed.stdout
    assert "--checkpoint" in completed.stdout
    assert "--checkpoint-step" in completed.stdout
    assert "--output" in completed.stdout
