from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bosscrafts_model.bcnlt_adapter import BCNLTCanonicalRecord
from bosscrafts_model.bcnlt_source_pipeline import _NativeSemanticStreamCodec
from bosscrafts_model.semantic_acceptance import (
    SemanticAcceptanceBatchResult,
    build_semantic_acceptance_receipt,
    publish_semantic_acceptance_receipt,
    validate_semantic_acceptance_receipt,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _semantic_record(*, scope: str = "module") -> BCNLTCanonicalRecord:
    payload = {
        "ast_relation": "identifier.self",
        "confidence_band": "high",
        "defines": "value",
        "event": "DEFINE",
        "ir": {
            "children": [],
            "kind": "variable",
            "metadata": {},
            "name": "value",
            "value": None,
        },
        "metadata": {
            "knowledge_state": "known",
            "uncertainty_signals": [],
            "uyr_rounds": 0,
        },
        "recent_history": [],
        "scope": scope,
        "span": [1, 0, 1, 5],
        "surface": "value",
        "symbol_id": f"{scope}.value",
        "uncertainty_band": "low",
    }
    canonical = _canonical(payload).decode()
    return BCNLTCanonicalRecord(
        language="python",
        canonical_json=canonical,
        payload_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        event="DEFINE",
        surface="value",
        confidence_band="high",
        uncertainty_band="low",
        span=(1, 0, 1, 5),
    )


def _result() -> SemanticAcceptanceBatchResult:
    binding = NativeTokenizerBinding.load()
    record = _semantic_record()
    ids = [
        *_NativeSemanticStreamCodec(binding).encode((record,)),
        binding.structural_token_ids["<|bcnlt:task:end|>"],
    ]
    jsonl = record.canonical_json + "\n"
    payload = json.loads(record.canonical_json)
    projection = [
        {
            field: payload.get(field)
            for field in (
                "event",
                "scope",
                "defines",
                "symbol_id",
                "span",
                "ast_relation",
            )
        }
    ]
    example_id = "e" * 64
    detail: dict[str, object] = {
        "example_id": example_id,
        "task_example_id": "a" * 64,
        "language": "python",
        "task_input_ids_sha256": "b" * 64,
        "expected_ids": ids,
        "expected_ids_sha256": _sha256(ids),
        "generated_ids": ids,
        "generated_ids_sha256": _sha256(ids),
        "protocol_complete": True,
        "expected_semantic_jsonl_sha256": hashlib.sha256(
            jsonl.encode()
        ).hexdigest(),
        "generated_semantic_jsonl_sha256": hashlib.sha256(
            jsonl.encode()
        ).hexdigest(),
        "expected_record_count": 1,
        "generated_record_count": 1,
        "expected_ownership_sha256": _sha256(projection),
        "generated_ownership_sha256": _sha256(projection),
    }
    return SemanticAcceptanceBatchResult(
        case_bundle_manifest_sha256="c" * 64,
        case_inventory_sha256="d" * 64,
        task_database_identity_sha256="f" * 64,
        binding_sha256=binding.binding_sha256,
        tokenizer_sha256=binding.tokenizer_sha256,
        records=(
            {
                "example_id": example_id,
                "language": "python",
                "exact_semantic_record": True,
                "ownership_correct": True,
                "unknown_semantic": False,
            },
        ),
        details=(detail,),
    )


def test_builds_validates_and_publishes_semantic_receipt(tmp_path: Path) -> None:
    receipt = build_semantic_acceptance_receipt(
        _result(), checkpoint_sha256="3" * 64, checkpoint_step=17
    )
    assert (
        validate_semantic_acceptance_receipt(
            receipt,
            expected_checkpoint_sha256="3" * 64,
            expected_checkpoint_step=17,
            expected_example_ids=("e" * 64,),
            expected_records=[receipt["records"][0]],
            expected_task_database_identity_sha256="f" * 64,
        )
        == receipt
    )
    output = tmp_path / "semantic.json"
    assert publish_semantic_acceptance_receipt(output, receipt) == output.resolve()
    with pytest.raises(ValueError, match="already exists"):
        publish_semantic_acceptance_receipt(output, receipt)


def test_semantic_receipt_recomputes_records_instead_of_trusting_booleans() -> None:
    receipt = build_semantic_acceptance_receipt(
        _result(), checkpoint_sha256="3" * 64, checkpoint_step=17
    )
    forged = json.loads(json.dumps(receipt))
    forged["records"][0]["ownership_correct"] = False
    forged["records_sha256"] = _sha256(forged["records"])
    body = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = _sha256(body)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_semantic_acceptance_receipt(forged)


def test_semantic_receipt_rejects_rehashed_generated_stream_substitution() -> None:
    receipt = build_semantic_acceptance_receipt(
        _result(), checkpoint_sha256="3" * 64, checkpoint_step=17
    )
    forged = json.loads(json.dumps(receipt))
    forged["details"][0]["generated_ids"] = []
    forged["details"][0]["generated_ids_sha256"] = _sha256([])
    forged["details_sha256"] = _sha256(forged["details"])
    body = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = _sha256(body)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_semantic_acceptance_receipt(forged)


def test_semantic_acceptance_runner_exposes_exact_lineage_inputs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).parents[1]
                / "scripts"
                / "run_semantic_acceptance.py"
            ),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "--case-bundle",
        "--checkpoint-database",
        "--validation-database",
        "--test-database",
        "--heldout-manifest",
        "--checkpoint",
        "--max-new-tokens",
        "--output",
    ):
        assert option in completed.stdout
