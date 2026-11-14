from __future__ import annotations

import copy
import hashlib
import json

import pytest

from bosscrafts_model.model_response_generation import (
    DETERMINISTIC_BASELINE_SEED,
    ModelResponseGenerationResult,
    build_model_response_generation_receipt,
    responses_from_model_response_generation_receipt,
    validate_model_response_generation_receipt,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config

BINDING = NativeTokenizerBinding.load()
CHECKPOINT_ID = "e" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _detail(example_id: str, response: str = "candidate") -> dict[str, object]:
    control = BINDING.structural_token_ids
    payload_ids = BINDING.encode_payload(response)
    generated_ids = [
        control["<|bcnlt:stream:begin|>"],
        control["<|bcnlt:mode:raw_code|>"],
        *payload_ids,
        control["<|bcnlt:stream:end|>"],
        control["<|bcnlt:task:end|>"],
    ]
    return {
        "example_id": example_id,
        "prompt_sha256": hashlib.sha256(b"prompt").hexdigest(),
        "prompt_token_count": 4,
        "prompt_ids_sha256": hashlib.sha256(_canonical([1, 2, 3, 4])).hexdigest(),
        "generated_ids": generated_ids,
        "generated_ids_sha256": hashlib.sha256(
            _canonical(generated_ids)
        ).hexdigest(),
        "protocol_complete": True,
        "payload_ids_sha256": hashlib.sha256(_canonical(payload_ids)).hexdigest(),
        "response": response,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
    }


def _result(
    category: str = "ordinary_instruction",
    response: str = "candidate",
) -> ModelResponseGenerationResult:
    example_ids = sorted(
        hashlib.sha256(f"case:{index}".encode()).hexdigest() for index in range(2)
    )
    return ModelResponseGenerationResult(
        category=category,
        case_bundle_manifest_sha256="1" * 64,
        case_inventory_sha256="2" * 64,
        binding_sha256=BINDING.binding_sha256,
        tokenizer_sha256=BINDING.tokenizer_sha256,
        model_fingerprint=bound_native_65k_config(BINDING).fingerprint(),
        device_type="cpu",
        dtype="float32",
        maximum_new_tokens=64,
        details=tuple(_detail(example_id, response) for example_id in example_ids),
    )


def _checkpoint_receipt(
    category: str = "ordinary_instruction",
) -> dict[str, object]:
    return build_model_response_generation_receipt(
        _result(category),
        source_type="checkpoint",
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
        random_seed=None,
    )


def test_checkpoint_generation_receipt_rederives_responses() -> None:
    receipt = _checkpoint_receipt()
    validated = validate_model_response_generation_receipt(
        receipt,
        expected_category="ordinary_instruction",
        expected_source_type="checkpoint",
        expected_checkpoint_sha256=CHECKPOINT_ID,
        expected_checkpoint_step=86_016,
    )
    assert validated == receipt
    assert set(responses_from_model_response_generation_receipt(receipt).values()) == {
        "candidate"
    }


def test_random_baseline_requires_fixed_seed_and_no_checkpoint() -> None:
    receipt = build_model_response_generation_receipt(
        _result(response=""),
        source_type="deterministic_random_initialization",
        checkpoint_sha256=None,
        checkpoint_step=None,
        random_seed=DETERMINISTIC_BASELINE_SEED,
    )
    assert receipt["random_seed"] == DETERMINISTIC_BASELINE_SEED
    with pytest.raises(ValueError, match="random response generation lineage"):
        build_model_response_generation_receipt(
            _result(response=""),
            source_type="deterministic_random_initialization",
            checkpoint_sha256=None,
            checkpoint_step=None,
            random_seed=8,
        )


def test_rehashed_forgery_cannot_substitute_response_for_generated_ids() -> None:
    forged = copy.deepcopy(_checkpoint_receipt())
    details = forged["details"]
    assert type(details) is list
    details[0]["response"] = "forged"
    details[0]["response_sha256"] = hashlib.sha256(b"forged").hexdigest()
    response_hashes = {
        str(detail["example_id"]): detail["response_sha256"] for detail in details
    }
    forged["response_sha256_by_example_id"] = response_hashes
    forged["response_manifest_sha256"] = hashlib.sha256(
        _canonical(response_hashes)
    ).hexdigest()
    forged["details_sha256"] = hashlib.sha256(_canonical(details)).hexdigest()
    body = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(ValueError, match="detail is inconsistent"):
        validate_model_response_generation_receipt(forged)


def test_incomplete_protocol_can_only_record_an_empty_response() -> None:
    result = _result()
    detail = dict(result.details[0])
    detail["generated_ids"] = [100]
    detail["generated_ids_sha256"] = hashlib.sha256(_canonical([100])).hexdigest()
    detail["protocol_complete"] = False
    detail["payload_ids_sha256"] = hashlib.sha256(_canonical([])).hexdigest()
    detail["response"] = ""
    detail["response_sha256"] = hashlib.sha256(b"").hexdigest()
    incomplete = ModelResponseGenerationResult(
        **{
            **result.__dict__,
            "details": (detail, *result.details[1:]),
        }
    )
    receipt = build_model_response_generation_receipt(
        incomplete,
        source_type="checkpoint",
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
        random_seed=None,
    )
    responses = responses_from_model_response_generation_receipt(receipt)
    assert responses[str(detail["example_id"])] == ""
