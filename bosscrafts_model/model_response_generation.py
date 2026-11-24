"""Ground free-form acceptance responses in one exact model generation run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from .category_case_inventory import (
    load_category_case_bundle,
    load_category_case_records,
)
from .immutable_artifacts import publish_new_files
from .inference_protocol import (
    build_inference_prompt,
    response_payload_token_ids,
    response_protocol_complete,
    task_end_token_id,
)
from .model import EngineeringModel
from .tokenizer import (
    EXPECTED_TOKENIZER_SHA256,
    NativeTokenizerBinding,
    bound_native_65k_config,
)

MODEL_RESPONSE_GENERATION_SCHEMA = "bosscrafts-model-response-generation-v1"
MODEL_RESPONSE_CATEGORIES = ("ordinary_instruction", "safety")
DETERMINISTIC_BASELINE_SEED = 7
MAX_MODEL_RESPONSE_CASES = 4_096
MAX_MODEL_RESPONSE_TOKENS = 2_048
MAX_MODEL_RESPONSE_BYTES = 1024 * 1024
MAX_MODEL_RESPONSE_RECEIPT_BYTES = 256 * 1024 * 1024
_GENERATION_TASK = "instruction_to_code"
_GENERATION_ALGORITHM = "greedy_argmax"
_DTYPE_NAMES = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}


@dataclass(frozen=True)
class ModelResponseGenerationResult:
    category: str
    case_bundle_manifest_sha256: str
    case_inventory_sha256: str
    binding_sha256: str
    tokenizer_sha256: str
    model_fingerprint: str
    device_type: str
    dtype: str
    maximum_new_tokens: int
    details: tuple[dict[str, object], ...]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _prompt_from_case(record: dict[str, object], category: str) -> str:
    case = record.get("case")
    prompt = case.get("prompt") if type(case) is dict else None
    if type(prompt) is not str or not prompt:
        raise ValueError(f"{category} generation case prompt is invalid")
    return prompt


def generate_model_response_cases(
    *,
    model: EngineeringModel,
    binding: NativeTokenizerBinding,
    category: str,
    case_bundle_root: str | Path,
    maximum_new_tokens: int,
) -> ModelResponseGenerationResult:
    """Greedily generate exact frozen prompts without accepting response files."""
    if type(model) is not EngineeringModel or type(binding) is not NativeTokenizerBinding:
        raise TypeError("model response generation requires the production model and binding")
    if category not in MODEL_RESPONSE_CATEGORIES:
        raise ValueError("model response generation category is invalid")
    if (
        type(maximum_new_tokens) is not int
        or not 1 <= maximum_new_tokens <= MAX_MODEL_RESPONSE_TOKENS
    ):
        raise ValueError("model response generation token limit is invalid")
    if (
        model.config.vocab_size != binding.vocab_size
        or model.config.tokenizer_manifest != binding.binding_sha256
    ):
        raise ValueError("model response generation binding mismatch")
    dtype_name = _DTYPE_NAMES.get(model.embedding.weight.dtype)
    device_type = model.embedding.weight.device.type
    if dtype_name is None or device_type not in {"cpu", "cuda"}:
        raise ValueError("model response generation runtime is unsupported")

    bundle = load_category_case_bundle(case_bundle_root)
    records = load_category_case_records(bundle, category)
    if not 1 <= len(records) <= MAX_MODEL_RESPONSE_CASES:
        raise ValueError("model response generation case count is invalid")
    device = model.embedding.weight.device
    end_token = task_end_token_id(binding, _GENERATION_TASK)
    details: list[dict[str, object]] = []
    for record in records:
        example_id = record.get("example_id")
        if not _is_sha256(example_id):
            raise ValueError("model response generation example ID is invalid")
        prompt_text = _prompt_from_case(record, category)
        prompt_ids = build_inference_prompt(
            binding,
            task=_GENERATION_TASK,
            prompt=prompt_text,
        )
        per_case_limit = min(
            maximum_new_tokens,
            model.config.max_sequence - len(prompt_ids),
        )
        if per_case_limit < 1:
            raise ValueError("model response generation has no response capacity")
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=per_case_limit,
            eos_token_id=end_token,
        )
        generated_ids = generated[0, len(prompt_ids) :].tolist()
        if not generated_ids or len(generated_ids) > MAX_MODEL_RESPONSE_TOKENS:
            raise ValueError("model response generation output size is invalid")
        complete = response_protocol_complete(
            binding,
            task=_GENERATION_TASK,
            generated_ids=generated_ids,
        )
        payload_ids = (
            response_payload_token_ids(binding, _GENERATION_TASK, generated_ids)
            if complete
            else []
        )
        response = binding.decode(payload_ids) if complete else ""
        if len(response.encode("utf-8")) > MAX_MODEL_RESPONSE_BYTES:
            raise ValueError("model response generation response is too large")
        details.append(
            {
                "example_id": example_id,
                "prompt_sha256": hashlib.sha256(
                    prompt_text.encode("utf-8")
                ).hexdigest(),
                "prompt_token_count": len(prompt_ids),
                "prompt_ids_sha256": _sha256(prompt_ids),
                "generated_ids": generated_ids,
                "generated_ids_sha256": _sha256(generated_ids),
                "protocol_complete": complete,
                "payload_ids_sha256": _sha256(payload_ids),
                "response": response,
                "response_sha256": hashlib.sha256(
                    response.encode("utf-8")
                ).hexdigest(),
            }
        )

    reloaded = load_category_case_bundle(bundle.bundle_root)
    if (
        reloaded.manifest_sha256 != bundle.manifest_sha256
        or reloaded.inventory.inventory_sha256 != bundle.inventory.inventory_sha256
    ):
        raise ValueError("model response generation cases changed during execution")
    return ModelResponseGenerationResult(
        category=category,
        case_bundle_manifest_sha256=bundle.manifest_sha256,
        case_inventory_sha256=bundle.inventory.inventory_sha256,
        binding_sha256=binding.binding_sha256,
        tokenizer_sha256=binding.tokenizer_sha256,
        model_fingerprint=model.config.fingerprint(),
        device_type=device_type,
        dtype=dtype_name,
        maximum_new_tokens=maximum_new_tokens,
        details=tuple(details),
    )


_DETAIL_FIELDS = {
    "example_id",
    "prompt_sha256",
    "prompt_token_count",
    "prompt_ids_sha256",
    "generated_ids",
    "generated_ids_sha256",
    "protocol_complete",
    "payload_ids_sha256",
    "response",
    "response_sha256",
}


def _validated_details(
    binding: NativeTokenizerBinding,
    details: object,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    if (
        type(details) is not list
        or not 1 <= len(details) <= MAX_MODEL_RESPONSE_CASES
    ):
        raise ValueError("model response generation detail inventory is invalid")
    normalized: list[dict[str, object]] = []
    responses: dict[str, str] = {}
    for detail in details:
        if type(detail) is not dict or set(detail) != _DETAIL_FIELDS:
            raise ValueError("model response generation detail fields are invalid")
        example_id = detail.get("example_id")
        generated_ids = detail.get("generated_ids")
        response = detail.get("response")
        if (
            not _is_sha256(example_id)
            or not _is_sha256(detail.get("prompt_sha256"))
            or type(detail.get("prompt_token_count")) is not int
            or not 1
            <= int(detail["prompt_token_count"])
            <= bound_native_65k_config(binding).max_sequence
            or not _is_sha256(detail.get("prompt_ids_sha256"))
            or type(generated_ids) is not list
            or not 1 <= len(generated_ids) <= MAX_MODEL_RESPONSE_TOKENS
            or any(
                type(token_id) is not int
                or not 0 <= token_id < binding.vocab_size
                for token_id in generated_ids
            )
            or not _is_sha256(detail.get("generated_ids_sha256"))
            or type(detail.get("protocol_complete")) is not bool
            or not _is_sha256(detail.get("payload_ids_sha256"))
            or type(response) is not str
            or len(response.encode("utf-8")) > MAX_MODEL_RESPONSE_BYTES
            or not _is_sha256(detail.get("response_sha256"))
        ):
            raise ValueError("model response generation detail is invalid")
        complete = response_protocol_complete(
            binding,
            task=_GENERATION_TASK,
            generated_ids=generated_ids,
        )
        payload_ids = (
            response_payload_token_ids(binding, _GENERATION_TASK, generated_ids)
            if complete
            else []
        )
        expected_response = binding.decode(payload_ids) if complete else ""
        if (
            detail["generated_ids_sha256"] != _sha256(generated_ids)
            or detail["protocol_complete"] is not complete
            or detail["payload_ids_sha256"] != _sha256(payload_ids)
            or response != expected_response
            or detail["response_sha256"]
            != hashlib.sha256(response.encode("utf-8")).hexdigest()
            or str(example_id) in responses
        ):
            raise ValueError("model response generation detail is inconsistent")
        normalized.append(detail)
        responses[str(example_id)] = response
    observed_ids = tuple(str(detail["example_id"]) for detail in normalized)
    if observed_ids != tuple(sorted(observed_ids)):
        raise ValueError("model response generation details are not ordered")
    return normalized, responses


_RECEIPT_FIELDS = {
    "schema_version",
    "category",
    "source_type",
    "checkpoint_sha256",
    "checkpoint_step",
    "random_seed",
    "generation_algorithm",
    "generation_task",
    "maximum_new_tokens",
    "model_fingerprint",
    "binding_sha256",
    "tokenizer_sha256",
    "case_bundle_manifest_sha256",
    "case_inventory_sha256",
    "device_type",
    "dtype",
    "example_ids",
    "example_ids_sha256",
    "response_sha256_by_example_id",
    "response_manifest_sha256",
    "generation_sha256_by_example_id",
    "generation_manifest_sha256",
    "details",
    "details_sha256",
    "receipt_sha256",
}


def build_model_response_generation_receipt(
    result: ModelResponseGenerationResult,
    *,
    source_type: str,
    checkpoint_sha256: str | None,
    checkpoint_step: int | None,
    random_seed: int | None,
) -> dict[str, object]:
    """Build a receipt that independently re-decodes every recorded response."""
    if type(result) is not ModelResponseGenerationResult:
        raise TypeError("result must be a ModelResponseGenerationResult")
    if source_type == "checkpoint":
        if (
            not _is_sha256(checkpoint_sha256)
            or type(checkpoint_step) is not int
            or checkpoint_step < 1
            or random_seed is not None
        ):
            raise ValueError("checkpoint response generation lineage is invalid")
    elif source_type == "deterministic_random_initialization":
        if (
            checkpoint_sha256 is not None
            or checkpoint_step is not None
            or random_seed != DETERMINISTIC_BASELINE_SEED
        ):
            raise ValueError("random response generation lineage is invalid")
    else:
        raise ValueError("model response generation source type is invalid")
    body: dict[str, object] = {
        "schema_version": MODEL_RESPONSE_GENERATION_SCHEMA,
        "category": result.category,
        "source_type": source_type,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "random_seed": random_seed,
        "generation_algorithm": _GENERATION_ALGORITHM,
        "generation_task": _GENERATION_TASK,
        "maximum_new_tokens": result.maximum_new_tokens,
        "model_fingerprint": result.model_fingerprint,
        "binding_sha256": result.binding_sha256,
        "tokenizer_sha256": result.tokenizer_sha256,
        "case_bundle_manifest_sha256": result.case_bundle_manifest_sha256,
        "case_inventory_sha256": result.case_inventory_sha256,
        "device_type": result.device_type,
        "dtype": result.dtype,
        "example_ids": [detail["example_id"] for detail in result.details],
        "example_ids_sha256": _sha256(
            [detail["example_id"] for detail in result.details]
        ),
        "response_sha256_by_example_id": {
            str(detail["example_id"]): detail["response_sha256"]
            for detail in result.details
        },
        "response_manifest_sha256": _sha256(
            {
                str(detail["example_id"]): detail["response_sha256"]
                for detail in result.details
            }
        ),
        "generation_sha256_by_example_id": {
            str(detail["example_id"]): detail["generated_ids_sha256"]
            for detail in result.details
        },
        "generation_manifest_sha256": _sha256(
            {
                str(detail["example_id"]): detail["generated_ids_sha256"]
                for detail in result.details
            }
        ),
        "details": list(result.details),
        "details_sha256": _sha256(list(result.details)),
    }
    payload = {**body, "receipt_sha256": _sha256(body)}
    return validate_model_response_generation_receipt(payload)


def validate_model_response_generation_receipt(
    payload: object,
    *,
    expected_category: str | None = None,
    expected_source_type: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
    expected_example_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Recompute response text and manifests from recorded token IDs."""
    if type(payload) is not dict or set(payload) != _RECEIPT_FIELDS:
        raise ValueError("model response generation receipt fields are invalid")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != _sha256(body):
        raise ValueError("model response generation receipt integrity mismatch")
    category = payload.get("category")
    source_type = payload.get("source_type")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    checkpoint_step = payload.get("checkpoint_step")
    random_seed = payload.get("random_seed")
    if (
        payload.get("schema_version") != MODEL_RESPONSE_GENERATION_SCHEMA
        or category not in MODEL_RESPONSE_CATEGORIES
        or payload.get("generation_algorithm") != _GENERATION_ALGORITHM
        or payload.get("generation_task") != _GENERATION_TASK
        or type(payload.get("maximum_new_tokens")) is not int
        or not 1 <= int(payload["maximum_new_tokens"]) <= MAX_MODEL_RESPONSE_TOKENS
        or not _is_sha256(payload.get("model_fingerprint"))
        or not _is_sha256(payload.get("binding_sha256"))
        or payload.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256
        or not _is_sha256(payload.get("case_bundle_manifest_sha256"))
        or not _is_sha256(payload.get("case_inventory_sha256"))
        or payload.get("device_type") not in {"cpu", "cuda"}
        or payload.get("dtype") not in {"float32", "bfloat16"}
    ):
        raise ValueError("model response generation receipt lineage is invalid")
    if source_type == "checkpoint":
        if (
            not _is_sha256(checkpoint_sha256)
            or type(checkpoint_step) is not int
            or checkpoint_step < 1
            or random_seed is not None
        ):
            raise ValueError("checkpoint response generation lineage is invalid")
    elif source_type == "deterministic_random_initialization":
        if (
            checkpoint_sha256 is not None
            or checkpoint_step is not None
            or random_seed != DETERMINISTIC_BASELINE_SEED
        ):
            raise ValueError("random response generation lineage is invalid")
    else:
        raise ValueError("model response generation source type is invalid")

    binding = NativeTokenizerBinding.load()
    if (
        payload.get("binding_sha256") != binding.binding_sha256
        or payload.get("model_fingerprint")
        != bound_native_65k_config(binding).fingerprint()
    ):
        raise ValueError("model response generation binding lineage mismatch")
    details, responses = _validated_details(binding, payload.get("details"))
    maximum_new_tokens = int(payload["maximum_new_tokens"])
    maximum_sequence = bound_native_65k_config(binding).max_sequence
    if any(
        len(cast(list[int], detail["generated_ids"])) > maximum_new_tokens
        or int(cast(int, detail["prompt_token_count"]))
        + len(cast(list[int], detail["generated_ids"]))
        > maximum_sequence
        for detail in details
    ):
        raise ValueError("model response generation sequence bound is invalid")
    example_ids = tuple(str(detail["example_id"]) for detail in details)
    response_hashes = {
        example_id: hashlib.sha256(responses[example_id].encode("utf-8")).hexdigest()
        for example_id in example_ids
    }
    generation_hashes = {
        str(detail["example_id"]): detail["generated_ids_sha256"]
        for detail in details
    }
    if (
        payload.get("example_ids") != list(example_ids)
        or payload.get("example_ids_sha256") != _sha256(list(example_ids))
        or payload.get("response_sha256_by_example_id") != response_hashes
        or payload.get("response_manifest_sha256") != _sha256(response_hashes)
        or payload.get("generation_sha256_by_example_id") != generation_hashes
        or payload.get("generation_manifest_sha256") != _sha256(generation_hashes)
        or payload.get("details_sha256") != _sha256(details)
    ):
        raise ValueError("model response generation receipt inventory mismatch")
    if expected_category is not None and category != expected_category:
        raise ValueError("model response generation category mismatch")
    if expected_source_type is not None and source_type != expected_source_type:
        raise ValueError("model response generation source mismatch")
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError("model response generation checkpoint mismatch")
    if expected_checkpoint_step is not None and checkpoint_step != expected_checkpoint_step:
        raise ValueError("model response generation checkpoint step mismatch")
    if expected_example_ids is not None and tuple(expected_example_ids) != example_ids:
        raise ValueError("model response generation expected inventory mismatch")
    return payload


def responses_from_model_response_generation_receipt(
    payload: object,
    *,
    expected_category: str | None = None,
    expected_source_type: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
    expected_example_ids: Sequence[str] | None = None,
) -> dict[str, str]:
    """Return only responses independently derived from the generation receipt."""
    validated = validate_model_response_generation_receipt(
        payload,
        expected_category=expected_category,
        expected_source_type=expected_source_type,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_step=expected_checkpoint_step,
        expected_example_ids=expected_example_ids,
    )
    details = validated["details"]
    if type(details) is not list:
        raise AssertionError("validated generation details are not a list")
    return {str(detail["example_id"]): str(detail["response"]) for detail in details}


def publish_model_response_generation_receipt(
    output: str | Path,
    payload: dict[str, object],
) -> Path:
    validated = validate_model_response_generation_receipt(payload)
    rendered = (
        json.dumps(validated, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if len(rendered) > MAX_MODEL_RESPONSE_RECEIPT_BYTES:
        raise ValueError("model response generation receipt exceeds its size bound")
    destination = Path(output).resolve()
    publish_new_files(
        ((destination, rendered),),
        artifact_label="model response generation receipt",
    )
    return destination
