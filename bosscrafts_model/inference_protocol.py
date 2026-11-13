"""Inference prompts that exactly mirror the frozen BCNLT training protocol."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .bcnlt_source_pipeline import _NativeSemanticStreamCodec
from .tokenizer import NativeTokenizerBinding

INFERENCE_TASKS = (
    "raw",
    "code_to_semantic",
    "semantic_to_code",
    "semantic_continuation",
    "instruction_to_code",
)


@dataclass(frozen=True)
class HeldoutInferenceCase:
    example_id: str
    family: str
    prompt_ids: tuple[int, ...]
    expected_ids: tuple[int, ...]


def select_heldout_inference_case(
    examples: Iterable[Mapping[str, object]],
    *,
    family: str,
    max_new_tokens: int,
    maximum_sequence: int,
) -> HeldoutInferenceCase:
    """Select the first bounded held-out example using its exact packed prefix."""
    if family not in INFERENCE_TASKS or family == "raw":
        raise ValueError("held-out inference family must be a structured task")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256")
    if type(maximum_sequence) is not int or maximum_sequence < 2:
        raise ValueError("maximum_sequence must be at least two")
    for example in examples:
        if example.get("family") != family:
            continue
        example_id = example.get("example_id")
        input_ids = example.get("input_ids")
        loss_start = example.get("loss_start_index")
        if (
            type(example_id) is not str
            or len(example_id) != 64
            or any(character not in "0123456789abcdef" for character in example_id)
            or type(input_ids) is not list
            or len(input_ids) < 2
            or any(type(token_id) is not int for token_id in input_ids)
            or type(loss_start) is not int
            or not 0 < loss_start < len(input_ids)
        ):
            raise ValueError("held-out task example is invalid")
        prompt_ids = input_ids[:loss_start]
        expected_ids = input_ids[loss_start:]
        if len(expected_ids) > max_new_tokens:
            continue
        if len(prompt_ids) + max_new_tokens > maximum_sequence:
            continue
        return HeldoutInferenceCase(
            example_id=example_id,
            family=family,
            prompt_ids=tuple(prompt_ids),
            expected_ids=tuple(expected_ids),
        )
    raise ValueError("no eligible held-out example fits the inference bounds")


def _id(binding: NativeTokenizerBinding, name: str) -> int:
    return binding.structural_token_ids[f"<|bcnlt:{name}|>"]


def build_inference_prompt(
    binding: NativeTokenizerBinding,
    *,
    task: str,
    prompt: str | Sequence[int],
) -> list[int]:
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if task not in INFERENCE_TASKS:
        raise ValueError("unsupported inference task")
    if task == "raw":
        if type(prompt) is not str or not prompt:
            raise ValueError("raw inference prompt must be a non-empty string")
        return binding.encode_payload(prompt)
    if task == "code_to_semantic":
        if type(prompt) is not str or not prompt:
            raise ValueError("code prompt must be a non-empty string")
        return [
            _id(binding, "task:code_to_semantic"),
            _id(binding, "stream:begin"),
            _id(binding, "mode:raw_code"),
            *binding.encode_payload(prompt),
            _id(binding, "stream:end"),
        ]
    if task == "instruction_to_code":
        if type(prompt) is not str or not prompt:
            raise ValueError("instruction prompt must be a non-empty string")
        return [
            _id(binding, "task:mixed_instruction"),
            _id(binding, "stream:begin"),
            _id(binding, "mode:raw_code"),
            *binding.encode_payload(prompt),
            _id(binding, "stream:end"),
        ]
    if type(prompt) is str:
        if not prompt:
            raise ValueError("semantic prompt must not be empty")
        # Text is retained for backward compatibility. Token IDs are the
        # lossless form when a semantic payload literally spells a control
        # token, because decoded stream text is intentionally ambiguous.
        semantic_ids = binding.encode(prompt)
    elif isinstance(prompt, (list, tuple)) and prompt and all(
        type(token_id) is int and 0 <= token_id < binding.vocab_size
        for token_id in prompt
    ):
        semantic_ids = list(prompt)
    else:
        raise TypeError("semantic prompt must be a stream string or bounded token IDs")
    if (
        semantic_ids[:2]
        != [_id(binding, "stream:begin"), _id(binding, "mode:semantic_code")]
        or semantic_ids[-1:] != [_id(binding, "stream:end")]
    ):
        raise ValueError("semantic tasks require a complete semantic stream")
    try:
        _NativeSemanticStreamCodec(binding).decode(semantic_ids)
    except (TypeError, ValueError) as error:
        raise ValueError("semantic stream is invalid") from error
    task_token = (
        "task:semantic_to_code"
        if task == "semantic_to_code"
        else "task:semantic_continuation"
    )
    return [_id(binding, task_token), *semantic_ids]


def task_end_token_id(binding: NativeTokenizerBinding, task: str) -> int | None:
    if task == "raw":
        return None
    if task not in INFERENCE_TASKS:
        raise ValueError("unsupported inference task")
    return _id(binding, "task:end")


def response_token_ids(
    binding: NativeTokenizerBinding, task: str, generated_ids: list[int]
) -> list[int]:
    if type(generated_ids) is not list or any(
        type(token_id) is not int for token_id in generated_ids
    ):
        raise TypeError("generated_ids must be an integer list")
    end = task_end_token_id(binding, task)
    if end is not None and generated_ids[-1:] == [end]:
        return generated_ids[:-1]
    return list(generated_ids)


def response_protocol_complete(
    binding: NativeTokenizerBinding, *, task: str, generated_ids: list[int]
) -> bool:
    if task == "raw":
        return True
    if task not in INFERENCE_TASKS:
        raise ValueError("unsupported inference task")
    if type(generated_ids) is not list or any(
        type(token_id) is not int for token_id in generated_ids
    ):
        raise TypeError("generated_ids must be an integer list")
    if generated_ids[-1:] != [_id(binding, "task:end")]:
        return False
    response = generated_ids[:-1]
    expected_mode = (
        "mode:raw_code"
        if task in {"semantic_to_code", "instruction_to_code"}
        else "mode:semantic_code"
    )
    framing_valid = (
        len(response) >= 3
        and response[:2]
        == [_id(binding, "stream:begin"), _id(binding, expected_mode)]
        and response[-1] == _id(binding, "stream:end")
    )
    if not framing_valid:
        return False
    if task in {"semantic_to_code", "instruction_to_code"}:
        return not any(0 <= token_id <= 76 for token_id in response[2:-1])
    try:
        _NativeSemanticStreamCodec(binding).decode(response)
    except (TypeError, ValueError):
        return False
    return True


def response_payload_token_ids(
    binding: NativeTokenizerBinding, task: str, generated_ids: list[int]
) -> list[int]:
    """Return only the payload inside a complete response stream.

    An incomplete structured response is returned unchanged (apart from a final
    task-end token) so callers cannot mistake malformed output for valid code.
    """
    response = response_token_ids(binding, task, generated_ids)
    if task == "raw":
        return response
    if not response_protocol_complete(binding, task=task, generated_ids=generated_ids):
        return response
    return response[2:-1]
