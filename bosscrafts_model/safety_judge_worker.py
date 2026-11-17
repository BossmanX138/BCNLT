"""Minimal isolated worker for the pinned local NVIDIA safety judge."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .safety_judge_runtime import (
    SAFETY_JUDGE_RUNTIME_VERSIONS,
    SafetyJudgeRuntime,
    verify_safety_judge_runtime,
)

SAFETY_JUDGE_WORKER_INPUT_SCHEMA = "bosscrafts-safety-judge-worker-input-v1"
SAFETY_JUDGE_WORKER_OUTPUT_SCHEMA = "bosscrafts-safety-judge-worker-output-v1"
MAX_SAFETY_BATCH_BYTES = 64 * 1024 * 1024
MAX_SAFETY_RESPONSE_BYTES = 256 * 1024
MAX_SAFETY_JUDGMENT_BYTES = 4 * 1024
MAX_SAFETY_CASES = 4_096
MAX_SAFETY_CONTEXT_TOKENS = 131_000


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safety judge key: {key}")
        result[key] = value
    return result


def _load_worker_dependencies() -> tuple[Any, Any]:
    try:
        accelerate = importlib.import_module("accelerate")
        pillow = importlib.import_module("PIL")
        sentencepiece = importlib.import_module("sentencepiece")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise ValueError("safety judge dependencies are unavailable") from error
    torch_version = str(getattr(torch, "__version__", "")).split("+", maxsplit=1)[0]
    transformers_version = str(getattr(transformers, "__version__", ""))
    if (
        str(getattr(accelerate, "__version__", ""))
        != SAFETY_JUDGE_RUNTIME_VERSIONS["accelerate"]
        or str(getattr(pillow, "__version__", ""))
        != SAFETY_JUDGE_RUNTIME_VERSIONS["pillow"]
        or str(getattr(sentencepiece, "__version__", ""))
        != SAFETY_JUDGE_RUNTIME_VERSIONS["sentencepiece"]
        or torch_version != SAFETY_JUDGE_RUNTIME_VERSIONS["torch"]
        or transformers_version != SAFETY_JUDGE_RUNTIME_VERSIONS["transformers"]
    ):
        raise ValueError("safety judge dependency versions are invalid")
    if not bool(torch.cuda.is_available()):
        raise ValueError("safety judge CUDA runtime is unavailable")
    return torch, transformers


def _evaluate_payload(
    runtime: SafetyJudgeRuntime, payload: dict[str, object]
) -> dict[str, object]:
    raw_cases = payload.get("cases")
    if (
        set(payload) != {"schema_version", "runtime_manifest_sha256", "cases"}
        or payload.get("schema_version") != SAFETY_JUDGE_WORKER_INPUT_SCHEMA
        or payload.get("runtime_manifest_sha256") != runtime.manifest_sha256
        or type(raw_cases) is not list
        or not 1 <= len(raw_cases) <= MAX_SAFETY_CASES
    ):
        raise ValueError("safety judge worker input contract is invalid")
    torch, transformers = _load_worker_dependencies()
    processor = transformers.AutoProcessor.from_pretrained(
        str(runtime.root), local_files_only=True, use_fast=False
    )
    model = transformers.Gemma3ForConditionalGeneration.from_pretrained(
        str(runtime.root),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "6GiB", "cpu": "32GiB"},
    )
    if str(model.device).split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("safety judge input device is not CUDA")
    model.eval()
    results: list[dict[str, object]] = []
    for case in raw_cases:
        if (
            type(case) is not dict
            or set(case) != {"example_id", "prompt", "candidate_response"}
            or not _is_sha256(case.get("example_id"))
            or type(case.get("prompt")) is not str
            or not str(case["prompt"]).strip()
            or type(case.get("candidate_response")) is not str
            or len(str(case["prompt"]).encode("utf-8")) > MAX_SAFETY_RESPONSE_BYTES
            or len(str(case["candidate_response"]).encode("utf-8"))
            > MAX_SAFETY_RESPONSE_BYTES
        ):
            raise ValueError("safety judge worker case is invalid")
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": case["prompt"]}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": case["candidate_response"]}
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            request_categories="/categories",
        )
        input_length = int(inputs["input_ids"].shape[-1])
        if input_length > MAX_SAFETY_CONTEXT_TOKENS:
            raise ValueError("safety judge input exceeds the context bound")
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )
        output = processor.decode(
            generated[0][input_length:], skip_special_tokens=True
        )
        if (
            type(output) is not str
            or len(output.encode("utf-8")) > MAX_SAFETY_JUDGMENT_BYTES
        ):
            raise ValueError("safety judge generated output is invalid")
        results.append({"example_id": case["example_id"], "raw_output": output})
    return {
        "schema_version": SAFETY_JUDGE_WORKER_OUTPUT_SCHEMA,
        "runtime_manifest_sha256": runtime.manifest_sha256,
        "results": results,
    }


def worker_main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        content = sys.stdin.buffer.read(MAX_SAFETY_BATCH_BYTES + 1)
        if not content or len(content) > MAX_SAFETY_BATCH_BYTES:
            raise ValueError("safety judge worker input size is invalid")
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
        if type(payload) is not dict:
            raise ValueError("safety judge worker input must be an object")
        runtime = verify_safety_judge_runtime(arguments.runtime_root)
        output = _evaluate_payload(runtime, payload)
        if verify_safety_judge_runtime(runtime.root).manifest_sha256 != runtime.manifest_sha256:
            raise ValueError("safety judge runtime changed during worker execution")
        rendered = _canonical(output)
        if len(rendered) > MAX_SAFETY_BATCH_BYTES:
            raise ValueError("safety judge worker output size is invalid")
        sys.stdout.buffer.write(rendered)
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
