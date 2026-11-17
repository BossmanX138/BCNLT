"""Measure checkpoint inference resources under one frozen benchmark workload."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from .category_case_inventory import (
    CATEGORY_CASE_SCHEMA,
    _valid_provenance,
    _validate_case_payload,
    load_category_case_bundle,
    load_category_case_records,
)
from .immutable_artifacts import publish_new_files
from .inference_protocol import build_inference_prompt, task_end_token_id
from .model import EngineeringModel
from .tokenizer import (
    EXPECTED_TOKENIZER_SHA256,
    NativeTokenizerBinding,
    bound_native_65k_config,
)

RESOURCE_ACCEPTANCE_RECEIPT_SCHEMA = "bosscrafts-resource-acceptance-receipt-v1"
RESOURCE_BENCHMARK_PROTOCOL = "bosscrafts-instruction-greedy-kv-v1"
RESOURCE_BENCHMARK_PROMPT = (
    "Implement a Python function named binary_search(values, target) that returns "
    "the index of target in a sorted list or -1 when absent. Include type hints "
    "and handle an empty list."
)
RESOURCE_BENCHMARK_WARMUP_ITERATIONS = 1
RESOURCE_BENCHMARK_MEASURED_ITERATIONS = 3
RESOURCE_BENCHMARK_MAX_NEW_TOKENS = 128
MAX_RESOURCE_RECEIPT_BYTES = 16 * 1024 * 1024
_TASK = "instruction_to_code"


@dataclass(frozen=True)
class ResourceAcceptanceResult:
    case_bundle_manifest_sha256: str
    case_inventory_sha256: str
    test_task_database_identity_sha256: str
    binding_sha256: str
    tokenizer_sha256: str
    model_fingerprint: str
    device_type: str
    dtype: str
    case_record: dict[str, object]
    prompt_ids: tuple[int, ...]
    peak_vram_bytes: int
    peak_ram_bytes: int
    iterations: tuple[dict[str, object], ...]


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


def _validated_resource_case(value: object) -> tuple[str, dict[str, object]]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "category",
        "case_kind",
        "case",
        "provenance",
        "example_id",
    }:
        raise ValueError("resource acceptance case fields are invalid")
    case = value.get("case")
    provenance = value.get("provenance")
    body = {key: item for key, item in value.items() if key != "example_id"}
    example_id = value.get("example_id")
    if (
        value.get("schema_version") != CATEGORY_CASE_SCHEMA
        or value.get("category") != "resources"
        or not _valid_provenance(provenance)
        or type(case) is not dict
        or _validate_case_payload("resources", value.get("case_kind"), case)
        is not None
        or not _is_sha256(example_id)
        or example_id != _sha256(body)
    ):
        raise ValueError("resource acceptance case integrity is invalid")
    return str(example_id), case


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def peak_process_rss_bytes() -> int:
    """Return the OS-reported peak resident set for this benchmark process."""
    if sys.platform == "win32":
        ctypes_api = cast(Any, ctypes)
        kernel32 = ctypes_api.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes_api.WinDLL("psapi", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not get_process_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        peak = int(counters.PeakWorkingSetSize)
    elif sys.platform.startswith("linux"):
        try:
            fields = {
                line.split(":", maxsplit=1)[0]: line.split(":", maxsplit=1)[1]
                for line in Path("/proc/self/status").read_text(
                    encoding="utf-8"
                ).splitlines()
                if ":" in line
            }
            value, unit = fields["VmHWM"].split()
            if unit != "kB":
                raise ValueError("unexpected VmHWM unit")
            peak = int(value) * 1024
        except (OSError, KeyError, ValueError) as error:
            raise ValueError("Linux peak process memory is unavailable") from error
    else:
        resource = importlib.import_module("resource")
        usage = resource.getrusage(resource.RUSAGE_SELF)
        raw_peak = int(usage.ru_maxrss)
        peak = raw_peak if sys.platform == "darwin" else raw_peak * 1024
    if peak < 1:
        raise ValueError("peak process memory is invalid")
    return peak


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(
        error
    ).lower()


def _run_iteration(
    model: EngineeringModel,
    prompt: torch.Tensor,
    prompt_length: int,
    end_token: int,
    *,
    phase: str,
    index: int,
) -> dict[str, object]:
    if prompt.device.type == "cuda":
        torch.cuda.synchronize(prompt.device)
    started = time.perf_counter_ns()
    status = "completed"
    generated_ids: list[int]
    try:
        generated = model.generate(
            prompt,
            max_new_tokens=RESOURCE_BENCHMARK_MAX_NEW_TOKENS,
            eos_token_id=end_token,
        )
        if prompt.device.type == "cuda":
            torch.cuda.synchronize(prompt.device)
        generated_ids = generated[0, prompt_length:].tolist()
    except RuntimeError as error:
        if not _is_cuda_oom(error):
            raise
        if prompt.device.type == "cuda":
            torch.cuda.synchronize(prompt.device)
        status = "oom"
        generated_ids = []
    elapsed_ns = max(1, time.perf_counter_ns() - started)
    return {
        "phase": phase,
        "iteration": index,
        "status": status,
        "elapsed_ns": elapsed_ns,
        "generated_token_count": len(generated_ids),
        "generated_ids": generated_ids,
        "generated_ids_sha256": _sha256(generated_ids),
    }


def run_resource_acceptance_benchmark(
    *,
    model: EngineeringModel,
    binding: NativeTokenizerBinding,
    case_bundle_root: str | Path,
) -> ResourceAcceptanceResult:
    """Run the fixed workload after the caller resets CUDA peak statistics."""
    if type(model) is not EngineeringModel or type(binding) is not NativeTokenizerBinding:
        raise TypeError("resource acceptance requires the production model and binding")
    if (
        model.config.vocab_size != binding.vocab_size
        or model.config.tokenizer_manifest != binding.binding_sha256
    ):
        raise ValueError("resource acceptance model binding mismatch")
    device = model.embedding.weight.device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("resource acceptance requires CUDA")
    dtype_names = {torch.float32: "float32", torch.bfloat16: "bfloat16"}
    dtype_name = dtype_names.get(model.embedding.weight.dtype)
    if dtype_name is None:
        raise ValueError("resource acceptance dtype is unsupported")

    bundle = load_category_case_bundle(case_bundle_root)
    case_records = load_category_case_records(bundle, "resources")
    if len(case_records) != 1:
        raise ValueError("resource acceptance requires exactly one budget case")
    _validated_resource_case(case_records[0])
    prompt_ids = build_inference_prompt(
        binding,
        task=_TASK,
        prompt=RESOURCE_BENCHMARK_PROMPT,
    )
    if len(prompt_ids) + RESOURCE_BENCHMARK_MAX_NEW_TOKENS > model.config.max_sequence:
        raise ValueError("resource acceptance benchmark exceeds model sequence limit")
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    end_token = task_end_token_id(binding, _TASK)
    if end_token is None:
        raise AssertionError("structured resource benchmark has no end token")
    iterations = [
        _run_iteration(
            model,
            prompt,
            len(prompt_ids),
            end_token,
            phase="warmup",
            index=0,
        )
    ]
    for index in range(RESOURCE_BENCHMARK_MEASURED_ITERATIONS):
        iterations.append(
            _run_iteration(
                model,
                prompt,
                len(prompt_ids),
                end_token,
                phase="measured",
                index=index,
            )
        )
    peak_vram = int(torch.cuda.max_memory_reserved(device))
    peak_ram = peak_process_rss_bytes()
    reloaded = load_category_case_bundle(bundle.bundle_root)
    if (
        reloaded.manifest_sha256 != bundle.manifest_sha256
        or reloaded.inventory.inventory_sha256 != bundle.inventory.inventory_sha256
    ):
        raise ValueError("resource acceptance case bundle changed during execution")
    return ResourceAcceptanceResult(
        case_bundle_manifest_sha256=bundle.manifest_sha256,
        case_inventory_sha256=bundle.inventory.inventory_sha256,
        test_task_database_identity_sha256=(
            bundle.inventory.test_task_database_identity_sha256
        ),
        binding_sha256=binding.binding_sha256,
        tokenizer_sha256=binding.tokenizer_sha256,
        model_fingerprint=model.config.fingerprint(),
        device_type=device.type,
        dtype=dtype_name,
        case_record=case_records[0],
        prompt_ids=tuple(prompt_ids),
        peak_vram_bytes=peak_vram,
        peak_ram_bytes=peak_ram,
        iterations=tuple(iterations),
    )


_ITERATION_FIELDS = {
    "phase",
    "iteration",
    "status",
    "elapsed_ns",
    "generated_token_count",
    "generated_ids",
    "generated_ids_sha256",
}


def _validated_iterations(
    value: object,
    binding: NativeTokenizerBinding,
) -> list[dict[str, object]]:
    expected_count = (
        RESOURCE_BENCHMARK_WARMUP_ITERATIONS
        + RESOURCE_BENCHMARK_MEASURED_ITERATIONS
    )
    if type(value) is not list or len(value) != expected_count:
        raise ValueError("resource acceptance iteration inventory is invalid")
    normalized: list[dict[str, object]] = []
    for position, detail in enumerate(value):
        phase = "warmup" if position == 0 else "measured"
        index = 0 if position == 0 else position - 1
        if (
            type(detail) is not dict
            or set(detail) != _ITERATION_FIELDS
            or detail.get("phase") != phase
            or detail.get("iteration") != index
            or detail.get("status") not in {"completed", "oom"}
            or type(detail.get("elapsed_ns")) is not int
            or int(detail["elapsed_ns"]) < 1
            or type(detail.get("generated_token_count")) is not int
            or type(detail.get("generated_ids")) is not list
            or any(
                type(token_id) is not int
                or not 0 <= token_id < binding.vocab_size
                for token_id in detail["generated_ids"]
            )
            or not _is_sha256(detail.get("generated_ids_sha256"))
        ):
            raise ValueError("resource acceptance iteration is invalid")
        generated_ids = cast(list[int], detail["generated_ids"])
        if (
            len(generated_ids) > RESOURCE_BENCHMARK_MAX_NEW_TOKENS
            or detail["generated_token_count"] != len(generated_ids)
            or detail["generated_ids_sha256"] != _sha256(generated_ids)
            or (detail["status"] == "completed" and not generated_ids)
            or (detail["status"] == "oom" and bool(generated_ids))
        ):
            raise ValueError("resource acceptance iteration is inconsistent")
        normalized.append(detail)
    return normalized


def _derived_measurements(
    iterations: list[dict[str, object]],
) -> tuple[int, float]:
    oom_count = sum(detail["status"] == "oom" for detail in iterations)
    measured = [detail for detail in iterations if detail["phase"] == "measured"]
    total_tokens = sum(
        cast(int, detail["generated_token_count"]) for detail in measured
    )
    total_ns = sum(cast(int, detail["elapsed_ns"]) for detail in measured)
    throughput = total_tokens * 1_000_000_000 / total_ns
    if not math.isfinite(throughput) or throughput < 0.0:
        raise ValueError("resource acceptance throughput is invalid")
    return oom_count, throughput


_RECEIPT_FIELDS = {
    "schema_version",
    "checkpoint_sha256",
    "checkpoint_step",
    "case_bundle_manifest_sha256",
    "case_inventory_sha256",
    "test_task_database_identity_sha256",
    "binding_sha256",
    "tokenizer_sha256",
    "model_fingerprint",
    "device_type",
    "dtype",
    "benchmark_protocol",
    "benchmark_prompt_sha256",
    "prompt_ids",
    "prompt_ids_sha256",
    "warmup_iterations",
    "measured_iterations",
    "maximum_new_tokens",
    "case_record",
    "peak_vram_bytes",
    "peak_ram_bytes",
    "iterations",
    "iterations_sha256",
    "record",
    "receipt_sha256",
}


def build_resource_acceptance_receipt(
    result: ResourceAcceptanceResult,
    *,
    checkpoint_sha256: str,
    checkpoint_step: int,
) -> dict[str, object]:
    if (
        type(result) is not ResourceAcceptanceResult
        or not _is_sha256(checkpoint_sha256)
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
    ):
        raise ValueError("resource acceptance receipt lineage is invalid")
    example_id, budget = _validated_resource_case(result.case_record)
    iterations = list(result.iterations)
    oom_count, throughput = _derived_measurements(
        _validated_iterations(iterations, NativeTokenizerBinding.load())
    )
    record = {
        "example_id": example_id,
        "peak_vram_bytes": result.peak_vram_bytes,
        "vram_limit_bytes": budget["vram_limit_bytes"],
        "peak_ram_bytes": result.peak_ram_bytes,
        "ram_limit_bytes": budget["ram_limit_bytes"],
        "oom_count": oom_count,
        "inference_tokens_per_second": throughput,
        "minimum_tokens_per_second": budget["minimum_tokens_per_second"],
    }
    body: dict[str, object] = {
        "schema_version": RESOURCE_ACCEPTANCE_RECEIPT_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "case_bundle_manifest_sha256": result.case_bundle_manifest_sha256,
        "case_inventory_sha256": result.case_inventory_sha256,
        "test_task_database_identity_sha256": (
            result.test_task_database_identity_sha256
        ),
        "binding_sha256": result.binding_sha256,
        "tokenizer_sha256": result.tokenizer_sha256,
        "model_fingerprint": result.model_fingerprint,
        "device_type": result.device_type,
        "dtype": result.dtype,
        "benchmark_protocol": RESOURCE_BENCHMARK_PROTOCOL,
        "benchmark_prompt_sha256": hashlib.sha256(
            RESOURCE_BENCHMARK_PROMPT.encode("utf-8")
        ).hexdigest(),
        "prompt_ids": list(result.prompt_ids),
        "prompt_ids_sha256": _sha256(list(result.prompt_ids)),
        "warmup_iterations": RESOURCE_BENCHMARK_WARMUP_ITERATIONS,
        "measured_iterations": RESOURCE_BENCHMARK_MEASURED_ITERATIONS,
        "maximum_new_tokens": RESOURCE_BENCHMARK_MAX_NEW_TOKENS,
        "case_record": result.case_record,
        "peak_vram_bytes": result.peak_vram_bytes,
        "peak_ram_bytes": result.peak_ram_bytes,
        "iterations": iterations,
        "iterations_sha256": _sha256(iterations),
        "record": record,
    }
    payload = {**body, "receipt_sha256": _sha256(body)}
    return validate_resource_acceptance_receipt(payload)


def validate_resource_acceptance_receipt(
    payload: object,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
    expected_example_ids: tuple[str, ...] | None = None,
    expected_records: list[dict[str, object]] | None = None,
    expected_task_database_identity_sha256: str | None = None,
) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != _RECEIPT_FIELDS:
        raise ValueError("resource acceptance receipt fields are invalid")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != _sha256(body):
        raise ValueError("resource acceptance receipt integrity mismatch")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    checkpoint_step = payload.get("checkpoint_step")
    peak_vram = payload.get("peak_vram_bytes")
    peak_ram = payload.get("peak_ram_bytes")
    if (
        payload.get("schema_version") != RESOURCE_ACCEPTANCE_RECEIPT_SCHEMA
        or not _is_sha256(checkpoint_sha256)
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
        or not _is_sha256(payload.get("case_bundle_manifest_sha256"))
        or not _is_sha256(payload.get("case_inventory_sha256"))
        or not _is_sha256(payload.get("test_task_database_identity_sha256"))
        or not _is_sha256(payload.get("binding_sha256"))
        or payload.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256
        or not _is_sha256(payload.get("model_fingerprint"))
        or payload.get("device_type") != "cuda"
        or payload.get("dtype") not in {"float32", "bfloat16"}
        or payload.get("benchmark_protocol") != RESOURCE_BENCHMARK_PROTOCOL
        or payload.get("benchmark_prompt_sha256")
        != hashlib.sha256(RESOURCE_BENCHMARK_PROMPT.encode("utf-8")).hexdigest()
        or payload.get("warmup_iterations")
        != RESOURCE_BENCHMARK_WARMUP_ITERATIONS
        or payload.get("measured_iterations")
        != RESOURCE_BENCHMARK_MEASURED_ITERATIONS
        or payload.get("maximum_new_tokens")
        != RESOURCE_BENCHMARK_MAX_NEW_TOKENS
        or type(peak_vram) is not int
        or peak_vram < 1
        or type(peak_ram) is not int
        or peak_ram < 1
    ):
        raise ValueError("resource acceptance receipt lineage is invalid")
    binding = NativeTokenizerBinding.load()
    prompt_ids = build_inference_prompt(
        binding, task=_TASK, prompt=RESOURCE_BENCHMARK_PROMPT
    )
    if (
        payload.get("binding_sha256") != binding.binding_sha256
        or payload.get("model_fingerprint")
        != bound_native_65k_config(binding).fingerprint()
        or payload.get("prompt_ids") != prompt_ids
        or payload.get("prompt_ids_sha256") != _sha256(prompt_ids)
    ):
        raise ValueError("resource acceptance benchmark lineage mismatch")
    example_id, budget = _validated_resource_case(payload.get("case_record"))
    iterations = _validated_iterations(payload.get("iterations"), binding)
    oom_count, throughput = _derived_measurements(iterations)
    expected_record = {
        "example_id": example_id,
        "peak_vram_bytes": peak_vram,
        "vram_limit_bytes": budget["vram_limit_bytes"],
        "peak_ram_bytes": peak_ram,
        "ram_limit_bytes": budget["ram_limit_bytes"],
        "oom_count": oom_count,
        "inference_tokens_per_second": throughput,
        "minimum_tokens_per_second": budget["minimum_tokens_per_second"],
    }
    if (
        payload.get("iterations_sha256") != _sha256(iterations)
        or payload.get("record") != expected_record
    ):
        raise ValueError("resource acceptance measurements are inconsistent")
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError("resource acceptance checkpoint lineage mismatch")
    if expected_checkpoint_step is not None and checkpoint_step != expected_checkpoint_step:
        raise ValueError("resource acceptance checkpoint step lineage mismatch")
    if expected_example_ids is not None and expected_example_ids != (example_id,):
        raise ValueError("resource acceptance example inventory mismatch")
    if expected_records is not None and expected_records != [expected_record]:
        raise ValueError("resource acceptance records mismatch")
    if (
        expected_task_database_identity_sha256 is not None
        and payload.get("test_task_database_identity_sha256")
        != expected_task_database_identity_sha256
    ):
        raise ValueError("resource acceptance test database mismatch")
    return payload


def resource_acceptance_receipt_metrics(payload: object) -> dict[str, object]:
    receipt = validate_resource_acceptance_receipt(payload)
    record = receipt["record"]
    if type(record) is not dict:
        raise AssertionError("validated resource record is not an object")
    return {
        key: record[key]
        for key in (
            "peak_vram_bytes",
            "vram_limit_bytes",
            "peak_ram_bytes",
            "ram_limit_bytes",
            "oom_count",
            "inference_tokens_per_second",
            "minimum_tokens_per_second",
        )
    }


def publish_resource_acceptance_receipt(
    output: str | Path,
    payload: dict[str, object],
) -> Path:
    validated = validate_resource_acceptance_receipt(payload)
    rendered = (
        json.dumps(validated, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if len(rendered) > MAX_RESOURCE_RECEIPT_BYTES:
        raise ValueError("resource acceptance receipt exceeds its size bound")
    destination = Path(output).resolve()
    publish_new_files(
        ((destination, rendered),), artifact_label="resource acceptance receipt"
    )
    return destination
