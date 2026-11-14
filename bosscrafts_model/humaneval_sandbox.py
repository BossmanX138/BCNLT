"""Hash-bound HumanEval execution through the attested per-task sandbox."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict, cast

from .scientific_sandbox import PROTOCOL, SandboxExecutionResult

RESULT_SCHEMA = "bosscrafts-humaneval-sandbox-result-v1"
RECEIPT_SCHEMA = "bosscrafts-humaneval-sandbox-receipt-v1"
MAX_RESULT_LINE_BYTES = 512 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "task_index",
        "task_id",
        "completion",
        "completion_sha256",
        "source_sha256",
        "passed",
        "result",
        "sandbox",
    }
)
_SANDBOX_FIELDS = frozenset(
    {
        "protocol",
        "language",
        "status",
        "phase",
        "exit_code",
        "stdout_b64",
        "stderr_b64",
        "timed_out",
        "output_limited",
        "duration_ms",
        "container_name",
        "request_sha256",
        "response_sha256",
        "attestation",
    }
)
_ATTESTATION_FIELDS = frozenset(
    {
        "image_id",
        "image_manifest_sha256",
        "policy_sha256",
        "container_id",
        "network_mode",
        "read_only_rootfs",
        "user",
        "cap_drop",
        "no_new_privileges",
        "privileged",
        "pids_limit",
        "memory_bytes",
        "memory_swap_bytes",
        "nano_cpus",
        "ipc_mode",
        "cgroupns_mode",
        "mount_count",
        "tmpfs",
        "log_driver",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_sha256",
        "samples_sha256",
        "results_sha256",
        "metrics_sha256",
        "evaluator_source_sha256",
        "task_count",
        "passed_count",
        "pass_at_1",
        "sandbox_image_manifest_sha256",
        "sandbox_image_ids",
        "result_attestation_sha256",
    }
)


class HumanEvalTask(TypedDict):
    task_id: str
    prompt: str
    test: str
    entry_point: str


class HumanEvalSample(TypedDict):
    task_id: str
    completion: str


class SandboxExecutor(Protocol):
    def execute(
        self,
        *,
        language: str,
        source: str,
        stdin_text: str,
    ) -> SandboxExecutionResult: ...


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def task_source(task: Mapping[str, object], completion: str) -> str:
    prompt = task.get("prompt")
    tests = task.get("test")
    entry_point = task.get("entry_point")
    if (
        type(prompt) is not str
        or type(tests) is not str
        or type(entry_point) is not str
        or not entry_point
        or type(completion) is not str
    ):
        raise ValueError("HumanEval task or completion is invalid")
    return prompt + completion + "\n" + tests + "\n" + f"check({entry_point})\n"


def _result_text(result: SandboxExecutionResult) -> str:
    if (
        result.status == "passed"
        and result.phase == "run"
        and result.exit_code == 0
        and not result.timed_out
        and not result.output_limited
    ):
        return "passed"
    if result.timed_out:
        return "timed out"
    if result.output_limited:
        return "output limited"
    return f"failed: {result.status}"


def evaluate_sample(
    sandbox: SandboxExecutor,
    task: Mapping[str, object],
    sample: Mapping[str, object],
    *,
    task_index: int,
) -> dict[str, object]:
    task_id = task.get("task_id")
    completion = sample.get("completion")
    if (
        type(task_index) is not int
        or task_index < 0
        or type(task_id) is not str
        or sample.get("task_id") != task_id
        or type(completion) is not str
    ):
        raise ValueError("HumanEval task and sample do not match")
    source = task_source(task, completion)
    result = sandbox.execute(language="python", source=source, stdin_text="")
    result_text = _result_text(result)
    return {
        "schema_version": RESULT_SCHEMA,
        "task_index": task_index,
        "task_id": task_id,
        "completion": completion,
        "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "passed": result_text == "passed",
        "result": result_text,
        "sandbox": result.to_dict(),
    }


def render_results(records: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json(record) for record in records)


def load_results(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline(MAX_RESULT_LINE_BYTES + 1)
            if not raw:
                return rows
            if len(raw) > MAX_RESULT_LINE_BYTES or not raw.endswith(b"\n"):
                raise ValueError("HumanEval sandbox result has invalid bounded framing")
            try:
                row = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise ValueError("HumanEval sandbox result is invalid JSON") from error
            if type(row) is not dict:
                raise ValueError("HumanEval sandbox result must be an object")
            rows.append(row)


def validate_result(
    record: Mapping[str, object],
    task: Mapping[str, object],
    sample: Mapping[str, object],
    *,
    task_index: int,
) -> None:
    if set(record) != _RESULT_FIELDS:
        raise ValueError("HumanEval sandbox result fields are invalid")
    task_id = task.get("task_id")
    completion = sample.get("completion")
    if (
        record.get("schema_version") != RESULT_SCHEMA
        or type(task_id) is not str
        or sample.get("task_id") != task_id
        or type(completion) is not str
        or record.get("task_index") != task_index
        or record.get("task_id") != task_id
        or record.get("completion") != completion
        or record.get("completion_sha256")
        != hashlib.sha256(completion.encode("utf-8")).hexdigest()
        or record.get("source_sha256")
        != hashlib.sha256(task_source(task, completion).encode("utf-8")).hexdigest()
    ):
        raise ValueError("HumanEval sandbox result task lineage mismatch")
    sandbox = record.get("sandbox")
    if type(sandbox) is not dict or set(sandbox) != _SANDBOX_FIELDS:
        raise ValueError("HumanEval sandbox execution fields are invalid")
    attestation = sandbox.get("attestation")
    if type(attestation) is not dict or set(attestation) != _ATTESTATION_FIELDS:
        raise ValueError("HumanEval sandbox attestation fields are invalid")
    source = task_source(task, completion)
    request = canonical_json(
        {
            "protocol": PROTOCOL,
            "language": "python",
            "source": source,
            "stdin": "",
        }
    )
    request_sha256 = hashlib.sha256(request).hexdigest()
    response = {
        key: sandbox[key]
        for key in (
            "protocol",
            "language",
            "status",
            "phase",
            "exit_code",
            "stdout_b64",
            "stderr_b64",
            "timed_out",
            "output_limited",
            "duration_ms",
        )
    }
    if (
        sandbox.get("protocol") != PROTOCOL
        or sandbox.get("language") != "python"
        or sandbox.get("request_sha256") != request_sha256
        or sandbox.get("response_sha256") != hashlib.sha256(canonical_json(response)).hexdigest()
        or type(sandbox.get("container_name")) is not str
        or not str(sandbox["container_name"]).startswith("bcnlt-science-")
    ):
        raise ValueError("HumanEval sandbox execution identity mismatch")
    for field in ("stdout_b64", "stderr_b64"):
        try:
            decoded = base64.b64decode(sandbox[field], validate=True)
        except (TypeError, ValueError) as error:
            raise ValueError("HumanEval sandbox output encoding is invalid") from error
        if len(decoded) > 64 * 1024:
            raise ValueError("HumanEval sandbox output exceeds its bound")
    _validate_attestation(attestation, request_sha256)
    passed = (
        sandbox.get("status") == "passed"
        and sandbox.get("phase") == "run"
        and sandbox.get("exit_code") == 0
        and sandbox.get("timed_out") is False
        and sandbox.get("output_limited") is False
    )
    expected_result = (
        "passed"
        if passed
        else "timed out"
        if sandbox.get("timed_out") is True
        else "output limited"
        if sandbox.get("output_limited") is True
        else f"failed: {sandbox.get('status')}"
    )
    if record.get("passed") is not passed or record.get("result") != expected_result:
        raise ValueError("HumanEval sandbox pass result is inconsistent")


def build_receipt(
    *,
    dataset_sha256: str,
    samples_sha256: str,
    results_bytes: bytes,
    metrics_bytes: bytes,
    evaluator_source_sha256: str,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    for digest in (dataset_sha256, samples_sha256, evaluator_source_sha256):
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("HumanEval receipt input digest is invalid")
    if not records:
        raise ValueError("HumanEval receipt requires at least one result")
    passed_count = sum(record.get("passed") is True for record in records)
    attestations = [cast(dict[str, object], record["sandbox"])["attestation"] for record in records]
    attestation_bytes = canonical_json(attestations)
    manifest_ids = {
        cast(dict[str, object], attestation)["image_manifest_sha256"]
        for attestation in attestations
    }
    image_ids = sorted(
        {
            str(cast(dict[str, object], attestation)["image_id"])
            for attestation in attestations
        }
    )
    if len(manifest_ids) != 1:
        raise ValueError("HumanEval sandbox manifest identity changed during evaluation")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "dataset_sha256": dataset_sha256,
        "samples_sha256": samples_sha256,
        "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "evaluator_source_sha256": evaluator_source_sha256,
        "task_count": len(records),
        "passed_count": passed_count,
        "pass_at_1": passed_count / len(records),
        "sandbox_image_manifest_sha256": next(iter(manifest_ids)),
        "sandbox_image_ids": image_ids,
        "result_attestation_sha256": hashlib.sha256(attestation_bytes).hexdigest(),
    }


def validate_receipt(
    receipt: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    *,
    dataset_sha256: str,
    samples_sha256: str,
    results_bytes: bytes,
    metrics_bytes: bytes,
    evaluator_source_sha256: str,
) -> None:
    if set(receipt) != _RECEIPT_FIELDS or receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("HumanEval sandbox receipt fields are invalid")
    expected = build_receipt(
        dataset_sha256=dataset_sha256,
        samples_sha256=samples_sha256,
        results_bytes=results_bytes,
        metrics_bytes=metrics_bytes,
        evaluator_source_sha256=evaluator_source_sha256,
        records=records,
    )
    if dict(receipt) != expected:
        raise ValueError("HumanEval sandbox receipt is inconsistent")


def _validate_attestation(attestation: dict[str, object], request_sha256: str) -> None:
    container_id = attestation.get("container_id")
    cap_drop = attestation.get("cap_drop")
    if (
        type(container_id) is not str
        or _SHA256.fullmatch(container_id) is None
        or type(attestation.get("image_id")) is not str
        or not str(attestation["image_id"]).startswith("sha256:")
        or type(attestation.get("image_manifest_sha256")) is not str
        or _SHA256.fullmatch(str(attestation["image_manifest_sha256"])) is None
        or attestation.get("network_mode") != "none"
        or attestation.get("read_only_rootfs") is not True
        or attestation.get("user") != "65532:65532"
        or cap_drop != ["ALL"]
        or attestation.get("no_new_privileges") is not True
        or attestation.get("privileged") is not False
        or attestation.get("pids_limit") != 64
        or attestation.get("memory_bytes") != 512 * 1024 * 1024
        or attestation.get("memory_swap_bytes") != 512 * 1024 * 1024
        or attestation.get("nano_cpus") != 500_000_000
        or attestation.get("ipc_mode") != "none"
        or attestation.get("cgroupns_mode") != "private"
        or attestation.get("mount_count") != 0
        or attestation.get("tmpfs")
        != "rw,exec,nosuid,nodev,size=134217728,mode=1777"
        or attestation.get("log_driver") != "none"
    ):
        raise ValueError("HumanEval sandbox isolation policy is invalid")
    policy = {
        "image_id": attestation["image_id"],
        "network_mode": "none",
        "read_only_rootfs": True,
        "user": "65532:65532",
        "cap_drop": ("ALL",),
        "no_new_privileges": True,
        "privileged": False,
        "pids_limit": 64,
        "memory_bytes": 512 * 1024 * 1024,
        "memory_swap_bytes": 512 * 1024 * 1024,
        "nano_cpus": 500_000_000,
        "ipc_mode": "none",
        "cgroupns_mode": "private",
        "mount_count": 0,
        "tmpfs": "rw,exec,nosuid,nodev,size=134217728,mode=1777",
        "log_driver": "none",
        "attach_stdin": True,
        "open_stdin": True,
        "device_count": 0,
        "device_request_count": 0,
        "port_binding_count": 0,
        "restart_policy": "no",
        "sandbox_label": "true",
        "request_label": request_sha256,
    }
    if attestation.get("policy_sha256") != hashlib.sha256(canonical_json(policy)).hexdigest():
        raise ValueError("HumanEval sandbox policy digest is inconsistent")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate HumanEval sandbox key: {key}")
        result[key] = value
    return result
