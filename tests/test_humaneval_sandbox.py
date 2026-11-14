from __future__ import annotations

import copy
import hashlib

import pytest

from bosscrafts_model.humaneval_sandbox import (
    build_receipt,
    canonical_json,
    evaluate_sample,
    render_results,
    task_source,
    validate_receipt,
    validate_result,
)
from bosscrafts_model.scientific_sandbox import (
    PROTOCOL,
    SandboxAttestation,
    SandboxExecutionResult,
    validate_sandbox_execution_payload,
)


def _execution(source: str) -> SandboxExecutionResult:
    request_sha256 = hashlib.sha256(
        canonical_json(
            {
                "protocol": PROTOCOL,
                "language": "python",
                "source": source,
                "stdin": "",
            }
        )
    ).hexdigest()
    image_id = "sha256:" + "a" * 64
    policy = {
        "image_id": image_id,
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
    response = {
        "protocol": PROTOCOL,
        "language": "python",
        "status": "passed",
        "phase": "run",
        "exit_code": 0,
        "stdout_b64": "",
        "stderr_b64": "",
        "timed_out": False,
        "output_limited": False,
        "duration_ms": 1,
    }
    return SandboxExecutionResult(
        protocol=PROTOCOL,
        language="python",
        status="passed",
        phase="run",
        exit_code=0,
        stdout_bytes=b"",
        stderr_bytes=b"",
        timed_out=False,
        output_limited=False,
        duration_ms=1,
        container_name="bcnlt-science-fixture",
        request_sha256=request_sha256,
        response_sha256=hashlib.sha256(canonical_json(response)).hexdigest(),
        attestation=SandboxAttestation(
            image_id=image_id,
            image_manifest_sha256="b" * 64,
            policy_sha256=hashlib.sha256(canonical_json(policy)).hexdigest(),
            container_id="c" * 64,
            network_mode="none",
            read_only_rootfs=True,
            user="65532:65532",
            cap_drop=("ALL",),
            no_new_privileges=True,
            privileged=False,
            pids_limit=64,
            memory_bytes=512 * 1024 * 1024,
            memory_swap_bytes=512 * 1024 * 1024,
            nano_cpus=500_000_000,
            ipc_mode="none",
            cgroupns_mode="private",
            mount_count=0,
            tmpfs="rw,exec,nosuid,nodev,size=134217728,mode=1777",
            log_driver="none",
        ),
    )


class _Sandbox:
    def execute(self, *, language: str, source: str, stdin_text: str):
        assert (language, stdin_text) == ("python", "")
        return _execution(source)


def test_common_sandbox_payload_validator_rejects_identity_and_policy_tamper(
) -> None:
    source = "print('ok')\n"
    payload = _execution(source).to_dict()
    stdout, stderr = validate_sandbox_execution_payload(
        payload,
        language="python",
        source=source,
        stdin_text="",
    )
    assert (stdout, stderr) == (b"", b"")

    forged_request = copy.deepcopy(payload)
    forged_request["request_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="identity"):
        validate_sandbox_execution_payload(
            forged_request,
            language="python",
            source=source,
            stdin_text="",
        )

    forged_policy = copy.deepcopy(payload)
    forged_policy["attestation"]["network_mode"] = "bridge"
    with pytest.raises(ValueError, match="isolation policy"):
        validate_sandbox_execution_payload(
            forged_policy,
            language="python",
            source=source,
            stdin_text="",
        )

    forged_status = copy.deepcopy(payload)
    forged_status["status"] = "runtime_failed"
    with pytest.raises(ValueError, match="status"):
        validate_sandbox_execution_payload(
            forged_status,
            language="python",
            source=source,
            stdin_text="",
        )


def test_result_is_bound_to_source_sample_and_sandbox_policy() -> None:
    task = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\n",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
        "entry_point": "add",
    }
    sample = {"task_id": "HumanEval/0", "completion": "    return a + b\n"}
    record = evaluate_sample(_Sandbox(), task, sample, task_index=0)

    validate_result(record, task, sample, task_index=0)
    assert record["passed"] is True
    assert record["source_sha256"] == hashlib.sha256(
        task_source(task, sample["completion"]).encode("utf-8")
    ).hexdigest()

    record["passed"] = False
    with pytest.raises(ValueError, match="pass result is inconsistent"):
        validate_result(record, task, sample, task_index=0)


def test_receipt_recomputes_every_evidence_hash() -> None:
    task = {
        "task_id": "HumanEval/0",
        "prompt": "def answer():\n",
        "test": "def check(candidate):\n    assert candidate() == 42",
        "entry_point": "answer",
    }
    sample = {"task_id": "HumanEval/0", "completion": "    return 42\n"}
    record = evaluate_sample(_Sandbox(), task, sample, task_index=0)
    results = render_results([record])
    metrics = b'{"pass@1": 1.0}\n'
    receipt = build_receipt(
        dataset_sha256="d" * 64,
        samples_sha256="e" * 64,
        results_bytes=results,
        metrics_bytes=metrics,
        evaluator_source_sha256="f" * 64,
        records=[record],
    )

    validate_receipt(
        receipt,
        [record],
        dataset_sha256="d" * 64,
        samples_sha256="e" * 64,
        results_bytes=results,
        metrics_bytes=metrics,
        evaluator_source_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="receipt is inconsistent"):
        validate_receipt(
            receipt,
            [record],
            dataset_sha256="d" * 64,
            samples_sha256="e" * 64,
            results_bytes=results,
            metrics_bytes=b'{"pass@1": 0.0}\n',
            evaluator_source_sha256="f" * 64,
        )
