from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bosscrafts_model.code_acceptance import (
    CodeAcceptanceBatchResult,
    build_code_acceptance_receipt,
    publish_code_acceptance_receipt,
    validate_code_acceptance_receipt,
)
from bosscrafts_model.humaneval_sandbox import canonical_json
from bosscrafts_model.scientific_sandbox import (
    PROTOCOL,
    SandboxAttestation,
    SandboxExecutionResult,
)
from bosscrafts_model.tokenizer import (
    EXPECTED_BINDING_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    NativeTokenizerBinding,
)


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


def _execution(
    *,
    language: str,
    source: str,
    stdin_text: str,
    stdout: bytes,
) -> SandboxExecutionResult:
    request_sha256 = hashlib.sha256(
        canonical_json(
            {
                "protocol": PROTOCOL,
                "language": language,
                "source": source,
                "stdin": stdin_text,
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
        "language": language,
        "status": "passed",
        "phase": "run",
        "exit_code": 0,
        "stdout_b64": base64.b64encode(stdout).decode("ascii"),
        "stderr_b64": "",
        "timed_out": False,
        "output_limited": False,
        "duration_ms": 1,
    }
    return SandboxExecutionResult(
        protocol=PROTOCOL,
        language=language,
        status="passed",
        phase="run",
        exit_code=0,
        stdout_bytes=stdout,
        stderr_bytes=b"",
        timed_out=False,
        output_limited=False,
        duration_ms=1,
        container_name="bcnlt-science-code-fixture",
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


def _result(category: str) -> CodeAcceptanceBatchResult:
    binding = NativeTokenizerBinding.load()
    source = "print(int(input()) + 1)\n"
    payload_ids = binding.encode_payload(source)
    structural = binding.structural_token_ids
    generated_ids = [
        structural["<|bcnlt:stream:begin|>"],
        structural["<|bcnlt:mode:raw_code|>"],
        *payload_ids,
        structural["<|bcnlt:stream:end|>"],
        structural["<|bcnlt:task:end|>"],
    ]
    stdin_text = "1\n" if category == "execution_correctness" else ""
    expected_stdout = "2\n" if category == "execution_correctness" else None
    sandbox = _execution(
        language="python",
        source=source,
        stdin_text=stdin_text,
        stdout=b"2\n" if category == "execution_correctness" else b"",
    ).to_dict()
    example_id = "e" * 64
    detail: dict[str, object] = {
        "example_id": example_id,
        "task_example_id": "a" * 64,
        "language": "python",
        "task_input_ids_sha256": "b" * 64,
        "generated_ids": generated_ids,
        "generated_ids_sha256": _sha256(generated_ids),
        "protocol_complete": True,
        "payload_ids_sha256": _sha256(payload_ids),
        "generated_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "stdin": stdin_text,
        "expected_stdout": expected_stdout,
        "sandbox": sandbox,
    }
    record: dict[str, object]
    if category == "raw_code":
        record = {
            "example_id": example_id,
            "language": "python",
            "exact_roundtrip": True,
            "compile_passed": True,
            "silent_fallback": False,
        }
    else:
        record = {
            "example_id": example_id,
            "language": "python",
            "sandbox_executed": True,
            "expected_output_exact": True,
            "compile_passed": True,
            "sandbox_escape": False,
        }
    return CodeAcceptanceBatchResult(
        category=category,
        case_bundle_manifest_sha256="c" * 64,
        case_inventory_sha256="d" * 64,
        task_database_identity_sha256="f" * 64,
        binding_sha256=EXPECTED_BINDING_SHA256,
        tokenizer_sha256=EXPECTED_TOKENIZER_SHA256,
        records=(record,),
        details=(detail,),
    )


@pytest.mark.parametrize("category", ["raw_code", "execution_correctness"])
def test_builds_validates_and_publishes_code_receipt(
    tmp_path: Path,
    category: str,
) -> None:
    receipt = build_code_acceptance_receipt(
        _result(category),
        checkpoint_sha256="3" * 64,
        checkpoint_step=17,
    )
    assert (
        validate_code_acceptance_receipt(
            receipt,
            expected_category=category,
            expected_checkpoint_sha256="3" * 64,
            expected_checkpoint_step=17,
            expected_example_ids=("e" * 64,),
            expected_records=[receipt["records"][0]],
            expected_task_database_identity_sha256="f" * 64,
        )
        == receipt
    )
    output = tmp_path / f"{category}.json"
    assert publish_code_acceptance_receipt(output, receipt) == output.resolve()
    with pytest.raises(ValueError, match="already exists"):
        publish_code_acceptance_receipt(output, receipt)


def test_receipt_recomputes_execution_result_instead_of_trusting_boolean() -> None:
    receipt = build_code_acceptance_receipt(
        _result("execution_correctness"),
        checkpoint_sha256="3" * 64,
        checkpoint_step=17,
    )
    forged = json.loads(json.dumps(receipt))
    forged["records"][0]["expected_output_exact"] = False
    forged["records_sha256"] = _sha256(forged["records"])
    body = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = _sha256(body)

    with pytest.raises(ValueError, match="inconsistent"):
        validate_code_acceptance_receipt(forged)


def test_receipt_rejects_sandbox_policy_substitution_even_when_rehashed() -> None:
    receipt = build_code_acceptance_receipt(
        _result("raw_code"),
        checkpoint_sha256="3" * 64,
        checkpoint_step=17,
    )
    forged = json.loads(json.dumps(receipt))
    forged["details"][0]["sandbox"]["attestation"]["network_mode"] = "bridge"
    forged["details_sha256"] = _sha256(forged["details"])
    body = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = _sha256(body)

    with pytest.raises(ValueError, match="isolation policy"):
        validate_code_acceptance_receipt(forged)


def test_code_acceptance_runner_exposes_lineage_and_runtime_inputs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "run_code_acceptance.py"),
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
        "--sandbox-manifest",
        "--max-new-tokens",
        "--output",
    ):
        assert option in completed.stdout
