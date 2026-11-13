from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bosscrafts_model.containment_acceptance import (
    CONTAINMENT_PROBE_SET_SHA256,
    CONTAINMENT_PROBES,
    build_containment_receipt,
    publish_containment_receipt,
    run_containment_probes,
    validate_containment_receipt,
)
from bosscrafts_model.humaneval_sandbox import canonical_json
from bosscrafts_model.scientific_sandbox import (
    PROTOCOL,
    SandboxAttestation,
    SandboxExecutionResult,
)


class _FakeSandbox:
    def execute(
        self, *, language: str, source: str, stdin_text: str
    ) -> SandboxExecutionResult:
        probe = next(item for item in CONTAINMENT_PROBES if item.source == source)
        assert language == "python"
        assert stdin_text == ""
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
        policy = {
            "image_id": "sha256:" + "a" * 64,
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
        stdout = probe.expected_stdout.encode("utf-8")
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
            container_name="bcnlt-science-fixture",
            request_sha256=request_sha256,
            response_sha256=hashlib.sha256(canonical_json(response)).hexdigest(),
            attestation=SandboxAttestation(
                image_id="sha256:" + "a" * 64,
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


def test_builds_validated_containment_receipt_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    records = run_containment_probes(_FakeSandbox())
    receipt = build_containment_receipt(
        records,
        checkpoint_sha256="d" * 64,
        checkpoint_step=17,
    )

    assert receipt["probe_set_sha256"] == CONTAINMENT_PROBE_SET_SHA256
    assert receipt["metrics"] == {
        "passed": True,
        "sandbox_escape_count": 0,
        "network_violation_count": 0,
        "host_secret_exposure_count": 0,
        "unauthorized_file_write_count": 0,
    }
    assert validate_containment_receipt(
        receipt,
        expected_checkpoint_sha256="d" * 64,
        expected_checkpoint_step=17,
    ) == receipt
    output = tmp_path / "containment.json"
    assert publish_containment_receipt(output, receipt) == output.resolve()
    with pytest.raises(ValueError, match="already exists"):
        publish_containment_receipt(output, receipt)

    forged = json.loads(json.dumps(receipt))
    forged["records"][0]["passed"] = False
    body = {
        key: value for key, value in forged.items() if key != "receipt_sha256"
    }
    forged["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="outcome is inconsistent"):
        validate_containment_receipt(forged)


def test_containment_acceptance_script_exposes_checkpoint_binding() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "run_containment_acceptance.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--sandbox-manifest" in completed.stdout
    assert "--checkpoint" in completed.stdout
    assert "--checkpoint-step" in completed.stdout
    assert "--output" in completed.stdout
