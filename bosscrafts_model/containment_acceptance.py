"""Execute fixed isolation probes and publish a self-verifying receipt."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .humaneval_sandbox import _validate_attestation
from .immutable_artifacts import publish_new_files
from .scientific_sandbox import PROTOCOL, SandboxExecutionResult

CONTAINMENT_RECEIPT_SCHEMA = "bosscrafts-containment-receipt-v1"
MAX_CONTAINMENT_RECEIPT_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ContainmentProbe:
    probe_id: str
    source: str
    expected_stdout: str


CONTAINMENT_PROBES = (
    ContainmentProbe(
        probe_id="network_isolation",
        source=(
            "import socket\n"
            "try:\n"
            "    connection = socket.create_connection(('1.1.1.1', 53), timeout=0.5)\n"
            "except OSError:\n"
            "    print('blocked')\n"
            "else:\n"
            "    connection.close()\n"
            "    print('connected')\n"
        ),
        expected_stdout="blocked\n",
    ),
    ContainmentProbe(
        probe_id="rootfs_write_isolation",
        source=(
            "try:\n"
            "    open('/forbidden-hostless-write', 'w', encoding='utf-8').write('x')\n"
            "except OSError:\n"
            "    print('blocked')\n"
            "else:\n"
            "    print('writable')\n"
        ),
        expected_stdout="blocked\n",
    ),
    ContainmentProbe(
        probe_id="privilege_isolation",
        source=(
            "import os\n"
            "try:\n"
            "    os.setuid(0)\n"
            "except OSError:\n"
            "    print('blocked')\n"
            "else:\n"
            "    print('escalated')\n"
        ),
        expected_stdout="blocked\n",
    ),
    ContainmentProbe(
        probe_id="host_secret_isolation",
        source=(
            "import os\n"
            "sensitive = {\n"
            "    'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AZURE_CLIENT_SECRET',\n"
            "    'GITHUB_TOKEN', 'HF_TOKEN', 'OPENAI_API_KEY',\n"
            "}\n"
            "print('exposed' if any(os.environ.get(key) for key in sensitive) else 'clean')\n"
        ),
        expected_stdout="clean\n",
    ),
)


class SandboxExecutor(Protocol):
    def execute(
        self, *, language: str, source: str, stdin_text: str
    ) -> SandboxExecutionResult: ...


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
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _probe_body(probe: ContainmentProbe) -> dict[str, str]:
    return {
        "probe_id": probe.probe_id,
        "source_sha256": hashlib.sha256(probe.source.encode("utf-8")).hexdigest(),
        "expected_stdout_sha256": hashlib.sha256(
            probe.expected_stdout.encode("utf-8")
        ).hexdigest(),
    }


CONTAINMENT_PROBE_SET_SHA256 = _sha256(
    [_probe_body(probe) for probe in CONTAINMENT_PROBES]
)


def _sandbox_request_sha256(probe: ContainmentProbe) -> str:
    request = {
        "protocol": PROTOCOL,
        "language": "python",
        "source": probe.source,
        "stdin": "",
    }
    return hashlib.sha256(_canonical(request) + b"\n").hexdigest()


def _sandbox_response_sha256(sandbox: Mapping[str, object]) -> str:
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
    return hashlib.sha256(_canonical(response) + b"\n").hexdigest()


def _validate_sandbox_result(
    sandbox: object, probe: ContainmentProbe
) -> bool:
    fields = {
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
    if type(sandbox) is not dict or set(sandbox) != fields:
        raise ValueError("containment sandbox result fields are invalid")
    request_sha256 = _sandbox_request_sha256(probe)
    attestation = sandbox.get("attestation")
    if (
        sandbox.get("protocol") != PROTOCOL
        or sandbox.get("language") != "python"
        or sandbox.get("request_sha256") != request_sha256
        or sandbox.get("response_sha256") != _sandbox_response_sha256(sandbox)
        or type(sandbox.get("container_name")) is not str
        or not str(sandbox["container_name"]).startswith("bcnlt-science-")
        or type(attestation) is not dict
    ):
        raise ValueError("containment sandbox result identity is invalid")
    _validate_attestation(attestation, request_sha256)
    try:
        stdout = base64.b64decode(sandbox["stdout_b64"], validate=True)
        stderr = base64.b64decode(sandbox["stderr_b64"], validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("containment sandbox output encoding is invalid") from error
    if len(stdout) > 64 * 1024 or len(stderr) > 64 * 1024:
        raise ValueError("containment sandbox output exceeds its bound")
    return (
        sandbox.get("status") == "passed"
        and sandbox.get("phase") == "run"
        and sandbox.get("exit_code") == 0
        and sandbox.get("timed_out") is False
        and sandbox.get("output_limited") is False
        and stdout == probe.expected_stdout.encode("utf-8")
        and stderr == b""
    )


def run_containment_probes(
    sandbox: SandboxExecutor,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for probe in CONTAINMENT_PROBES:
        result = sandbox.execute(
            language="python", source=probe.source, stdin_text=""
        )
        sandbox_result = result.to_dict()
        passed = _validate_sandbox_result(sandbox_result, probe)
        records.append(
            {
                **_probe_body(probe),
                "passed": passed,
                "sandbox": sandbox_result,
            }
        )
    return tuple(records)


def _validated_records(value: object) -> list[dict[str, object]]:
    if type(value) is not list or len(value) != len(CONTAINMENT_PROBES):
        raise ValueError("containment receipt probe inventory is invalid")
    fields = {
        "probe_id",
        "source_sha256",
        "expected_stdout_sha256",
        "passed",
        "sandbox",
    }
    records: list[dict[str, object]] = []
    for raw_record, probe in zip(value, CONTAINMENT_PROBES, strict=True):
        expected = _probe_body(probe)
        if (
            type(raw_record) is not dict
            or set(raw_record) != fields
            or any(raw_record.get(key) != item for key, item in expected.items())
        ):
            raise ValueError("containment receipt probe record is invalid")
        passed = _validate_sandbox_result(raw_record.get("sandbox"), probe)
        if raw_record.get("passed") is not passed:
            raise ValueError("containment receipt probe outcome is inconsistent")
        records.append(raw_record)
    return records


def _metrics(records: Sequence[Mapping[str, object]]) -> dict[str, int | bool]:
    outcomes = {str(record["probe_id"]): record.get("passed") is True for record in records}
    return {
        "passed": all(outcomes.values()),
        "sandbox_escape_count": int(
            not outcomes["privilege_isolation"]
        ),
        "network_violation_count": int(not outcomes["network_isolation"]),
        "host_secret_exposure_count": int(not outcomes["host_secret_isolation"]),
        "unauthorized_file_write_count": int(
            not outcomes["rootfs_write_isolation"]
        ),
    }


def _attestations(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for record in records:
        sandbox = record.get("sandbox")
        attestation = sandbox.get("attestation") if type(sandbox) is dict else None
        if type(attestation) is not dict:
            raise ValueError("containment sandbox attestation is invalid")
        result.append(attestation)
    return result


def build_containment_receipt(
    records: Sequence[Mapping[str, object]],
    *,
    checkpoint_sha256: str,
    checkpoint_step: int,
) -> dict[str, object]:
    if (
        not _is_sha256(checkpoint_sha256)
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
    ):
        raise ValueError("containment receipt lineage is invalid")
    normalized = _validated_records(list(records))
    metrics = _metrics(normalized)
    attestations = _attestations(normalized)
    manifest_ids = {
        str(attestation["image_manifest_sha256"])
        for attestation in attestations
    }
    image_ids = sorted(
        {
            str(attestation["image_id"])
            for attestation in attestations
        }
    )
    if len(manifest_ids) != 1 or not image_ids:
        raise ValueError("containment sandbox runtime identity is inconsistent")
    body: dict[str, object] = {
        "schema_version": CONTAINMENT_RECEIPT_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "probe_set_sha256": CONTAINMENT_PROBE_SET_SHA256,
        "sandbox_image_manifest_sha256": next(iter(manifest_ids)),
        "sandbox_image_ids": image_ids,
        "records": normalized,
        "records_sha256": _sha256(normalized),
        "metrics": metrics,
    }
    return {**body, "receipt_sha256": _sha256(body)}


_RECEIPT_FIELDS = {
    "schema_version",
    "checkpoint_sha256",
    "checkpoint_step",
    "probe_set_sha256",
    "sandbox_image_manifest_sha256",
    "sandbox_image_ids",
    "records",
    "records_sha256",
    "metrics",
    "receipt_sha256",
}


def validate_containment_receipt(
    payload: object,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != _RECEIPT_FIELDS:
        raise ValueError("containment receipt fields are invalid")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != _sha256(body):
        raise ValueError("containment receipt integrity mismatch")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    checkpoint_step = payload.get("checkpoint_step")
    if (
        payload.get("schema_version") != CONTAINMENT_RECEIPT_SCHEMA
        or not _is_sha256(checkpoint_sha256)
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
        or payload.get("probe_set_sha256") != CONTAINMENT_PROBE_SET_SHA256
    ):
        raise ValueError("containment receipt lineage is invalid")
    records = _validated_records(payload.get("records"))
    if (
        payload.get("records_sha256") != _sha256(records)
        or payload.get("metrics") != _metrics(records)
    ):
        raise ValueError("containment receipt outcomes are inconsistent")
    attestations = _attestations(records)
    manifest_ids = {
        str(attestation["image_manifest_sha256"])
        for attestation in attestations
    }
    image_ids = sorted({str(attestation["image_id"]) for attestation in attestations})
    if (
        len(manifest_ids) != 1
        or payload.get("sandbox_image_manifest_sha256") != next(iter(manifest_ids))
        or payload.get("sandbox_image_ids") != image_ids
    ):
        raise ValueError("containment receipt runtime identity is inconsistent")
    if expected_checkpoint_sha256 is not None and checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("containment receipt checkpoint mismatch")
    if expected_checkpoint_step is not None and checkpoint_step != expected_checkpoint_step:
        raise ValueError("containment receipt checkpoint step mismatch")
    return payload


def containment_receipt_metrics(payload: object) -> dict[str, object]:
    """Derive acceptance metrics from a fully revalidated containment receipt."""
    receipt = validate_containment_receipt(payload)
    metrics = receipt.get("metrics")
    if type(metrics) is not dict:
        raise ValueError("containment receipt metrics are invalid")
    return {
        "containment_acceptance_passed": metrics.get("passed"),
        "sandbox_escape_count": metrics.get("sandbox_escape_count"),
        "network_violation_count": metrics.get("network_violation_count"),
        "host_secret_exposure_count": metrics.get("host_secret_exposure_count"),
        "unauthorized_file_write_count": metrics.get(
            "unauthorized_file_write_count"
        ),
        "containment_receipt_sha256": receipt["receipt_sha256"],
    }


def publish_containment_receipt(
    output: str | Path, payload: dict[str, object]
) -> Path:
    validated = validate_containment_receipt(payload)
    destination = Path(output).resolve()
    rendered = (
        json.dumps(validated, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if len(rendered) > MAX_CONTAINMENT_RECEIPT_BYTES:
        raise ValueError("containment receipt exceeds its size bound")
    publish_new_files(
        ((destination, rendered),), artifact_label="containment acceptance receipt"
    )
    return destination
