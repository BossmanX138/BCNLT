from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
VERIFIER_ID = "bcnlt-bounded-projection"
VERIFIER_VERSION = "1"
SANDBOX_EVIDENCE_SCHEMA_VERSION = 3
SANDBOX_VERIFIER_ID = "bcnlt-sandbox-backed-projection"
SANDBOX_VERIFIER_VERSION = "1"
_ALLOWED_ORACLE_KINDS = {"human_authored", "language_spec", "trusted_test_vector"}
_MAX_SOURCE_BYTES = 64 * 1024
_MAX_FIXTURE_BYTES = 16 * 1024
_MAX_DEPTH = 8
_MAX_COLLECTION_ITEMS = 128
_MAX_STRING_BYTES = 4096


class VerificationStatus(str, Enum):
    PASSED = "passed"
    CANDIDATE_FAILED = "candidate_failed"
    CONTROL_FAILED = "control_failed"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"


@dataclass(frozen=True)
class HypothesisProjection:
    signature: str
    revision: int
    target_event: str
    target_ir_kind: str
    language: str
    semantic_spec_sha256: str
    extension_name: str | None = None

    def __post_init__(self) -> None:
        if not self.signature.strip():
            raise ValueError("Hypothesis signature is required")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("Hypothesis revision must be a positive integer")
        for name, value in (
            ("target event", self.target_event),
            ("target IR kind", self.target_ir_kind),
            ("language", self.language),
            ("semantic spec hash", self.semantic_spec_sha256),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Hypothesis {name} is required")

    @property
    def fingerprint(self) -> str:
        return _sha256(_canonical_json(asdict(self)))


@dataclass(frozen=True)
class CodingExperimentFixture:
    experiment_id: str
    fixture_id: str
    fixture_version: int
    domain: str
    document_identity: str
    oracle_id: str
    oracle_kind: str
    oracle_version: str
    operation: str
    inputs: dict[str, Any]
    expected_output: Any

    def __post_init__(self) -> None:
        for name, value in (
            ("experiment id", self.experiment_id),
            ("fixture id", self.fixture_id),
            ("domain", self.domain),
            ("document identity", self.document_identity),
            ("oracle id", self.oracle_id),
            ("oracle version", self.oracle_version),
            ("operation", self.operation),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Fixture {name} is required")
        if (
            not isinstance(self.fixture_version, int)
            or isinstance(self.fixture_version, bool)
            or self.fixture_version < 1
        ):
            raise ValueError("Fixture version must be a positive integer")
        if self.oracle_kind not in _ALLOWED_ORACLE_KINDS:
            raise ValueError("Fixture oracle kind must be independently authored")
        lowered = self.oracle_id.casefold()
        if "extractor" in lowered or "hypothesis" in lowered:
            raise ValueError("Extractor- or hypothesis-derived oracles are not independent")

    @property
    def fingerprint(self) -> str:
        return _sha256(_canonical_json(asdict(self)))

    @property
    def oracle_fingerprint(self) -> str:
        payload = {
            "oracle_id": self.oracle_id,
            "oracle_kind": self.oracle_kind,
            "oracle_version": self.oracle_version,
            "expected_output": self.expected_output,
        }
        return _sha256(_canonical_json(payload))

    @property
    def stimulus_fingerprint(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "domain": self.domain,
                    "oracle_id": self.oracle_id,
                    "oracle_kind": self.oracle_kind,
                    "oracle_version": self.oracle_version,
                    "operation": self.operation,
                    "inputs": self.inputs,
                    "expected_output": self.expected_output,
                }
            )
        )


@dataclass(frozen=True)
class SandboxExecutionEvidence:
    schema_version: int
    protocol: str
    request_sha256: str
    response_sha256: str
    image_id: str
    image_manifest_sha256: str
    policy_sha256: str
    container_id: str
    language: str
    status: str
    phase: str
    exit_code: int | None
    stdout_sha256: str
    stderr_sha256: str
    timed_out: bool
    output_limited: bool
    duration_ms: int
    actual_output: Any
    network_mode: str
    read_only_rootfs: bool
    user: str
    cap_drop: tuple[str, ...]
    no_new_privileges: bool
    privileged: bool
    pids_limit: int
    memory_bytes: int
    memory_swap_bytes: int
    nano_cpus: int
    ipc_mode: str
    cgroupns_mode: str
    mount_count: int
    tmpfs: str
    log_driver: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.protocol != "bosscrafts-bcnlt-sandbox-v1":
            raise ValueError("Sandbox execution evidence identity mismatch")
        for name, value in (
            ("request hash", self.request_sha256),
            ("response hash", self.response_sha256),
            ("image manifest hash", self.image_manifest_sha256),
            ("policy hash", self.policy_sha256),
            ("stdout hash", self.stdout_sha256),
            ("stderr hash", self.stderr_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError(f"Sandbox execution {name} is invalid")
        if not isinstance(self.image_id, str) or not self.image_id.startswith("sha256:") or not _is_sha256(self.image_id[7:]):
            raise ValueError("Sandbox execution image ID is invalid")
        if not _is_sha256(self.container_id):
            raise ValueError("Sandbox execution container ID is invalid")
        if not isinstance(self.language, str) or not self.language:
            raise ValueError("Sandbox execution language is required")
        if self.status not in {
            "passed", "compile_failed", "runtime_failed", "timed_out", "output_limited"
        } or self.phase not in {"compile", "run"}:
            raise ValueError("Sandbox execution status is invalid")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("Sandbox execution exit code is invalid")
        if type(self.timed_out) is not bool or type(self.output_limited) is not bool:
            raise ValueError("Sandbox execution flags are invalid")
        if type(self.duration_ms) is not int or self.duration_ms < 0:
            raise ValueError("Sandbox execution duration is invalid")
        _validate_json(self.actual_output)
        if (
            self.network_mode != "none"
            or self.read_only_rootfs is not True
            or self.user != "65532:65532"
            or self.cap_drop != ("ALL",)
            or self.no_new_privileges is not True
            or self.privileged is not False
            or self.pids_limit <= 0
            or self.memory_bytes <= 0
            or self.memory_swap_bytes != self.memory_bytes
            or self.nano_cpus <= 0
            or self.ipc_mode != "none"
            or self.cgroupns_mode != "private"
            or self.mount_count != 0
            or not self.tmpfs
            or self.log_driver != "none"
        ):
            raise ValueError("Sandbox execution isolation evidence is invalid")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cap_drop"] = list(self.cap_drop)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SandboxExecutionEvidence":
        data = dict(payload)
        cap_drop = data.get("cap_drop")
        if not isinstance(cap_drop, (list, tuple)):
            raise ValueError("Sandbox execution cap-drop evidence is invalid")
        data["cap_drop"] = tuple(cap_drop)
        return cls(**data)


@dataclass(frozen=True)
class ScientificExperimentEvidence:
    schema_version: int
    experiment_id: str
    evidence_sha256: str
    status: VerificationStatus
    failure_reason: str | None
    hypothesis_fingerprint: str
    hypothesis_revision: int
    target_event: str
    target_ir_kind: str
    language: str
    semantic_spec_sha256: str
    domain: str
    document_identity: str
    context_fingerprint: str
    source_sha256: str
    fixture_id: str
    fixture_version: int
    fixture_sha256: str
    stimulus_sha256: str
    oracle_id: str
    oracle_kind: str
    oracle_version: str
    oracle_sha256: str
    verifier_id: str
    verifier_version: str
    operation: str
    expected_output: Any
    actual_output: Any
    intended_passed: bool
    negative_control_fingerprint: str
    negative_control_output: Any
    negative_control_rejected: bool
    sandbox_execution: SandboxExecutionEvidence | None = None

    @property
    def passed(self) -> bool:
        return self.status is VerificationStatus.PASSED

    @property
    def independence_key(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "context_fingerprint": self.context_fingerprint,
                    "source_sha256": self.source_sha256,
                    "stimulus_sha256": self.stimulus_sha256,
                    "oracle_sha256": self.oracle_sha256,
                }
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        if self.schema_version < SANDBOX_EVIDENCE_SCHEMA_VERSION:
            payload.pop("sandbox_execution", None)
        elif self.sandbox_execution is not None:
            payload["sandbox_execution"] = self.sandbox_execution.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScientificExperimentEvidence":
        data = dict(payload)
        data["status"] = VerificationStatus(data["status"])
        schema_version = data.get("schema_version")
        if schema_version not in {SCHEMA_VERSION, SANDBOX_EVIDENCE_SCHEMA_VERSION}:
            raise ValueError("Scientific experiment evidence schema is unsupported")
        sandbox_payload = data.get("sandbox_execution")
        if sandbox_payload is None:
            data["sandbox_execution"] = None
        elif isinstance(sandbox_payload, dict):
            data["sandbox_execution"] = SandboxExecutionEvidence.from_dict(sandbox_payload)
        else:
            raise ValueError("Scientific experiment sandbox evidence is invalid")
        evidence = cls(**data)
        expected = _evidence_hash(evidence)
        if evidence.evidence_sha256 != expected:
            raise ValueError("Scientific experiment evidence integrity mismatch")
        return evidence


class BoundedProjectionVerifier:
    """Deterministic JSON-only verifier. It never executes source or generated code."""

    _PAIR_TO_OPERATION = {
        ("BRANCH", "branch"): "branch_select",
        ("LOOP", "loop"): "loop_sum",
        ("RETURN", "return"): "return_value",
        ("ASSIGN", "assignment"): "assign_value",
        ("OPERATOR", "operator"): "binary_operator",
        ("RAISE", "error"): "raise_signal",
    }

    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    def verify(
        self,
        projection: HypothesisProjection,
        fixture: CodingExperimentFixture,
        *,
        source: str,
        context_fingerprint: str,
        negative_control: HypothesisProjection,
    ) -> ScientificExperimentEvidence:
        if not isinstance(source, str) or len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
            return self._result(
                projection, fixture, source if isinstance(source, str) else "",
                context_fingerprint, negative_control,
                VerificationStatus.INVALID, "source is missing or exceeds the verifier limit",
            )
        if not isinstance(context_fingerprint, str) or not context_fingerprint.strip():
            return self._result(
                projection, fixture, source, "", negative_control,
                VerificationStatus.INVALID, "context fingerprint is required",
            )
        try:
            _validate_json(fixture.inputs)
            _validate_json(fixture.expected_output)
            fixture_bytes = _canonical_json(asdict(fixture))
            if len(fixture_bytes) > _MAX_FIXTURE_BYTES:
                raise ValueError("fixture exceeds the verifier size limit")
        except (TypeError, ValueError) as exc:
            return self._result(
                projection, fixture, source, context_fingerprint, negative_control,
                VerificationStatus.INVALID, str(exc),
            )
        if not self.available:
            return self._result(
                projection, fixture, source, context_fingerprint, negative_control,
                VerificationStatus.VERIFIER_UNAVAILABLE, "bounded verifier is unavailable",
            )
        if fixture.operation not in set(self._PAIR_TO_OPERATION.values()):
            return self._result(
                projection, fixture, source, context_fingerprint, negative_control,
                VerificationStatus.UNSUPPORTED, "fixture operation is not in the safe subset",
            )

        intended_supported, intended_output, intended_reason = self._execute(
            projection, fixture
        )
        if intended_supported:
            try:
                _validate_json(intended_output)
                if len(_canonical_json(intended_output)) > _MAX_FIXTURE_BYTES:
                    raise ValueError("bounded result exceeds the verifier size limit")
            except (TypeError, ValueError) as exc:
                intended_supported = False
                intended_output = None
                intended_reason = f"invalid bounded fixture: {exc}"
        intended_passed = (
            intended_supported and _json_equal(intended_output, fixture.expected_output)
        )
        control_supported, control_output, _ = self._execute(negative_control, fixture)
        control_rejected = not control_supported or not _json_equal(
            control_output, fixture.expected_output
        )

        if not intended_supported:
            status = (
                VerificationStatus.INVALID
                if intended_reason and intended_reason.startswith("invalid bounded fixture")
                else VerificationStatus.CANDIDATE_FAILED
            )
            reason = intended_reason
        elif not intended_passed:
            status = VerificationStatus.CANDIDATE_FAILED
            reason = "candidate projection did not produce the independent expected result"
        elif not control_rejected:
            status = VerificationStatus.CONTROL_FAILED
            reason = "wrong-projection control also produced the expected result"
        else:
            status = VerificationStatus.PASSED
            reason = None
        return self._result(
            projection,
            fixture,
            source,
            context_fingerprint,
            negative_control,
            status,
            reason,
            actual_output=intended_output,
            intended_passed=intended_passed,
            negative_control_output=control_output,
            negative_control_rejected=control_rejected,
        )

    def _execute(
        self,
        projection: HypothesisProjection,
        fixture: CodingExperimentFixture,
    ) -> tuple[bool, Any, str | None]:
        expected_operation = self._PAIR_TO_OPERATION.get(
            (projection.target_event.upper(), projection.target_ir_kind.casefold())
        )
        if expected_operation != fixture.operation:
            return False, None, "projection does not implement the fixture operation"
        inputs = fixture.inputs
        try:
            if fixture.operation == "branch_select":
                condition = inputs["condition"]
                if type(condition) is not bool:
                    raise ValueError("branch condition must be boolean")
                return True, inputs["when_true"] if condition else inputs["when_false"], None
            if fixture.operation == "loop_sum":
                values = inputs["values"]
                initial = inputs.get("initial", 0)
                if not isinstance(values, list) or not all(_is_number(item) for item in values):
                    raise ValueError("loop values must be a list of finite numbers")
                if not _is_number(initial):
                    raise ValueError("loop initial value must be a finite number")
                if len(values) > _MAX_COLLECTION_ITEMS:
                    raise ValueError("loop exceeds the verifier step limit")
                return True, sum(values, initial), None
            if fixture.operation == "return_value":
                return True, inputs["value"], None
            if fixture.operation == "assign_value":
                state = inputs["state"]
                name = inputs["name"]
                if not isinstance(state, dict) or not isinstance(name, str) or not name:
                    raise ValueError("assignment requires a JSON object and non-empty name")
                result = dict(state)
                result[name] = inputs["value"]
                return True, result, None
            if fixture.operation == "binary_operator":
                op = inputs["operator"]
                allowed = {
                    "+": operator.add,
                    "-": operator.sub,
                    "*": operator.mul,
                    "==": operator.eq,
                    "<": operator.lt,
                    ">": operator.gt,
                }
                if op not in allowed:
                    raise ValueError("binary operator is outside the safe allowlist")
                left = inputs["left"]
                right = inputs["right"]
                if op in {"+", "-", "*"} and not (
                    _is_number(left) and _is_number(right)
                ):
                    raise ValueError("arithmetic operators require finite numeric operands")
                return True, allowed[op](left, right), None
            if fixture.operation == "raise_signal":
                category = inputs["category"]
                if not isinstance(category, str) or not category:
                    raise ValueError("raise category must be a non-empty string")
                result = {"raised": category}
                if "message" in inputs:
                    message = inputs["message"]
                    if not isinstance(message, str):
                        raise ValueError("raise message must be a string")
                    result["message"] = message
                return True, result, None
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return False, None, f"invalid bounded fixture: {exc}"
        return False, None, "fixture operation is not in the safe subset"

    def _result(
        self,
        projection: HypothesisProjection,
        fixture: CodingExperimentFixture,
        source: str,
        context_fingerprint: str,
        negative_control: HypothesisProjection,
        status: VerificationStatus,
        failure_reason: str | None,
        *,
        actual_output: Any = None,
        intended_passed: bool = False,
        negative_control_output: Any = None,
        negative_control_rejected: bool = False,
        sandbox_execution: SandboxExecutionEvidence | None = None,
        schema_version: int = SCHEMA_VERSION,
        verifier_id: str = VERIFIER_ID,
        verifier_version: str = VERIFIER_VERSION,
    ) -> ScientificExperimentEvidence:
        provisional = ScientificExperimentEvidence(
            schema_version=schema_version,
            experiment_id=fixture.experiment_id,
            evidence_sha256="",
            status=status,
            failure_reason=failure_reason,
            hypothesis_fingerprint=projection.fingerprint,
            hypothesis_revision=projection.revision,
            target_event=projection.target_event,
            target_ir_kind=projection.target_ir_kind,
            language=projection.language,
            semantic_spec_sha256=projection.semantic_spec_sha256,
            domain=fixture.domain,
            document_identity=fixture.document_identity,
            context_fingerprint=context_fingerprint,
            source_sha256=_sha256(source.encode("utf-8")),
            fixture_id=fixture.fixture_id,
            fixture_version=fixture.fixture_version,
            fixture_sha256=fixture.fingerprint,
            stimulus_sha256=fixture.stimulus_fingerprint,
            oracle_id=fixture.oracle_id,
            oracle_kind=fixture.oracle_kind,
            oracle_version=fixture.oracle_version,
            oracle_sha256=fixture.oracle_fingerprint,
            verifier_id=verifier_id,
            verifier_version=verifier_version,
            operation=fixture.operation,
            expected_output=fixture.expected_output,
            actual_output=actual_output,
            intended_passed=intended_passed,
            negative_control_fingerprint=negative_control.fingerprint,
            negative_control_output=negative_control_output,
            negative_control_rejected=negative_control_rejected,
            sandbox_execution=sandbox_execution,
        )
        return ScientificExperimentEvidence(
            **{**provisional.__dict__, "evidence_sha256": _evidence_hash(provisional)}
        )


class SandboxBackedProjectionVerifier:
    """Require real isolated source execution plus projection/control checks."""

    def __init__(self, sandbox_executor: Any) -> None:
        if not callable(getattr(sandbox_executor, "execute", None)):
            raise TypeError("sandbox executor must provide execute")
        self._sandbox = sandbox_executor
        self._projection = BoundedProjectionVerifier()

    def verify(
        self,
        projection: HypothesisProjection,
        fixture: CodingExperimentFixture,
        *,
        source: str,
        context_fingerprint: str,
        negative_control: HypothesisProjection,
    ) -> ScientificExperimentEvidence:
        bounded = self._projection.verify(
            projection,
            fixture,
            source=source,
            context_fingerprint=context_fingerprint,
            negative_control=negative_control,
        )
        if not bounded.passed:
            return _rehash_evidence(
                replace(
                    bounded,
                    schema_version=SANDBOX_EVIDENCE_SCHEMA_VERSION,
                    evidence_sha256="",
                    verifier_id=SANDBOX_VERIFIER_ID,
                    verifier_version=SANDBOX_VERIFIER_VERSION,
                )
            )
        try:
            result = self._sandbox.execute(
                language=projection.language,
                source=source,
                stdin_text=_canonical_json(fixture.inputs).decode("utf-8"),
            )
        except Exception as error:
            return _rehash_evidence(
                replace(
                    bounded,
                    schema_version=SANDBOX_EVIDENCE_SCHEMA_VERSION,
                    evidence_sha256="",
                    status=VerificationStatus.VERIFIER_UNAVAILABLE,
                    failure_reason=f"sandbox execution unavailable: {type(error).__name__}",
                    verifier_id=SANDBOX_VERIFIER_ID,
                    verifier_version=SANDBOX_VERIFIER_VERSION,
                )
            )
        actual_output: Any = None
        output_valid = False
        if result.status == "passed" and result.exit_code == 0 and not result.stderr_bytes:
            try:
                actual_output = json.loads(result.stdout_bytes)
                _validate_json(actual_output)
                output_valid = True
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                output_valid = False
        execution = _sandbox_execution_evidence(result, actual_output)
        sandbox_passed = output_valid and _json_equal(actual_output, fixture.expected_output)
        return _rehash_evidence(
            replace(
                bounded,
                schema_version=SANDBOX_EVIDENCE_SCHEMA_VERSION,
                evidence_sha256="",
                status=(
                    VerificationStatus.PASSED
                    if sandbox_passed
                    else VerificationStatus.CANDIDATE_FAILED
                ),
                failure_reason=(
                    None
                    if sandbox_passed
                    else "sandboxed source did not produce the independent expected result"
                ),
                verifier_id=SANDBOX_VERIFIER_ID,
                verifier_version=SANDBOX_VERIFIER_VERSION,
                sandbox_execution=execution,
            )
        )


def _sandbox_execution_evidence(
    result: Any,
    actual_output: Any,
) -> SandboxExecutionEvidence:
    attestation = result.attestation
    return SandboxExecutionEvidence(
        schema_version=1,
        protocol=result.protocol,
        request_sha256=result.request_sha256,
        response_sha256=result.response_sha256,
        image_id=attestation.image_id,
        image_manifest_sha256=attestation.image_manifest_sha256,
        policy_sha256=attestation.policy_sha256,
        container_id=attestation.container_id,
        language=result.language,
        status=result.status,
        phase=result.phase,
        exit_code=result.exit_code,
        stdout_sha256=_sha256(result.stdout_bytes),
        stderr_sha256=_sha256(result.stderr_bytes),
        timed_out=result.timed_out,
        output_limited=result.output_limited,
        duration_ms=result.duration_ms,
        actual_output=actual_output,
        network_mode=attestation.network_mode,
        read_only_rootfs=attestation.read_only_rootfs,
        user=attestation.user,
        cap_drop=attestation.cap_drop,
        no_new_privileges=attestation.no_new_privileges,
        privileged=attestation.privileged,
        pids_limit=attestation.pids_limit,
        memory_bytes=attestation.memory_bytes,
        memory_swap_bytes=attestation.memory_swap_bytes,
        nano_cpus=attestation.nano_cpus,
        ipc_mode=attestation.ipc_mode,
        cgroupns_mode=attestation.cgroupns_mode,
        mount_count=attestation.mount_count,
        tmpfs=attestation.tmpfs,
        log_driver=attestation.log_driver,
    )


def _rehash_evidence(evidence: ScientificExperimentEvidence) -> ScientificExperimentEvidence:
    return replace(evidence, evidence_sha256=_evidence_hash(evidence))


def load_experiment_fixture(path: str | Path) -> CodingExperimentFixture:
    source = Path(path)
    payload_bytes = source.read_bytes()
    if len(payload_bytes) > _MAX_FIXTURE_BYTES:
        raise ValueError("Experiment fixture file exceeds the verifier size limit")
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Experiment fixture is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Experiment fixture must be a JSON object")
    try:
        fixture = CodingExperimentFixture(**payload)
    except TypeError as exc:
        raise ValueError(f"Experiment fixture schema is invalid: {exc}") from exc
    _validate_json(fixture.inputs)
    _validate_json(fixture.expected_output)
    return fixture


def _evidence_hash(evidence: ScientificExperimentEvidence) -> str:
    payload = evidence.to_dict()
    payload["evidence_sha256"] = ""
    return _sha256(_canonical_json(payload))


def _validate_json(value: Any, *, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise ValueError("fixture exceeds the verifier depth limit")
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str) and len(value.encode("utf-8")) > _MAX_STRING_BYTES:
            raise ValueError("fixture string exceeds the verifier limit")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fixture numbers must be finite")
        return
    if isinstance(value, list):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise ValueError("fixture collection exceeds the verifier limit")
        for item in value:
            _validate_json(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_COLLECTION_ITEMS or any(not isinstance(key, str) for key in value):
            raise ValueError("fixture objects require bounded string keys")
        for key, item in value.items():
            _validate_json(key, depth=depth + 1)
            _validate_json(item, depth=depth + 1)
        return
    raise TypeError("fixture must contain only finite JSON literals")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _json_equal(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and (not isinstance(value, float) or math.isfinite(value))
    )

