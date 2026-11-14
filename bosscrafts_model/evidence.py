"""Host-side typed evidence gate, not a secure sandbox or neural verifier.

The trusted host must own this object. Python code running in the same process
can bypass it; generated code is never executed here.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    task_id: str
    artifact_sha256: str
    check_id: str
    passed: bool
    output_sha256: str


class EvidenceLedger:
    def __init__(self) -> None:
        self._observations: list[Observation] = []

    def record_tool_result(
        self,
        *,
        task_id: str,
        artifact_sha256: str,
        check_id: str,
        passed: bool,
        output: str,
    ) -> Observation:
        if type(passed) is not bool:
            raise ValueError("passed must be an actual boolean")
        if not task_id or not check_id:
            raise ValueError("task and check IDs required")
        if len(artifact_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in artifact_sha256
        ):
            raise ValueError("artifact SHA-256 required")
        row = Observation(
            task_id,
            artifact_sha256,
            check_id,
            passed,
            hashlib.sha256(output.encode()).hexdigest(),
        )
        self._observations.append(row)
        return row

    def snapshot(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    def can_mark_verified(
        self,
        task_id: str,
        artifact_sha256: str,
        required_checks: Sequence[str],
    ) -> bool:
        if (
            not isinstance(required_checks, (list, tuple))
            or not required_checks
            or any(
                type(check) is not str or not check.strip() for check in required_checks
            )
        ):
            return False
        latest = {
            observation.check_id: observation.passed
            for observation in self._observations
            if observation.task_id == task_id
            and observation.artifact_sha256 == artifact_sha256
        }
        return all(latest.get(check) is True for check in required_checks)
