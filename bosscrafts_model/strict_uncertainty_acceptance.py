"""BossCrafts-level selective-risk gate for contained BCNLT calibration."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

_SCHEMA_VERSION = "bosscrafts-strict-uncertainty-acceptance-v1"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True, slots=True)
class StrictUncertaintyPolicy:
    minimum_global_selective_coverage_at_025: float
    maximum_global_selective_error_rate_at_025: float
    minimum_per_language_selective_coverage_at_025: float
    maximum_per_language_selective_error_rate_at_025: float

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in asdict(self).values()
        ):
            raise ValueError("strict uncertainty thresholds must be finite and bounded")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class StrictUncertaintyAcceptanceReceipt:
    schema_version: str
    calibration_artifact_sha256: str
    parser_runtime_identity_sha256: str
    policy: dict[str, float]
    global_metrics: dict[str, float | int | None]
    per_language_metrics: dict[str, dict[str, float | int | None]]
    upstream_promoted: bool
    passed: bool
    failure_codes: tuple[str, ...]
    receipt_sha256: str

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "parser_runtime_identity_sha256": self.parser_runtime_identity_sha256,
            "policy": self.policy,
            "global_metrics": self.global_metrics,
            "per_language_metrics": self.per_language_metrics,
            "upstream_promoted": self.upstream_promoted,
            "passed": self.passed,
            "failure_codes": list(self.failure_codes),
        }

    def compute_sha256(self) -> str:
        return _digest(self._body())

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "receipt_sha256": self.receipt_sha256}

    def validate(self) -> None:
        if (
            self.schema_version != _SCHEMA_VERSION
            or not _is_digest(self.calibration_artifact_sha256)
            or not _is_digest(self.parser_runtime_identity_sha256)
            or self.passed != (self.upstream_promoted and not self.failure_codes)
            or self.receipt_sha256 != self.compute_sha256()
        ):
            raise ValueError("strict uncertainty acceptance receipt integrity mismatch")


def _metrics_dict(metrics: Any) -> dict[str, float | int | None]:
    values = {
        "sample_count": metrics.sample_count,
        "brier": metrics.brier,
        "ece": metrics.ece,
        "selective_coverage_at_025": metrics.selective_coverage_at_025,
        "selective_error_rate_at_025": metrics.selective_error_rate_at_025,
    }
    if (
        type(values["sample_count"]) is not int
        or values["sample_count"] < 1
        or any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            )
            for key, value in values.items()
            if key != "sample_count"
        )
    ):
        raise ValueError("calibration metrics are invalid")
    return values


def verify_strict_uncertainty_acceptance(
    artifact: Any,
    policy: StrictUncertaintyPolicy,
) -> StrictUncertaintyAcceptanceReceipt:
    if type(policy) is not StrictUncertaintyPolicy:
        raise TypeError("policy must be StrictUncertaintyPolicy")
    try:
        artifact.validate()
    except AttributeError as exc:
        raise TypeError("artifact must be a validated uncertainty calibration artifact") from exc
    if not _is_digest(getattr(artifact, "artifact_sha256", None)):
        raise ValueError("calibration artifact identity is invalid")
    if not _is_digest(getattr(artifact, "parser_runtime_identity_sha256", None)):
        raise ValueError("calibration parser identity is invalid")

    global_metrics = _metrics_dict(artifact.calibrated_metrics)
    per_language_metrics = {
        str(language): _metrics_dict(metrics)
        for language, metrics in sorted(artifact.per_language.items())
    }
    failures: list[str] = []
    upstream_promoted = artifact.promoted is True
    if not upstream_promoted:
        failures.append("upstream_calibration_not_promoted")

    global_coverage = global_metrics["selective_coverage_at_025"]
    global_error = global_metrics["selective_error_rate_at_025"]
    if global_coverage is None:
        failures.append("global_selective_coverage_unavailable")
    elif float(global_coverage) < policy.minimum_global_selective_coverage_at_025:
        failures.append("global_selective_coverage_below_threshold")
    if global_error is None:
        if policy.minimum_global_selective_coverage_at_025 > 0:
            failures.append("global_selective_error_rate_unavailable")
    elif float(global_error) > policy.maximum_global_selective_error_rate_at_025:
        failures.append("global_selective_error_rate_above_threshold")

    for language, metrics in per_language_metrics.items():
        coverage = metrics["selective_coverage_at_025"]
        error = metrics["selective_error_rate_at_025"]
        if coverage is None:
            failures.append(f"{language}_selective_coverage_unavailable")
        elif float(coverage) < policy.minimum_per_language_selective_coverage_at_025:
            failures.append(f"{language}_selective_coverage_below_threshold")
        if error is None:
            if policy.minimum_per_language_selective_coverage_at_025 > 0:
                failures.append(f"{language}_selective_error_rate_unavailable")
        elif float(error) > policy.maximum_per_language_selective_error_rate_at_025:
            failures.append(f"{language}_selective_error_rate_above_threshold")

    provisional = StrictUncertaintyAcceptanceReceipt(
        schema_version=_SCHEMA_VERSION,
        calibration_artifact_sha256=artifact.artifact_sha256,
        parser_runtime_identity_sha256=artifact.parser_runtime_identity_sha256,
        policy=policy.to_dict(),
        global_metrics=global_metrics,
        per_language_metrics=per_language_metrics,
        upstream_promoted=upstream_promoted,
        passed=upstream_promoted and not failures,
        failure_codes=tuple(failures),
        receipt_sha256="",
    )
    receipt = replace(provisional, receipt_sha256=provisional.compute_sha256())
    receipt.validate()
    return receipt
