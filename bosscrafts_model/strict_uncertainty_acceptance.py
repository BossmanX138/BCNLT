"""BossCrafts-level selective-risk gate for contained BCNLT calibration."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
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


_POLICY_FIELDS = {
    "minimum_global_selective_coverage_at_025",
    "maximum_global_selective_error_rate_at_025",
    "minimum_per_language_selective_coverage_at_025",
    "maximum_per_language_selective_error_rate_at_025",
}
_METRIC_FIELDS = {
    "sample_count",
    "brier",
    "ece",
    "selective_coverage_at_025",
    "selective_error_rate_at_025",
}
_RECEIPT_FIELDS = {
    "calibration_artifact_sha256",
    "failure_codes",
    "global_metrics",
    "parser_runtime_identity_sha256",
    "passed",
    "per_language_metrics",
    "policy",
    "receipt_sha256",
    "schema_version",
    "upstream_promoted",
}


def _validated_policy_payload(value: dict[object, object]) -> dict[str, float]:
    if set(value) != _POLICY_FIELDS:
        raise ValueError("strict uncertainty policy fields are invalid")
    policy: dict[str, float] = {}
    for key, threshold in value.items():
        if (
            type(key) is not str
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("strict uncertainty policy is invalid")
        policy[key] = float(threshold)
    return policy


def _validated_metrics_payload(
    value: dict[object, object],
) -> dict[str, float | int | None]:
    if set(value) != _METRIC_FIELDS:
        raise ValueError("strict uncertainty metric fields are invalid")
    sample_count = value.get("sample_count")
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError("strict uncertainty sample count is invalid")
    metrics: dict[str, float | int | None] = {"sample_count": sample_count}
    for field in _METRIC_FIELDS - {"sample_count"}:
        metric = value.get(field)
        if metric is None:
            metrics[field] = None
        elif (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            or not 0.0 <= float(metric) <= 1.0
        ):
            raise ValueError("strict uncertainty metric is invalid")
        else:
            metrics[field] = float(metric)
    return metrics


def _receipt_thresholds_pass(
    policy: dict[str, float],
    global_metrics: dict[str, float | int | None],
    per_language: dict[str, dict[str, float | int | None]],
) -> bool:
    global_coverage = global_metrics["selective_coverage_at_025"]
    global_error = global_metrics["selective_error_rate_at_025"]
    if (
        global_coverage is None
        or float(global_coverage)
        < policy["minimum_global_selective_coverage_at_025"]
        or (
            global_error is None
            and policy["minimum_global_selective_coverage_at_025"] > 0.0
        )
        or (
            global_error is not None
            and float(global_error)
            > policy["maximum_global_selective_error_rate_at_025"]
        )
    ):
        return False
    for metrics in per_language.values():
        coverage = metrics["selective_coverage_at_025"]
        error = metrics["selective_error_rate_at_025"]
        if (
            coverage is None
            or float(coverage)
            < policy["minimum_per_language_selective_coverage_at_025"]
            or (
                error is None
                and policy["minimum_per_language_selective_coverage_at_025"] > 0.0
            )
            or (
                error is not None
                and float(error)
                > policy["maximum_per_language_selective_error_rate_at_025"]
            )
        ):
            return False
    return True


def validate_strict_uncertainty_receipt_payload(
    payload: dict[str, object],
    *,
    expected_parser_runtime_identity_sha256: str,
    expected_languages: Sequence[str],
) -> StrictUncertaintyAcceptanceReceipt:
    """Revalidate a serialized receipt rather than trusting its pass flag."""

    if set(payload) != _RECEIPT_FIELDS:
        raise ValueError("strict uncertainty receipt fields are invalid")
    if not _is_digest(expected_parser_runtime_identity_sha256):
        raise ValueError("expected parser runtime identity is invalid")
    languages = tuple(expected_languages)
    if (
        not languages
        or len(set(languages)) != len(languages)
        or any(type(language) is not str or not language for language in languages)
    ):
        raise ValueError("expected strict uncertainty languages are invalid")
    raw_policy = payload.get("policy")
    raw_global_metrics = payload.get("global_metrics")
    raw_per_language_metrics = payload.get("per_language_metrics")
    failure_codes = payload.get("failure_codes")
    if (
        type(raw_policy) is not dict
        or type(raw_global_metrics) is not dict
        or type(raw_per_language_metrics) is not dict
        or type(failure_codes) is not list
        or any(type(code) is not str or not code for code in failure_codes)
    ):
        raise ValueError("strict uncertainty receipt payload is invalid")
    policy = _validated_policy_payload(raw_policy)
    global_metrics = _validated_metrics_payload(raw_global_metrics)
    if set(raw_per_language_metrics) != set(languages):
        raise ValueError("strict uncertainty receipt language coverage is incomplete")
    per_language_metrics: dict[str, dict[str, float | int | None]] = {}
    for language, raw_metrics in raw_per_language_metrics.items():
        if type(language) is not str or type(raw_metrics) is not dict:
            raise ValueError("strict uncertainty language metrics are invalid")
        per_language_metrics[language] = _validated_metrics_payload(raw_metrics)
    if int(global_metrics["sample_count"] or 0) != sum(
        int(metrics["sample_count"] or 0)
        for metrics in per_language_metrics.values()
    ):
        raise ValueError("strict uncertainty sample inventory is inconsistent")
    receipt = StrictUncertaintyAcceptanceReceipt(
        schema_version=str(payload.get("schema_version")),
        calibration_artifact_sha256=str(
            payload.get("calibration_artifact_sha256")
        ),
        parser_runtime_identity_sha256=str(
            payload.get("parser_runtime_identity_sha256")
        ),
        policy=policy,
        global_metrics=global_metrics,
        per_language_metrics=per_language_metrics,
        upstream_promoted=payload.get("upstream_promoted") is True,
        passed=payload.get("passed") is True,
        failure_codes=tuple(failure_codes),
        receipt_sha256=str(payload.get("receipt_sha256")),
    )
    try:
        receipt.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("strict uncertainty receipt integrity mismatch") from error
    if (
        receipt.passed is not True
        or receipt.upstream_promoted is not True
        or receipt.failure_codes
        or receipt.parser_runtime_identity_sha256
        != expected_parser_runtime_identity_sha256
        or not _is_digest(receipt.calibration_artifact_sha256)
        or not _receipt_thresholds_pass(
            receipt.policy,
            receipt.global_metrics,
            receipt.per_language_metrics,
        )
    ):
        raise ValueError("strict uncertainty receipt did not pass current runtime gates")
    return receipt


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
