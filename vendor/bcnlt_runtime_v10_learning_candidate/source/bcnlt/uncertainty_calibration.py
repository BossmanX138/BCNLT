from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from .cache import ProjectIndex
from .core import ContextualUnit
from .uncertainty import UncertaintyScore


CALIBRATOR_SCHEMA_VERSION = "bcnlt-uncertainty-calibrator-v1"
ARTIFACT_SCHEMA_VERSION = "bcnlt-uncertainty-calibration-artifact-v1"


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True, slots=True)
class UncertaintyObservation:
    sample_id: str
    language: str
    raw_uncertainty: float
    incorrect: bool

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.language.strip():
            raise ValueError("Uncertainty observations require sample and language identities")
        if (
            isinstance(self.raw_uncertainty, bool)
            or not isinstance(self.raw_uncertainty, (int, float))
            or not math.isfinite(float(self.raw_uncertainty))
            or not 0.0 <= float(self.raw_uncertainty) <= 1.0
        ):
            raise ValueError("Uncertainty observation score must be finite and bounded")
        if type(self.incorrect) is not bool:
            raise ValueError("Uncertainty observation label must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "language": self.language,
            "raw_uncertainty": float(self.raw_uncertainty),
            "incorrect": self.incorrect,
        }


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    upper_bound: float
    calibrated_uncertainty: float
    sample_count: int

    def __post_init__(self) -> None:
        if (
            not 0.0 <= self.upper_bound <= 1.0
            or not 0.0 <= self.calibrated_uncertainty <= 1.0
            or self.sample_count < 1
        ):
            raise ValueError("Uncertainty calibration point is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UncertaintyCalibrator:
    schema_version: str
    parser_runtime_identity_sha256: str
    calibration_data_sha256: str
    calibration_sample_ids: tuple[str, ...]
    points: tuple[CalibrationPoint, ...]
    calibrator_sha256: str

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parser_runtime_identity_sha256": self.parser_runtime_identity_sha256,
            "calibration_data_sha256": self.calibration_data_sha256,
            "calibration_sample_ids_sha256": _digest(list(self.calibration_sample_ids)),
            "points": [point.to_dict() for point in self.points],
        }

    def validate(self) -> None:
        if (
            self.schema_version != CALIBRATOR_SCHEMA_VERSION
            or not _is_digest(self.parser_runtime_identity_sha256)
            or not _is_digest(self.calibration_data_sha256)
            or not self.calibration_sample_ids
            or len(self.calibration_sample_ids) != len(set(self.calibration_sample_ids))
            or not self.points
            or tuple(sorted(self.points, key=lambda point: point.upper_bound)) != self.points
            or any(
                left.calibrated_uncertainty > right.calibrated_uncertainty
                for left, right in zip(self.points, self.points[1:])
            )
            or self.calibrator_sha256 != _digest(self._body())
        ):
            raise ValueError("Uncertainty calibrator integrity mismatch")

    def apply(self, raw_uncertainty: float) -> float:
        self.validate()
        value = float(raw_uncertainty)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Uncertainty score must be finite and bounded")
        for point in self.points:
            if value <= point.upper_bound:
                return point.calibrated_uncertainty
        return self.points[-1].calibrated_uncertainty


@dataclass(frozen=True, slots=True)
class CalibrationThresholds:
    minimum_heldout_samples: int
    minimum_languages: int
    minimum_samples_per_language: int
    maximum_calibrated_ece: float
    maximum_calibrated_brier: float
    maximum_brier_degradation: float

    def __post_init__(self) -> None:
        if (
            self.minimum_heldout_samples < 1
            or self.minimum_languages < 1
            or self.minimum_samples_per_language < 1
            or not 0.0 <= self.maximum_calibrated_ece <= 1.0
            or not 0.0 <= self.maximum_calibrated_brier <= 1.0
            or not 0.0 <= self.maximum_brier_degradation <= 1.0
        ):
            raise ValueError("Uncertainty calibration thresholds are invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    sample_count: int
    brier: float
    ece: float
    selective_coverage_at_025: float
    selective_error_rate_at_025: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UncertaintyCalibrationArtifact:
    schema_version: str
    parser_runtime_identity_sha256: str
    calibrator_sha256: str
    calibration_data_sha256: str
    calibration_sample_ids_sha256: str
    heldout_data_sha256: str
    heldout_manifest_sha256: str
    points: tuple[CalibrationPoint, ...]
    raw_metrics: CalibrationMetrics
    calibrated_metrics: CalibrationMetrics
    per_language: dict[str, CalibrationMetrics]
    thresholds: CalibrationThresholds
    promoted: bool
    failure_codes: tuple[str, ...]
    artifact_sha256: str

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parser_runtime_identity_sha256": self.parser_runtime_identity_sha256,
            "calibrator_sha256": self.calibrator_sha256,
            "calibration_data_sha256": self.calibration_data_sha256,
            "calibration_sample_ids_sha256": self.calibration_sample_ids_sha256,
            "heldout_data_sha256": self.heldout_data_sha256,
            "heldout_manifest_sha256": self.heldout_manifest_sha256,
            "points": [point.to_dict() for point in self.points],
            "raw_metrics": self.raw_metrics.to_dict(),
            "calibrated_metrics": self.calibrated_metrics.to_dict(),
            "per_language": {
                language: metrics.to_dict()
                for language, metrics in sorted(self.per_language.items())
            },
            "thresholds": self.thresholds.to_dict(),
            "promoted": self.promoted,
            "failure_codes": list(self.failure_codes),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "artifact_sha256": self.artifact_sha256}

    def validate(self) -> None:
        if (
            self.schema_version != ARTIFACT_SCHEMA_VERSION
            or not all(
                _is_digest(value)
                for value in (
                    self.parser_runtime_identity_sha256,
                    self.calibrator_sha256,
                    self.calibration_data_sha256,
                    self.calibration_sample_ids_sha256,
                    self.heldout_data_sha256,
                    self.heldout_manifest_sha256,
                )
            )
            or not self.points
            or self.promoted != (not self.failure_codes)
            or self.artifact_sha256 != _digest(self._body())
        ):
            raise ValueError("Uncertainty calibration artifact integrity mismatch")


def fit_uncertainty_calibrator(
    observations: tuple[UncertaintyObservation, ...],
    *,
    parser_runtime_identity_sha256: str,
) -> UncertaintyCalibrator:
    rows = _validate_observations(observations, name="calibration")
    if not _is_digest(parser_runtime_identity_sha256):
        raise ValueError("Parser runtime identity must be a SHA-256 digest")
    grouped: list[list[float]] = []
    for score in sorted({float(row.raw_uncertainty) for row in rows}):
        labels = [float(row.incorrect) for row in rows if row.raw_uncertainty == score]
        grouped.append([score, score, float(len(labels)), sum(labels)])
    blocks: list[list[float]] = []
    for group in grouped:
        blocks.append(group)
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left[3] / left[2] <= right[3] / right[2]:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    points = tuple(
        CalibrationPoint(
            upper_bound=block[1],
            calibrated_uncertainty=block[3] / block[2],
            sample_count=int(block[2]),
        )
        for block in blocks
    )
    data_sha256 = _observation_digest(rows)
    sample_ids = tuple(sorted(row.sample_id for row in rows))
    provisional = UncertaintyCalibrator(
        schema_version=CALIBRATOR_SCHEMA_VERSION,
        parser_runtime_identity_sha256=parser_runtime_identity_sha256,
        calibration_data_sha256=data_sha256,
        calibration_sample_ids=sample_ids,
        points=points,
        calibrator_sha256="",
    )
    calibrator = replace(
        provisional,
        calibrator_sha256=_digest(provisional._body()),
    )
    calibrator.validate()
    return calibrator


def evaluate_uncertainty_calibration(
    calibrator: UncertaintyCalibrator,
    heldout: tuple[UncertaintyObservation, ...],
    *,
    thresholds: CalibrationThresholds,
    heldout_manifest_sha256: str,
) -> UncertaintyCalibrationArtifact:
    calibrator.validate()
    rows = _validate_observations(heldout, name="held-out")
    if not _is_digest(heldout_manifest_sha256):
        raise ValueError("Held-out manifest identity must be a SHA-256 digest")
    overlap = set(calibrator.calibration_sample_ids).intersection(
        row.sample_id for row in rows
    )
    if overlap:
        raise ValueError("Calibration and held-out sample identities overlap")
    raw_predictions = tuple(float(row.raw_uncertainty) for row in rows)
    calibrated_predictions = tuple(
        calibrator.apply(float(row.raw_uncertainty)) for row in rows
    )
    labels = tuple(float(row.incorrect) for row in rows)
    raw_metrics = _metrics(raw_predictions, labels)
    calibrated_metrics = _metrics(calibrated_predictions, labels)
    per_language: dict[str, CalibrationMetrics] = {}
    for language in sorted({row.language for row in rows}):
        positions = [index for index, row in enumerate(rows) if row.language == language]
        per_language[language] = _metrics(
            tuple(calibrated_predictions[index] for index in positions),
            tuple(labels[index] for index in positions),
        )
    failures: list[str] = []
    if len(rows) < thresholds.minimum_heldout_samples:
        failures.append("insufficient_heldout_samples")
    if len(per_language) < thresholds.minimum_languages:
        failures.append("insufficient_language_coverage")
    if any(
        metrics.sample_count < thresholds.minimum_samples_per_language
        for metrics in per_language.values()
    ):
        failures.append("insufficient_per_language_samples")
    if calibrated_metrics.ece > thresholds.maximum_calibrated_ece:
        failures.append("calibrated_ece_above_threshold")
    if calibrated_metrics.brier > thresholds.maximum_calibrated_brier:
        failures.append("calibrated_brier_above_threshold")
    if (
        calibrated_metrics.brier - raw_metrics.brier
        > thresholds.maximum_brier_degradation + 1e-15
    ):
        failures.append("calibrated_brier_degraded")
    provisional = UncertaintyCalibrationArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        parser_runtime_identity_sha256=calibrator.parser_runtime_identity_sha256,
        calibrator_sha256=calibrator.calibrator_sha256,
        calibration_data_sha256=calibrator.calibration_data_sha256,
        calibration_sample_ids_sha256=_digest(list(calibrator.calibration_sample_ids)),
        heldout_data_sha256=_observation_digest(rows),
        heldout_manifest_sha256=heldout_manifest_sha256,
        points=calibrator.points,
        raw_metrics=raw_metrics,
        calibrated_metrics=calibrated_metrics,
        per_language=per_language,
        thresholds=thresholds,
        promoted=not failures,
        failure_codes=tuple(failures),
        artifact_sha256="",
    )
    artifact = replace(
        provisional,
        artifact_sha256=_digest(provisional._body()),
    )
    artifact.validate()
    return artifact


class _Scorer(Protocol):
    def score(self, unit: ContextualUnit, index: ProjectIndex) -> UncertaintyScore:
        raise NotImplementedError


class CalibratedUncertaintyScorer:
    def __init__(
        self,
        base: _Scorer,
        artifact: UncertaintyCalibrationArtifact,
    ) -> None:
        artifact.validate()
        if not artifact.promoted:
            raise ValueError("Uncertainty calibration artifact is not promoted")
        self.base = base
        self.artifact = artifact

    def score(self, unit: ContextualUnit, index: ProjectIndex) -> UncertaintyScore:
        raw = self.base.score(unit, index)
        uncertainty = _apply_points(self.artifact.points, raw.uncertainty)
        return UncertaintyScore(
            uncertainty=uncertainty,
            confidence=1.0 - uncertainty,
            signals=(*raw.signals, "held-out calibrated"),
        )


def write_calibration_artifact(
    path: str | Path,
    artifact: UncertaintyCalibrationArtifact,
) -> None:
    artifact.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(artifact.to_dict())
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_calibration_artifact(path: str | Path) -> UncertaintyCalibrationArtifact:
    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load uncertainty calibration artifact: {exc}") from exc
    if not isinstance(payload, dict) or raw != _canonical(payload):
        raise ValueError("Uncertainty calibration artifact is not canonical")
    try:
        artifact = UncertaintyCalibrationArtifact(
            schema_version=str(payload["schema_version"]),
            parser_runtime_identity_sha256=str(payload["parser_runtime_identity_sha256"]),
            calibrator_sha256=str(payload["calibrator_sha256"]),
            calibration_data_sha256=str(payload["calibration_data_sha256"]),
            calibration_sample_ids_sha256=str(payload["calibration_sample_ids_sha256"]),
            heldout_data_sha256=str(payload["heldout_data_sha256"]),
            heldout_manifest_sha256=str(payload["heldout_manifest_sha256"]),
            points=tuple(CalibrationPoint(**item) for item in payload["points"]),
            raw_metrics=CalibrationMetrics(**payload["raw_metrics"]),
            calibrated_metrics=CalibrationMetrics(**payload["calibrated_metrics"]),
            per_language={
                str(language): CalibrationMetrics(**metrics)
                for language, metrics in payload["per_language"].items()
            },
            thresholds=CalibrationThresholds(**payload["thresholds"]),
            promoted=payload["promoted"],
            failure_codes=tuple(str(item) for item in payload["failure_codes"]),
            artifact_sha256=str(payload["artifact_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid uncertainty calibration artifact: {exc}") from exc
    artifact.validate()
    return artifact


def _validate_observations(
    observations: tuple[UncertaintyObservation, ...],
    *,
    name: str,
) -> tuple[UncertaintyObservation, ...]:
    if not observations:
        raise ValueError(f"The {name} uncertainty split is empty")
    rows = tuple(sorted(observations, key=lambda row: row.sample_id))
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError(f"The {name} uncertainty split has duplicate sample identities")
    return rows


def _observation_digest(rows: tuple[UncertaintyObservation, ...]) -> str:
    return _digest([row.to_dict() for row in rows])


def _apply_points(points: tuple[CalibrationPoint, ...], raw: float) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Uncertainty score must be finite and bounded")
    for point in points:
        if value <= point.upper_bound:
            return point.calibrated_uncertainty
    return points[-1].calibrated_uncertainty


def _metrics(
    predictions: tuple[float, ...],
    labels: tuple[float, ...],
) -> CalibrationMetrics:
    if not predictions or len(predictions) != len(labels):
        raise ValueError("Calibration metrics require aligned non-empty observations")
    count = len(predictions)
    brier = sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels)) / count
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10.0
        upper = (bin_index + 1) / 10.0
        positions = [
            index
            for index, prediction in enumerate(predictions)
            if (lower <= prediction < upper) or (bin_index == 9 and prediction == 1.0)
        ]
        if positions:
            mean_prediction = sum(predictions[index] for index in positions) / len(positions)
            error_rate = sum(labels[index] for index in positions) / len(positions)
            ece += len(positions) / count * abs(mean_prediction - error_rate)
    accepted = [index for index, prediction in enumerate(predictions) if prediction <= 0.25]
    coverage = len(accepted) / count
    selective_error = (
        None if not accepted else sum(labels[index] for index in accepted) / len(accepted)
    )
    return CalibrationMetrics(
        sample_count=count,
        brier=brier,
        ece=ece,
        selective_coverage_at_025=coverage,
        selective_error_rate_at_025=selective_error,
    )
