"""Promote reviewed semantic labels through contained and strict calibration gates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Protocol

from .strict_uncertainty_acceptance import (
    StrictUncertaintyPolicy,
    verify_strict_uncertainty_acceptance,
)
from .runtime_identity import aggregate_sha256


_DATASET_SCHEMA = "bosscrafts-semantic-calibration-dataset-v1"
_OBSERVATION_SCHEMA = "bosscrafts-semantic-uncertainty-observation-v1"
_PROMOTION_SCHEMA = "bosscrafts-semantic-calibration-promotion-v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONTAINED_PACKAGE_ROOT = (
    _PROJECT_ROOT
    / "vendor"
    / "bcnlt_runtime_v10_learning_candidate"
    / "source"
    / "bcnlt"
)
_CONTAINED_PACKAGE_NAME = "_bosscrafts_bcnlt_v10_calibration_runtime"
_EXPECTED_SOURCE_SHA256 = (
    "bd6c89a6d5a295cf95c0af545743ad9792eef67fd2d8aa641519e16b82228aa9"
)
_EXPECTED_SOURCE_FILE_COUNT = 79
_EXPECTED_SOURCE_BYTES = 691654


def _contained_source_identity() -> tuple[str, int, int]:
    return aggregate_sha256(_CONTAINED_PACKAGE_ROOT, "*.py")


def _load_contained_calibration_runtime() -> ModuleType:
    """Load the verified V10 package privately, independent of import order."""

    actual = _contained_source_identity()
    expected = (
        _EXPECTED_SOURCE_SHA256,
        _EXPECTED_SOURCE_FILE_COUNT,
        _EXPECTED_SOURCE_BYTES,
    )
    if actual != expected:
        raise ValueError("contained V10 BCNLT source identity mismatch")
    module_name = f"{_CONTAINED_PACKAGE_NAME}.uncertainty_calibration"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing
    package_spec = importlib.util.spec_from_file_location(
        _CONTAINED_PACKAGE_NAME,
        _CONTAINED_PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(_CONTAINED_PACKAGE_ROOT)],
    )
    if package_spec is None or package_spec.loader is None:
        raise ImportError("contained V10 BCNLT package could not be loaded")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[_CONTAINED_PACKAGE_NAME] = package
    try:
        package_spec.loader.exec_module(package)
        return importlib.import_module(module_name)
    except BaseException:
        for name in tuple(sys.modules):
            if name == _CONTAINED_PACKAGE_NAME or name.startswith(
                f"{_CONTAINED_PACKAGE_NAME}."
            ):
                sys.modules.pop(name, None)
        raise


_CALIBRATION_RUNTIME = _load_contained_calibration_runtime()
CalibrationThresholds = _CALIBRATION_RUNTIME.CalibrationThresholds
UncertaintyObservation = _CALIBRATION_RUNTIME.UncertaintyObservation
evaluate_uncertainty_calibration = (
    _CALIBRATION_RUNTIME.evaluate_uncertainty_calibration
)
fit_uncertainty_calibrator = _CALIBRATION_RUNTIME.fit_uncertainty_calibrator
write_calibration_artifact = _CALIBRATION_RUNTIME.write_calibration_artifact


class _CalibrationThresholdsContract(Protocol):
    minimum_heldout_samples: int
    minimum_languages: int
    minimum_samples_per_language: int
    maximum_calibrated_ece: float
    maximum_calibrated_brier: float
    maximum_brier_degradation: float


class _UncertaintyObservationContract(Protocol):
    sample_id: str


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


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object) -> str:
    canonical = _canonical(value)
    return _digest_bytes(canonical[:-1])


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class SemanticCalibrationPromotionPolicy:
    contained_thresholds: _CalibrationThresholdsContract
    strict_policy: StrictUncertaintyPolicy

    def __post_init__(self) -> None:
        # The contained runtime is deliberately loaded from a pinned source tree.
        # A verifier may load that tree more than once in one Python process, which
        # creates distinct class identities even though the threshold contract and
        # values are identical.  Normalize at this boundary so acceptance does not
        # depend on import order or a particular module-loader instance.
        try:
            contained = CalibrationThresholds(
                minimum_heldout_samples=self.contained_thresholds.minimum_heldout_samples,
                minimum_languages=self.contained_thresholds.minimum_languages,
                minimum_samples_per_language=(
                    self.contained_thresholds.minimum_samples_per_language
                ),
                maximum_calibrated_ece=self.contained_thresholds.maximum_calibrated_ece,
                maximum_calibrated_brier=(
                    self.contained_thresholds.maximum_calibrated_brier
                ),
                maximum_brier_degradation=(
                    self.contained_thresholds.maximum_brier_degradation
                ),
            )
        except AttributeError as exc:
            raise TypeError(
                "contained_thresholds must satisfy the CalibrationThresholds contract"
            ) from exc
        try:
            strict = StrictUncertaintyPolicy(
                minimum_global_selective_coverage_at_025=(
                    self.strict_policy.minimum_global_selective_coverage_at_025
                ),
                maximum_global_selective_error_rate_at_025=(
                    self.strict_policy.maximum_global_selective_error_rate_at_025
                ),
                minimum_per_language_selective_coverage_at_025=(
                    self.strict_policy.minimum_per_language_selective_coverage_at_025
                ),
                maximum_per_language_selective_error_rate_at_025=(
                    self.strict_policy.maximum_per_language_selective_error_rate_at_025
                ),
            )
        except AttributeError as exc:
            raise TypeError(
                "strict_policy must satisfy the StrictUncertaintyPolicy contract"
            ) from exc
        object.__setattr__(self, "contained_thresholds", contained)
        object.__setattr__(self, "strict_policy", strict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contained_thresholds": {
                "minimum_heldout_samples": (
                    self.contained_thresholds.minimum_heldout_samples
                ),
                "minimum_languages": self.contained_thresholds.minimum_languages,
                "minimum_samples_per_language": (
                    self.contained_thresholds.minimum_samples_per_language
                ),
                "maximum_calibrated_ece": (
                    self.contained_thresholds.maximum_calibrated_ece
                ),
                "maximum_calibrated_brier": (
                    self.contained_thresholds.maximum_calibrated_brier
                ),
                "maximum_brier_degradation": (
                    self.contained_thresholds.maximum_brier_degradation
                ),
            },
            "strict_policy": self.strict_policy.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SemanticCalibrationPromotionResult:
    output_root: Path
    promoted: bool
    failure_codes: tuple[str, ...]
    calibration_artifact_sha256: str
    strict_receipt_sha256: str
    manifest_sha256: str


def promote_semantic_uncertainty(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    policy: SemanticCalibrationPromotionPolicy,
) -> SemanticCalibrationPromotionResult:
    if type(policy) is not SemanticCalibrationPromotionPolicy:
        raise TypeError("policy must be SemanticCalibrationPromotionPolicy")
    dataset_path = Path(dataset_root).resolve()
    output_path = Path(output_root).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(dataset_path / "manifest.json")
    if manifest.get("schema_version") != _DATASET_SCHEMA:
        raise ValueError("semantic calibration dataset schema mismatch")
    if manifest.get("calibration_ready") is not True:
        raise ValueError("semantic calibration dataset is not ready")
    calibration_rows = _load_observations(
        dataset_path / str(manifest.get("calibration_file")),
        expected_sha256=manifest.get("calibration_sha256"),
        expected_count=manifest.get("calibration_count"),
        parser_identity=manifest.get("parser_runtime_identity_sha256"),
        name="calibration observations",
    )
    heldout_rows = _load_observations(
        dataset_path / str(manifest.get("heldout_file")),
        expected_sha256=manifest.get("heldout_sha256"),
        expected_count=manifest.get("heldout_count"),
        parser_identity=manifest.get("parser_runtime_identity_sha256"),
        name="heldout observations",
    )
    calibration_ids = {row.sample_id for row in calibration_rows}
    heldout_ids = {row.sample_id for row in heldout_rows}
    if calibration_ids.intersection(heldout_ids):
        raise ValueError("calibration and held-out observation identities overlap")

    calibrator = fit_uncertainty_calibrator(
        calibration_rows,
        parser_runtime_identity_sha256=str(
            manifest["parser_runtime_identity_sha256"]
        ),
    )
    artifact = evaluate_uncertainty_calibration(
        calibrator,
        heldout_rows,
        thresholds=policy.contained_thresholds,
        heldout_manifest_sha256=str(manifest["manifest_sha256"]),
    )
    strict_receipt = verify_strict_uncertainty_acceptance(
        artifact,
        policy.strict_policy,
    )
    write_calibration_artifact(
        output_path / "uncertainty-calibration.json",
        artifact,
    )
    calibrator_payload = {
        "schema_version": calibrator.schema_version,
        "parser_runtime_identity_sha256": calibrator.parser_runtime_identity_sha256,
        "calibration_data_sha256": calibrator.calibration_data_sha256,
        "calibration_sample_ids": list(calibrator.calibration_sample_ids),
        "points": [point.to_dict() for point in calibrator.points],
        "calibrator_sha256": calibrator.calibrator_sha256,
    }
    _atomic_write(output_path / "uncertainty-calibrator.json", _canonical(calibrator_payload))
    _atomic_write(
        output_path / "strict-uncertainty-acceptance.json",
        _canonical(strict_receipt.to_dict()),
    )

    failures = [f"contained:{code}" for code in artifact.failure_codes]
    failures.extend(strict_receipt.failure_codes)
    promoted = artifact.promoted and strict_receipt.passed and not failures
    unsigned = {
        "schema_version": _PROMOTION_SCHEMA,
        "source_dataset_manifest_sha256": manifest["manifest_sha256"],
        "parser_runtime_identity_sha256": manifest[
            "parser_runtime_identity_sha256"
        ],
        "policy": policy.to_dict(),
        "policy_sha256": _digest(policy.to_dict()),
        "calibrator_sha256": calibrator.calibrator_sha256,
        "calibration_artifact_sha256": artifact.artifact_sha256,
        "strict_receipt_sha256": strict_receipt.receipt_sha256,
        "promoted": promoted,
        "failure_codes": failures,
        "training_ready": False,
        "training_blockers": (
            ["model acceptance categories remain incomplete"]
            if promoted
            else ["semantic uncertainty calibration promotion failed"]
        ),
    }
    manifest_sha256 = _digest(unsigned)
    promotion_manifest = {**unsigned, "manifest_sha256": manifest_sha256}
    _atomic_write(output_path / "manifest.json", _canonical(promotion_manifest))
    return SemanticCalibrationPromotionResult(
        output_root=output_path,
        promoted=promoted,
        failure_codes=tuple(failures),
        calibration_artifact_sha256=artifact.artifact_sha256,
        strict_receipt_sha256=strict_receipt.receipt_sha256,
        manifest_sha256=manifest_sha256,
    )


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("semantic calibration dataset manifest is invalid") from exc
    if type(manifest) is not dict or raw != _canonical(manifest):
        raise ValueError("semantic calibration dataset manifest is not canonical")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != _digest(unsigned):
        raise ValueError("semantic calibration dataset manifest SHA-256 mismatch")
    return manifest


def _load_observations(
    path: Path,
    *,
    expected_sha256: object,
    expected_count: object,
    parser_identity: object,
    name: str,
) -> tuple[_UncertaintyObservationContract, ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{name} cannot be read") from exc
    if _digest_bytes(raw) != expected_sha256:
        raise ValueError(f"{name} SHA-256 mismatch")
    rows = []
    seen: set[str] = set()
    for raw_line in raw.splitlines(keepends=True):
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} JSON is invalid") from exc
        if type(payload) is not dict or raw_line != _canonical(payload):
            raise ValueError(f"{name} row is not canonical")
        if (
            payload.get("schema_version") != _OBSERVATION_SCHEMA
            or payload.get("parser_runtime_identity_sha256") != parser_identity
            or payload.get("decision") not in {"keep", "reject"}
            or payload.get("incorrect") != (payload.get("decision") == "reject")
        ):
            raise ValueError(f"{name} row lineage or label is invalid")
        sample_id = payload.get("sample_id")
        language = payload.get("language")
        raw_uncertainty = payload.get("raw_uncertainty")
        incorrect = payload.get("incorrect")
        if (
            type(sample_id) is not str
            or not sample_id
            or sample_id in seen
            or type(language) is not str
            or not language.strip()
            or isinstance(raw_uncertainty, bool)
            or not isinstance(raw_uncertainty, (int, float))
            or type(incorrect) is not bool
        ):
            raise ValueError(f"{name} contains an invalid or duplicate sample ID")
        seen.add(sample_id)
        rows.append(
            UncertaintyObservation(
                sample_id=sample_id,
                language=language,
                raw_uncertainty=float(raw_uncertainty),
                incorrect=incorrect,
            )
        )
    if type(expected_count) is not int or len(rows) != expected_count:
        raise ValueError(f"{name} count mismatch")
    return tuple(rows)
