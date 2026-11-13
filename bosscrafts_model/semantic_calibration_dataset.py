"""Freeze dual-reviewed semantic records into disjoint calibration splits."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .heldout_semantic_review import REVIEW_ITEM_SCHEMA, REVIEW_MANIFEST_SCHEMA
from .jsonl import split_jsonl_records
from .local_semantic_review import (
    DualLocalSemanticReviewer,
    LocalReviewerSpec,
)

_MANIFEST_SCHEMA = "bosscrafts-semantic-calibration-dataset-v1"
_OBSERVATION_SCHEMA = "bosscrafts-semantic-uncertainty-observation-v1"
_SIGNAL_WEIGHTS = {
    "unknown semantic": 0.55,
    "missing definition": 0.25,
    "dynamic call target": 0.35,
    "dynamic binding": 0.30,
    "macro expansion": 0.35,
    "overload unresolved": 0.25,
    "preprocessor condition": 0.20,
    "member type unknown": 0.10,
    "missing AST relation": 0.10,
}


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


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
class SemanticCalibrationDatasetResult:
    output_root: Path
    calibration_count: int
    heldout_count: int
    language_counts: dict[str, dict[str, int]]
    manifest_sha256: str


class SemanticCalibrationDatasetBuilder:
    def __init__(
        self,
        *,
        calibration_per_language: int,
        heldout_per_language: int,
    ) -> None:
        if (
            type(calibration_per_language) is not int
            or calibration_per_language < 1
            or type(heldout_per_language) is not int
            or heldout_per_language < 1
        ):
            raise ValueError("calibration and held-out counts must be positive integers")
        self.calibration_per_language = calibration_per_language
        self.heldout_per_language = heldout_per_language

    def materialize(
        self,
        queue_root: str | Path,
        dual_review_root: str | Path,
        manual_decisions: Iterable[object],
        output_root: str | Path,
    ) -> SemanticCalibrationDatasetResult:
        queue_path = Path(queue_root).resolve()
        dual_path = Path(dual_review_root).resolve()
        output_path = Path(output_root).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        dual_manifest = self._load_hashed_manifest(dual_path / "manifest.json")
        reviewer_rows = dual_manifest.get("reviewers")
        if type(reviewer_rows) is not list or len(reviewer_rows) != 2:
            raise ValueError("dual review manifest reviewer inventory is invalid")
        try:
            reviewers = tuple(
                LocalReviewerSpec(
                    str(row["reviewer_id"]),
                    str(row["base_url"]),
                    str(row["model"]),
                    timeout_seconds=float(row["timeout_seconds"]),
                )
                for row in reviewer_rows
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("dual review manifest reviewer identity is invalid") from exc
        verifier = DualLocalSemanticReviewer(queue_path, dual_path, reviewers)  # type: ignore[arg-type]
        verified = verifier.verify()
        if not verified.review_complete:
            raise ValueError("dual semantic review is incomplete")

        dual_manifest = self._load_hashed_manifest(dual_path / "manifest.json")
        queue_manifest, items = self._load_queue(queue_path)
        if (
            dual_manifest.get("source_review_manifest_sha256")
            != queue_manifest.get("manifest_sha256")
            or dual_manifest.get("source_queue_sha256")
            != queue_manifest.get("queue_sha256")
            or dual_manifest.get("parser_runtime_identity_sha256")
            != queue_manifest.get("parser_runtime_identity_sha256")
        ):
            raise ValueError("dual review and queue lineage mismatch")
        consensus = self._load_hashed_jsonl(
            dual_path / str(dual_manifest.get("consensus_file")),
            dual_manifest.get("consensus_sha256"),
            name="dual review consensus",
        )
        if len(consensus) != len(items):
            raise ValueError("dual review consensus item count mismatch")
        consensus_by_id = {str(row.get("review_id")): row for row in consensus}
        if len(consensus_by_id) != len(consensus):
            raise ValueError("dual review consensus contains duplicate IDs")

        manual_by_id, manual_text = self._validate_manual_decisions(
            manual_decisions,
            consensus_by_id,
        )
        items_by_id = {str(item["review_id"]): item for item in items}
        final_rows: list[dict[str, object]] = []
        for review_id, item in items_by_id.items():
            consensus_row = consensus_by_id.get(review_id)
            if consensus_row is None:
                raise ValueError("dual review consensus is missing an item")
            status = consensus_row.get("status")
            if status == "agree_keep":
                decision = "keep"
                resolution = "dual_agreement"
                resolver = None
            elif status == "agree_reject":
                decision = "reject"
                resolution = "dual_agreement"
                resolver = None
            elif status == "manual_review":
                manual = manual_by_id.get(review_id)
                if manual is None:
                    raise ValueError(f"missing manual decisions: {review_id}")
                decision = str(manual["decision"])
                resolution = "manual"
                resolver = str(manual["reviewer"])
            else:
                raise ValueError("dual semantic review contains a pending item")
            raw_uncertainty = self._raw_uncertainty(item["canonical_record"])
            final_rows.append(
                {
                    "schema_version": _OBSERVATION_SCHEMA,
                    "sample_id": review_id,
                    "review_id": review_id,
                    "language": item["language"],
                    "raw_uncertainty": raw_uncertainty,
                    "incorrect": decision == "reject",
                    "decision": decision,
                    "resolution": resolution,
                    "resolver": resolver,
                    "reviewer_entry_sha256": consensus_row.get(
                        "reviewer_entry_sha256"
                    ),
                    "parser_runtime_identity_sha256": item[
                        "parser_runtime_identity_sha256"
                    ],
                }
            )

        calibration: list[dict[str, object]] = []
        heldout: list[dict[str, object]] = []
        language_counts: dict[str, dict[str, int]] = {}
        for language in sorted({str(row["language"]) for row in final_rows}):
            language_rows = sorted(
                (row for row in final_rows if row["language"] == language),
                key=lambda row: str(row["sample_id"]),
            )
            required = self.calibration_per_language + self.heldout_per_language
            if len(language_rows) < required:
                raise ValueError(
                    f"language {language} has {len(language_rows)} labels but requires {required}"
                )
            calibration.extend(language_rows[: self.calibration_per_language])
            heldout.extend(language_rows[self.calibration_per_language : required])
            language_counts[language] = {
                "calibration": self.calibration_per_language,
                "heldout": self.heldout_per_language,
            }
        calibration.sort(key=lambda row: str(row["sample_id"]))
        heldout.sort(key=lambda row: str(row["sample_id"]))
        if {row["sample_id"] for row in calibration}.intersection(
            row["sample_id"] for row in heldout
        ):
            raise ValueError("calibration and held-out observation identities overlap")

        def label_count(rows: list[dict[str, object]]) -> dict[str, int]:
            rejects = sum(row["incorrect"] is True for row in rows)
            return {"keep": len(rows) - rejects, "reject": rejects}

        label_counts: dict[str, object] = {
            "global": {
                "calibration": label_count(calibration),
                "heldout": label_count(heldout),
            },
            "per_language": {},
        }
        calibration_blockers: list[str] = []
        for split_name, split_rows in (
            ("calibration", calibration),
            ("heldout", heldout),
        ):
            counts = label_count(split_rows)
            for label in ("keep", "reject"):
                if counts[label] < 1:
                    calibration_blockers.append(
                        f"calibration_split_missing_{label}:{split_name}"
                    )
        per_language_counts: dict[str, object] = {}
        for language in sorted(language_counts):
            split_counts: dict[str, dict[str, int]] = {}
            for split_name, split_rows in (
                ("calibration", calibration),
                ("heldout", heldout),
            ):
                counts = label_count(
                    [row for row in split_rows if row["language"] == language]
                )
                split_counts[split_name] = counts
                for label in ("keep", "reject"):
                    if counts[label] < 1:
                        calibration_blockers.append(
                            f"language_split_missing_{label}:{language}:{split_name}"
                        )
            per_language_counts[language] = split_counts
        label_counts["per_language"] = per_language_counts
        calibration_ready = not calibration_blockers

        calibration_text = "".join(f"{_canonical(row)}\n" for row in calibration)
        heldout_text = "".join(f"{_canonical(row)}\n" for row in heldout)
        _atomic_write(
            output_path / "calibration_observations.jsonl",
            calibration_text.encode("utf-8"),
        )
        _atomic_write(
            output_path / "heldout_observations.jsonl",
            heldout_text.encode("utf-8"),
        )
        _atomic_write(
            output_path / "manual_decisions.jsonl",
            manual_text.encode("utf-8"),
        )
        unsigned = {
            "schema_version": _MANIFEST_SCHEMA,
            "source_review_manifest_sha256": queue_manifest["manifest_sha256"],
            "source_queue_sha256": queue_manifest["queue_sha256"],
            "source_dual_review_manifest_sha256": dual_manifest["manifest_sha256"],
            "parser_runtime_identity_sha256": queue_manifest[
                "parser_runtime_identity_sha256"
            ],
            "calibration_per_language": self.calibration_per_language,
            "heldout_per_language": self.heldout_per_language,
            "language_counts": language_counts,
            "label_counts": label_counts,
            "calibration_file": "calibration_observations.jsonl",
            "calibration_count": len(calibration),
            "calibration_sha256": _sha256_text(calibration_text),
            "heldout_file": "heldout_observations.jsonl",
            "heldout_count": len(heldout),
            "heldout_sha256": _sha256_text(heldout_text),
            "manual_decisions_file": "manual_decisions.jsonl",
            "manual_decisions_sha256": _sha256_text(manual_text),
            "calibration_ready": calibration_ready,
            "calibration_blockers": calibration_blockers,
            "training_ready": False,
            "training_blockers": [
                *calibration_blockers,
                "calibration metrics and strict selective-risk gate have not passed",
                "model acceptance categories remain incomplete",
            ],
        }
        manifest_sha256 = _sha256_text(_canonical(unsigned))
        manifest = {**unsigned, "manifest_sha256": manifest_sha256}
        _atomic_write(
            output_path / "manifest.json",
            f"{_canonical(manifest)}\n".encode(),
        )
        return SemanticCalibrationDatasetResult(
            output_root=output_path,
            calibration_count=len(calibration),
            heldout_count=len(heldout),
            language_counts=language_counts,
            manifest_sha256=manifest_sha256,
        )

    @staticmethod
    def _load_hashed_manifest(path: Path) -> dict[str, object]:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"manifest is invalid: {path}") from exc
        if type(manifest) is not dict:
            raise ValueError("manifest must be an object")
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if claimed != _sha256_text(_canonical(unsigned)):
            raise ValueError("manifest SHA-256 mismatch")
        return manifest

    @classmethod
    def _load_queue(
        cls, root: Path
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        manifest = cls._load_hashed_manifest(root / "manifest.json")
        if manifest.get("schema_version") != REVIEW_MANIFEST_SCHEMA:
            raise ValueError("review queue manifest schema mismatch")
        rows = cls._load_hashed_jsonl(
            root / str(manifest.get("queue_file")),
            manifest.get("queue_sha256"),
            name="review queue",
        )
        if (
            len(rows) != manifest.get("item_count")
            or any(row.get("schema_version") != REVIEW_ITEM_SCHEMA for row in rows)
        ):
            raise ValueError("review queue inventory mismatch")
        return manifest, rows

    @staticmethod
    def _load_hashed_jsonl(
        path: Path,
        expected_sha256: object,
        *,
        name: str,
    ) -> list[dict[str, object]]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"{name} cannot be read") from exc
        if _sha256_text(text) != expected_sha256:
            raise ValueError(f"{name} SHA-256 mismatch")
        rows = []
        for line in split_jsonl_records(text):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} JSON is invalid") from exc
            if type(row) is not dict:
                raise ValueError(f"{name} row must be an object")
            rows.append(row)
        return rows

    @staticmethod
    def _validate_manual_decisions(
        decisions: Iterable[object],
        consensus_by_id: dict[str, dict[str, object]],
    ) -> tuple[dict[str, dict[str, object]], str]:
        try:
            supplied = list(decisions)
        except TypeError as exc:
            raise TypeError("manual decisions must be iterable") from exc
        required = {
            review_id
            for review_id, row in consensus_by_id.items()
            if row.get("status") == "manual_review"
        }
        by_id: dict[str, dict[str, object]] = {}
        for decision in supplied:
            if type(decision) is not dict or set(decision) != {
                "review_id",
                "decision",
                "reviewer",
                "notes",
            }:
                raise ValueError("manual decision fields mismatch")
            review_id = decision.get("review_id")
            if type(review_id) is not str or review_id not in required:
                raise ValueError("manual decision references a non-manual review item")
            if review_id in by_id:
                raise ValueError("manual decisions contain a duplicate review ID")
            if decision.get("decision") not in {"keep", "reject"}:
                raise ValueError("manual decision must be keep or reject")
            if (
                type(decision.get("reviewer")) is not str
                or not str(decision["reviewer"]).strip()
                or type(decision.get("notes")) is not str
                or not str(decision["notes"]).strip()
            ):
                raise ValueError("manual decision reviewer and notes are required")
            by_id[review_id] = decision
        missing = sorted(required - set(by_id))
        if missing:
            raise ValueError(f"missing manual decisions: {missing}")
        if set(by_id) - required:
            raise ValueError("unexpected manual decisions were supplied")
        ordered = [by_id[review_id] for review_id in sorted(by_id)]
        return by_id, "".join(f"{_canonical(row)}\n" for row in ordered)

    @staticmethod
    def _raw_uncertainty(record: object) -> float:
        if type(record) is not dict:
            raise ValueError("canonical semantic record is invalid")
        metadata = record.get("metadata", {})
        if type(metadata) is not dict:
            raise ValueError("canonical semantic uncertainty metadata is invalid")
        signals = metadata.get("uncertainty_signals", [])
        if (
            type(signals) is not list
            or any(type(signal) is not str for signal in signals)
            or len(signals) != len(set(signals))
        ):
            raise ValueError("canonical semantic uncertainty signals are invalid")
        unknown = sorted(set(signals) - set(_SIGNAL_WEIGHTS))
        if unknown:
            raise ValueError(f"unsupported raw uncertainty signals: {unknown}")
        return min(1.0, 0.05 + sum(_SIGNAL_WEIGHTS[signal] for signal in signals))
