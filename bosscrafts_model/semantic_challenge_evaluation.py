"""Fail-closed qualification of local reviewers on blinded semantic controls."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from .jsonl import split_jsonl_records
from .local_semantic_review import DualLocalSemanticReviewer, LocalReviewerSpec
from .semantic_challenge_review import SemanticChallengeReviewBuilder

_REPORT_SCHEMA = "bosscrafts-semantic-challenge-qualification-v1"
_REVIEW_MANIFEST_SCHEMA = "bosscrafts-dual-local-semantic-review-manifest-v2"
_DEFECT_TYPES = (
    "surface_mismatch",
    "span_mismatch",
    "scope_mismatch",
    "event_ir_mismatch",
    "uncertainty_understated",
)


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


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
class SemanticChallengeQualificationPolicy:
    minimum_terminal_items: int = 10
    minimum_positive_controls: int = 5
    minimum_negative_controls: int = 5
    minimum_accuracy: float = 1.0
    minimum_negative_sensitivity: float = 1.0
    minimum_positive_specificity: float = 1.0
    required_defect_types: tuple[str, ...] = _DEFECT_TYPES
    require_no_exhausted_failures: bool = True

    def __post_init__(self) -> None:
        for name in (
            "minimum_terminal_items",
            "minimum_positive_controls",
            "minimum_negative_controls",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "minimum_accuracy",
            "minimum_negative_sensitivity",
            "minimum_positive_specificity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
            object.__setattr__(self, name, float(value))
        if (
            type(self.required_defect_types) is not tuple
            or not self.required_defect_types
            or any(value not in _DEFECT_TYPES for value in self.required_defect_types)
            or len(set(self.required_defect_types)) != len(self.required_defect_types)
        ):
            raise ValueError("required_defect_types must be unique supported defects")
        if type(self.require_no_exhausted_failures) is not bool:
            raise TypeError("require_no_exhausted_failures must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_terminal_items": self.minimum_terminal_items,
            "minimum_positive_controls": self.minimum_positive_controls,
            "minimum_negative_controls": self.minimum_negative_controls,
            "minimum_accuracy": self.minimum_accuracy,
            "minimum_negative_sensitivity": self.minimum_negative_sensitivity,
            "minimum_positive_specificity": self.minimum_positive_specificity,
            "required_defect_types": list(self.required_defect_types),
            "require_no_exhausted_failures": self.require_no_exhausted_failures,
        }


@dataclass(frozen=True, slots=True)
class SemanticChallengeQualificationResult:
    output_root: Path
    qualification_passed: bool
    reviewer_count: int
    report_sha256: str


class SemanticChallengeEvaluator:
    """Score reviewer evidence without exposing the answer key during review."""

    def __init__(
        self, policy: SemanticChallengeQualificationPolicy | None = None
    ) -> None:
        self.policy = policy or SemanticChallengeQualificationPolicy()
        if type(self.policy) is not SemanticChallengeQualificationPolicy:
            raise TypeError("policy must be SemanticChallengeQualificationPolicy")

    def materialize(
        self,
        challenge_root: str | Path,
        review_root: str | Path,
        output_root: str | Path,
    ) -> SemanticChallengeQualificationResult:
        output = Path(output_root).resolve()
        report = self._evaluate(Path(challenge_root).resolve(), Path(review_root).resolve())
        _atomic_write(
            output / "qualification.json",
            f"{_canonical(report)}\n".encode(),
        )
        return self._result(output, report)

    def verify(
        self,
        challenge_root: str | Path,
        review_root: str | Path,
        output_root: str | Path,
    ) -> SemanticChallengeQualificationResult:
        output = Path(output_root).resolve()
        try:
            existing = json.loads(
                (output / "qualification.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("semantic challenge qualification report is invalid") from exc
        expected = self._evaluate(Path(challenge_root).resolve(), Path(review_root).resolve())
        if existing != expected:
            raise ValueError("semantic challenge qualification report is stale or tampered")
        if type(existing) is not dict:
            raise ValueError("semantic challenge qualification report is invalid")
        return self._result(output, existing)

    @staticmethod
    def _result(
        output: Path, report: dict[str, object]
    ) -> SemanticChallengeQualificationResult:
        passed = report.get("qualification_passed")
        reviewer_metrics = report.get("reviewer_metrics")
        report_sha256 = report.get("report_sha256")
        if (
            type(passed) is not bool
            or type(reviewer_metrics) is not list
            or any(type(row) is not dict for row in reviewer_metrics)
            or not _is_sha256(report_sha256)
        ):
            raise ValueError("semantic challenge qualification result is invalid")
        return SemanticChallengeQualificationResult(
            output_root=output,
            qualification_passed=passed,
            reviewer_count=len(reviewer_metrics),
            report_sha256=report_sha256,
        )

    def _evaluate(self, challenge_root: Path, review_root: Path) -> dict[str, object]:
        challenge_manifest = self._load_manifest(
            challenge_root / "manifest.json", "semantic challenge"
        )
        pairs_per_language = challenge_manifest.get("pairs_per_language")
        if type(pairs_per_language) is not int:
            raise ValueError("semantic challenge pairs-per-language is invalid")
        SemanticChallengeReviewBuilder(
            pairs_per_language=pairs_per_language
        ).verify(challenge_root)

        review_manifest = self._load_manifest(
            review_root / "manifest.json", "local review"
        )
        if review_manifest.get("schema_version") != _REVIEW_MANIFEST_SCHEMA:
            raise ValueError("local review manifest schema mismatch")
        if (
            review_manifest.get("source_review_manifest_sha256")
            != challenge_manifest.get("manifest_sha256")
            or review_manifest.get("source_queue_sha256")
            != challenge_manifest.get("queue_sha256")
            or review_manifest.get("parser_runtime_identity_sha256")
            != challenge_manifest.get("parser_runtime_identity_sha256")
        ):
            raise ValueError("local review and semantic challenge lineage mismatch")
        reviewers = self._reviewers(review_manifest)
        review_protocol = review_manifest.get("review_protocol", "legacy-v1")
        if review_protocol not in {
            "legacy-v1",
            "span-evidence-v2",
            "contract-aware-v3",
        }:
            raise ValueError("local review protocol is unsupported")
        DualLocalSemanticReviewer(
            challenge_root,
            review_root,
            reviewers,
            review_protocol=str(review_protocol),
        ).verify()
        review_manifest = self._load_manifest(
            review_root / "manifest.json", "local review"
        )

        answers = self._load_rows(
            challenge_root / str(challenge_manifest["answer_key_file"]),
            "semantic challenge answer key",
        )
        answer_by_id = {str(row["review_id"]): row for row in answers}
        decisions = self._load_rows(
            review_root / "reviewer_decisions.jsonl",
            "review decision ledger",
            missing_ok=True,
        )
        failures = self._load_rows(
            review_root / "reviewer_failures.jsonl",
            "review failure ledger",
            missing_ok=True,
        )
        decisions_by_pair = {
            (str(row["review_id"]), str(row["reviewer_id"])): row
            for row in decisions
        }
        latest_failure_by_pair = {
            (str(row["review_id"]), str(row["reviewer_id"])): row
            for row in failures
        }
        exhausted_pairs = {
            pair
            for pair, row in latest_failure_by_pair.items()
            if pair not in decisions_by_pair and row.get("request_kind") == "repair"
        }

        reviewer_metrics = []
        terminal_sets: list[set[str]] = []
        for reviewer in reviewers:
            reviewer_id = reviewer.reviewer_id
            reviewer_decisions = {
                pair[0]: row
                for pair, row in decisions_by_pair.items()
                if pair[1] == reviewer_id
            }
            reviewer_exhausted = {
                pair[0] for pair in exhausted_pairs if pair[1] == reviewer_id
            }
            terminal_ids = set(reviewer_decisions) | reviewer_exhausted
            terminal_sets.append(terminal_ids)
            reviewer_metrics.append(
                self._score_reviewer(
                    reviewer,
                    answer_by_id,
                    reviewer_decisions,
                    reviewer_exhausted,
                )
            )

        samples_aligned = bool(terminal_sets) and all(
            ids == terminal_sets[0] for ids in terminal_sets[1:]
        )
        blockers: list[str] = []
        if not samples_aligned:
            blockers.append("reviewers were not evaluated on identical controls")
        if not all(row["qualification_passed"] for row in reviewer_metrics):
            blockers.append("one or more reviewers failed qualification")
        qualification_passed = not blockers
        unsigned = {
            "schema_version": _REPORT_SCHEMA,
            "challenge_manifest_sha256": challenge_manifest["manifest_sha256"],
            "challenge_queue_sha256": challenge_manifest["queue_sha256"],
            "challenge_answer_key_sha256": challenge_manifest["answer_key_sha256"],
            "review_manifest_sha256": review_manifest["manifest_sha256"],
            "review_decisions_sha256": review_manifest[
                "reviewer_decisions_sha256"
            ],
            "review_failures_sha256": review_manifest["failure_ledger_sha256"],
            "policy": self.policy.to_dict(),
            "samples_aligned": samples_aligned,
            "reviewer_metrics": reviewer_metrics,
            "qualification_passed": qualification_passed,
            "blockers": blockers,
            "training_ready": False,
            "training_blockers": [
                "reviewer qualification alone does not authorize model training"
            ],
        }
        return {**unsigned, "report_sha256": _sha256_text(_canonical(unsigned))}

    def _score_reviewer(
        self,
        reviewer: LocalReviewerSpec,
        answers: dict[str, dict[str, object]],
        decisions: dict[str, dict[str, object]],
        exhausted: set[str],
    ) -> dict[str, object]:
        terminal_ids = sorted(set(decisions) | exhausted)
        if any(review_id not in answers for review_id in terminal_ids):
            raise ValueError("review evidence references an unknown challenge control")
        scored = []
        for review_id in terminal_ids:
            answer = answers[review_id]
            decision_row = decisions.get(review_id, {})
            actual = decision_row.get("decision")
            label_correct = actual == answer["expected_decision"]
            defect_detected = (
                label_correct
                if answer["expected_decision"] == "keep"
                else label_correct
                and self._detects_planted_defect(
                    str(answer["defect_type"]), decision_row.get("rubric")
                )
            )
            scored.append(
                {
                    "review_id": review_id,
                    "expected": answer["expected_decision"],
                    "actual": actual,
                    "correct": defect_detected,
                    "label_correct": label_correct,
                    "defect_type": answer["defect_type"],
                    "language": answer["language"],
                }
            )
        positives = [row for row in scored if row["expected"] == "keep"]
        negatives = [row for row in scored if row["expected"] == "reject"]
        correct = sum(bool(row["correct"]) for row in scored)
        true_keeps = sum(bool(row["correct"]) for row in positives)
        true_rejects = sum(bool(row["correct"]) for row in negatives)
        accuracy = correct / len(scored) if scored else 0.0
        sensitivity = true_rejects / len(negatives) if negatives else 0.0
        specificity = true_keeps / len(positives) if positives else 0.0
        defect_metrics = [
            self._group_metrics(
                defect,
                [row for row in negatives if row["defect_type"] == defect],
            )
            for defect in self.policy.required_defect_types
        ]
        languages = sorted({str(row["language"]) for row in scored})
        language_metrics = [
            self._group_metrics(
                language,
                [row for row in scored if row["language"] == language],
            )
            for language in languages
        ]
        blockers: list[str] = []
        if len(scored) < self.policy.minimum_terminal_items:
            blockers.append("insufficient terminal controls")
        if len(positives) < self.policy.minimum_positive_controls:
            blockers.append("insufficient positive controls")
        if len(negatives) < self.policy.minimum_negative_controls:
            blockers.append("insufficient negative controls")
        missing_defects = [
            str(row["group"]) for row in defect_metrics if row["items"] == 0
        ]
        if missing_defects:
            blockers.append(
                "missing required defect controls: " + ", ".join(missing_defects)
            )
        if self.policy.require_no_exhausted_failures and exhausted:
            blockers.append("one or more reviewer calls exhausted bounded repair")
        if accuracy < self.policy.minimum_accuracy:
            blockers.append("overall accuracy below threshold")
        if sensitivity < self.policy.minimum_negative_sensitivity:
            blockers.append("negative-control sensitivity below threshold")
        if specificity < self.policy.minimum_positive_specificity:
            blockers.append("positive-control specificity below threshold")
        return {
            "reviewer_id": reviewer.reviewer_id,
            "model": reviewer.model,
            "terminal_items": len(scored),
            "decision_items": len(decisions),
            "exhausted_items": len(exhausted),
            "correct_items": correct,
            "overall_accuracy": accuracy,
            "positive_controls": len(positives),
            "true_keeps": true_keeps,
            "positive_specificity": specificity,
            "negative_controls": len(negatives),
            "true_rejects": true_rejects,
            "negative_sensitivity": sensitivity,
            "defect_metrics": defect_metrics,
            "language_metrics": language_metrics,
            "qualification_passed": not blockers,
            "blockers": blockers,
        }

    @staticmethod
    def _detects_planted_defect(defect: str, rubric: object) -> bool:
        if type(rubric) is not dict:
            return False
        expected_false_fields = {
            "surface_mismatch": ("event_matches_source", "span_matches_source"),
            "span_mismatch": ("span_matches_source",),
            "scope_mismatch": ("scope_plausible",),
            "event_ir_mismatch": ("event_matches_source", "ir_kind_plausible"),
            "uncertainty_understated": ("uncertainty_calibrated",),
        }.get(defect)
        return expected_false_fields is not None and any(
            rubric.get(field) is False for field in expected_false_fields
        )

    @staticmethod
    def _group_metrics(group: str, rows: list[dict[str, object]]) -> dict[str, object]:
        correct = sum(bool(row["correct"]) for row in rows)
        return {
            "group": group,
            "items": len(rows),
            "correct_items": correct,
            "accuracy": correct / len(rows) if rows else None,
        }

    @staticmethod
    def _load_manifest(path: Path, name: str) -> dict[str, object]:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} manifest is invalid") from exc
        if type(manifest) is not dict:
            raise ValueError(f"{name} manifest must be an object")
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if claimed != _sha256_text(_canonical(unsigned)):
            raise ValueError(f"{name} manifest SHA-256 mismatch")
        return manifest

    @staticmethod
    def _load_rows(path: Path, name: str, *, missing_ok: bool = False) -> list[dict[str, object]]:
        if missing_ok and not path.exists():
            return []
        try:
            lines = split_jsonl_records(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"{name} cannot be read") from exc
        rows: list[dict[str, object]] = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} JSON is invalid") from exc
            if type(row) is not dict or line != _canonical(row):
                raise ValueError(f"{name} is not canonical")
            rows.append(row)
        return rows

    @staticmethod
    def _reviewers(
        manifest: dict[str, object]
    ) -> tuple[LocalReviewerSpec, LocalReviewerSpec]:
        rows = manifest.get("reviewers")
        if type(rows) is not list or len(rows) != 2 or any(type(row) is not dict for row in rows):
            raise ValueError("local review reviewer inventory is invalid")
        reviewers = tuple(
            LocalReviewerSpec(
                str(row.get("reviewer_id", "")),
                str(row.get("base_url", "")),
                str(row.get("model", "")),
                timeout_seconds=row.get("timeout_seconds", 120.0),
            )
            for row in rows
        )
        return reviewers  # type: ignore[return-value]

