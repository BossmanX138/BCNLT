"""Qualification of combined deterministic and raw-AST semantic audits."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .semantic_challenge_review import SemanticChallengeReviewBuilder


_REPORT_SCHEMA = "bosscrafts-combined-semantic-audit-qualification-v1"
_DETERMINISTIC_SCHEMA = "bosscrafts-deterministic-semantic-audit-manifest-v1"
_DETERMINISTIC_FINDING_SCHEMA = "bosscrafts-deterministic-semantic-finding-v1"
_RAW_AST_SCHEMA = "bosscrafts-raw-ast-semantic-audit-manifest-v1"
_RAW_AST_ITEM_SCHEMA = "bosscrafts-raw-ast-semantic-witness-item-v1"
_ANSWER_SCHEMA = "bosscrafts-semantic-challenge-answer-v1"
_DEFECT_TYPES = (
    "surface_mismatch",
    "span_mismatch",
    "scope_mismatch",
    "event_ir_mismatch",
    "uncertainty_understated",
)
_RAW_AST_SIDECAR = Path(__file__).with_name("raw_ast_witness_sidecar.py")


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
class SemanticAuditQualificationResult:
    output_root: Path
    audit_qualification_passed: bool
    positive_control_count: int
    positive_false_flag_count: int
    negative_control_count: int
    negative_detected_count: int
    training_ready: bool
    report_sha256: str


class SemanticAuditQualificationEvaluator:
    """Require perfect union coverage without flagging unchanged controls."""

    def materialize(
        self,
        challenge_root: str | Path,
        deterministic_root: str | Path,
        raw_ast_root: str | Path,
        output_root: str | Path,
    ) -> SemanticAuditQualificationResult:
        output = Path(output_root).resolve()
        report = self._evaluate(
            Path(challenge_root).resolve(),
            Path(deterministic_root).resolve(),
            Path(raw_ast_root).resolve(),
        )
        _atomic_write(
            output / "qualification.json", f"{_canonical(report)}\n".encode("utf-8")
        )
        return self._result(output, report)

    def verify(
        self,
        challenge_root: str | Path,
        deterministic_root: str | Path,
        raw_ast_root: str | Path,
        output_root: str | Path,
    ) -> SemanticAuditQualificationResult:
        output = Path(output_root).resolve()
        expected = self._evaluate(
            Path(challenge_root).resolve(),
            Path(deterministic_root).resolve(),
            Path(raw_ast_root).resolve(),
        )
        try:
            actual = (output / "qualification.json").read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError("semantic audit qualification is stale or tampered") from exc
        if actual != f"{_canonical(expected)}\n":
            raise ValueError("semantic audit qualification is stale or tampered")
        return self._result(output, expected)

    def _evaluate(
        self, challenge_root: Path, deterministic_root: Path, raw_ast_root: Path
    ) -> dict[str, object]:
        challenge_manifest = self._load_manifest(
            challenge_root / "manifest.json", "semantic challenge"
        )
        pairs_per_language = challenge_manifest.get("pairs_per_language")
        if type(pairs_per_language) is not int:
            raise ValueError("semantic challenge pair count is invalid")
        SemanticChallengeReviewBuilder(
            pairs_per_language=pairs_per_language
        ).verify(challenge_root)
        queue_rows = self._read_rows(
            challenge_root / str(challenge_manifest["queue_file"]),
            challenge_manifest["queue_sha256"],
            "semantic challenge queue",
        )
        answer_rows = self._read_rows(
            challenge_root / str(challenge_manifest["answer_key_file"]),
            challenge_manifest["answer_key_sha256"],
            "semantic challenge answer key",
        )
        queue_ids = {str(row["review_id"]) for row in queue_rows}
        answer_by_id: dict[str, dict[str, object]] = {}
        for row in answer_rows:
            review_id = row.get("review_id")
            if (
                row.get("schema_version") != _ANSWER_SCHEMA
                or type(review_id) is not str
                or review_id in answer_by_id
                or row.get("expected_decision") not in {"keep", "reject"}
                or row.get("defect_type") not in {"none", *_DEFECT_TYPES}
            ):
                raise ValueError("semantic challenge answer row is invalid")
            answer_by_id[review_id] = row
        if set(answer_by_id) != queue_ids:
            raise ValueError("semantic challenge queue and answer IDs differ")

        deterministic_manifest = self._load_manifest(
            deterministic_root / "manifest.json", "deterministic audit"
        )
        if (
            deterministic_manifest.get("schema_version") != _DETERMINISTIC_SCHEMA
            or deterministic_manifest.get("source_review_manifest_sha256")
            != challenge_manifest.get("manifest_sha256")
            or deterministic_manifest.get("source_queue_sha256")
            != challenge_manifest.get("queue_sha256")
            or deterministic_manifest.get("parser_runtime_identity_sha256")
            != challenge_manifest.get("parser_runtime_identity_sha256")
        ):
            raise ValueError("deterministic audit challenge lineage mismatch")
        deterministic_rows = self._read_rows(
            deterministic_root / str(deterministic_manifest["findings_file"]),
            deterministic_manifest["findings_sha256"],
            "deterministic audit findings",
        )
        deterministic_ids: set[str] = set()
        for row in deterministic_rows:
            review_id = row.get("review_id")
            if (
                row.get("schema_version") != _DETERMINISTIC_FINDING_SCHEMA
                or type(review_id) is not str
                or review_id not in queue_ids
            ):
                raise ValueError("deterministic audit finding is invalid")
            deterministic_ids.add(review_id)
        if (
            len(deterministic_rows) != deterministic_manifest.get("finding_count")
            or len(deterministic_ids)
            != deterministic_manifest.get("failed_item_count")
        ):
            raise ValueError("deterministic audit finding inventory mismatch")

        raw_manifest = self._load_manifest(
            raw_ast_root / "manifest.json", "raw AST audit"
        )
        if (
            raw_manifest.get("schema_version") != _RAW_AST_SCHEMA
            or raw_manifest.get("source_review_manifest_sha256")
            != challenge_manifest.get("manifest_sha256")
            or raw_manifest.get("source_queue_sha256")
            != challenge_manifest.get("queue_sha256")
            or raw_manifest.get("parser_runtime_identity_sha256")
            != challenge_manifest.get("parser_runtime_identity_sha256")
            or raw_manifest.get("witness_sidecar_sha256")
            != _sha256_file(_RAW_AST_SIDECAR)
            or raw_manifest.get("complete_source_queue") is not True
        ):
            raise ValueError("raw AST audit challenge lineage mismatch")
        raw_rows = self._read_rows(
            raw_ast_root / str(raw_manifest["witnesses_file"]),
            raw_manifest["witnesses_sha256"],
            "raw AST witnesses",
        )
        raw_by_id: dict[str, dict[str, object]] = {}
        raw_flagged_ids: set[str] = set()
        for row in raw_rows:
            review_id = row.get("review_id")
            codes = row.get("finding_codes")
            if (
                row.get("schema_version") != _RAW_AST_ITEM_SCHEMA
                or type(review_id) is not str
                or review_id in raw_by_id
                or review_id not in queue_ids
                or type(codes) is not list
                or any(type(code) is not str or not code for code in codes)
            ):
                raise ValueError("raw AST witness row is invalid")
            raw_by_id[review_id] = row
            if codes:
                raw_flagged_ids.add(review_id)
        if (
            set(raw_by_id) != queue_ids
            or len(raw_rows) != raw_manifest.get("witness_item_count")
            or len(raw_flagged_ids) != raw_manifest.get("failed_item_count")
        ):
            raise ValueError("raw AST witness inventory mismatch")

        flagged_ids = deterministic_ids | raw_flagged_ids
        positive_ids = {
            review_id
            for review_id, row in answer_by_id.items()
            if row["expected_decision"] == "keep"
        }
        negative_ids = set(answer_by_id) - positive_ids
        defect_metrics: dict[str, dict[str, int]] = {}
        for defect in _DEFECT_TYPES:
            ids = {
                review_id
                for review_id, row in answer_by_id.items()
                if row["defect_type"] == defect
            }
            defect_metrics[defect] = {
                "detected": len(ids & flagged_ids),
                "total": len(ids),
            }
        positive_false_flags = len(positive_ids & flagged_ids)
        negative_detected = len(negative_ids & flagged_ids)
        qualified = (
            positive_false_flags == 0
            and negative_detected == len(negative_ids)
            and all(
                metric["detected"] == metric["total"] and metric["total"] > 0
                for metric in defect_metrics.values()
            )
        )
        unsigned: dict[str, object] = {
            "schema_version": _REPORT_SCHEMA,
            "challenge_manifest_sha256": challenge_manifest["manifest_sha256"],
            "challenge_queue_sha256": challenge_manifest["queue_sha256"],
            "answer_key_sha256": challenge_manifest["answer_key_sha256"],
            "deterministic_audit_manifest_sha256": deterministic_manifest[
                "manifest_sha256"
            ],
            "raw_ast_audit_manifest_sha256": raw_manifest["manifest_sha256"],
            "raw_ast_witness_sidecar_sha256": raw_manifest[
                "witness_sidecar_sha256"
            ],
            "positive_control_count": len(positive_ids),
            "positive_false_flag_count": positive_false_flags,
            "negative_control_count": len(negative_ids),
            "negative_detected_count": negative_detected,
            "deterministic_flagged_count": len(deterministic_ids),
            "raw_ast_flagged_count": len(raw_flagged_ids),
            "combined_flagged_count": len(flagged_ids),
            "defect_metrics": dict(sorted(defect_metrics.items())),
            "audit_qualification_passed": qualified,
            "training_ready": False,
            "training_blockers": [
                "combined audit qualification alone does not authorize training"
            ],
        }
        return {**unsigned, "report_sha256": _sha256_text(_canonical(unsigned))}

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
    def _read_rows(
        path: Path, expected_sha256: object, name: str
    ) -> list[dict[str, object]]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"{name} cannot be read") from exc
        if _sha256_text(text) != expected_sha256:
            raise ValueError(f"{name} SHA-256 mismatch")
        rows: list[dict[str, object]] = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} JSON is invalid") from exc
            if type(row) is not dict or line != _canonical(row):
                raise ValueError(f"{name} row is not canonical")
            rows.append(row)
        return rows

    @staticmethod
    def _result(
        output: Path, report: dict[str, object]
    ) -> SemanticAuditQualificationResult:
        return SemanticAuditQualificationResult(
            output_root=output,
            audit_qualification_passed=bool(report["audit_qualification_passed"]),
            positive_control_count=int(report["positive_control_count"]),
            positive_false_flag_count=int(report["positive_false_flag_count"]),
            negative_control_count=int(report["negative_control_count"]),
            negative_detected_count=int(report["negative_detected_count"]),
            training_ready=bool(report["training_ready"]),
            report_sha256=str(report["report_sha256"]),
        )
