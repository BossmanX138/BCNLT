"""Deterministic pre-audit for objective BCNLT semantic-record invariants."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .local_semantic_review import DualLocalSemanticReviewer, _SEMANTIC_CONTRACT


_SOURCE_MANIFEST_SCHEMA = "bosscrafts-heldout-semantic-review-manifest-v2"
_SOURCE_ITEM_SCHEMA = "bosscrafts-heldout-semantic-review-item-v2"
_FINDING_SCHEMA = "bosscrafts-deterministic-semantic-finding-v1"
_REPORT_SCHEMA = "bosscrafts-deterministic-semantic-audit-manifest-v1"


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
class DeterministicSemanticFinding:
    code: str
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class DeterministicSemanticAuditResult:
    review_id: str
    passed: bool
    extracted_surface: str
    findings: tuple[DeterministicSemanticFinding, ...]


@dataclass(frozen=True, slots=True)
class DeterministicSemanticAuditReportResult:
    output_root: Path
    item_count: int
    failed_item_count: int
    finding_count: int
    training_ready: bool
    findings_sha256: str
    manifest_sha256: str


class DeterministicSemanticAuditor:
    """Reject machine-verifiable defects before any model judgment is used."""

    def audit_item(self, item: object) -> DeterministicSemanticAuditResult:
        if type(item) is not dict:
            raise TypeError("semantic review item must be an object")
        review_id = item.get("review_id")
        record = item.get("canonical_record")
        if type(review_id) is not str or not review_id or type(record) is not dict:
            raise ValueError("semantic review item identity or record is invalid")
        event = record.get("event")
        surface = record.get("surface")
        span = record.get("span")
        scope = record.get("scope")
        ir = record.get("ir")
        metadata = record.get("metadata")
        if (
            type(event) is not str
            or type(surface) is not str
            or type(span) is not list
            or len(span) != 4
            or any(type(value) is not int or value < 0 for value in span)
            or type(scope) is not str
            or type(ir) is not dict
            or type(metadata) is not dict
        ):
            raise ValueError("semantic review canonical record is invalid")
        evidence = DualLocalSemanticReviewer._span_evidence(item)
        extracted = str(evidence["extracted_surface"])
        findings: list[DeterministicSemanticFinding] = []
        boundary_events = set(_SEMANTIC_CONTRACT["zero_width_boundary_events"])
        is_zero_width = span[0] == span[2] and span[1] == span[3]
        if event in boundary_events:
            if not is_zero_width:
                findings.append(
                    DeterministicSemanticFinding(
                        "scope_boundary_span_not_zero",
                        "span",
                        "scope boundary events must use a zero-width point span",
                    )
                )
            source_excerpt = item.get("source_excerpt")
            ast_relation = record.get("ast_relation")
            target_source = "\n".join(
                line.split("| ", 1)[1]
                for line in str(source_excerpt).splitlines()
                if line.startswith(">") and "| " in line
            )
            node_kind = (
                ast_relation[: -len(".scope")]
                if type(ast_relation) is str and ast_relation.endswith(".scope")
                else ""
            )
            scope_tail = scope.rsplit(".", 1)[-1]
            surface_is_anchored = bool(surface) and (
                surface == node_kind
                or surface in target_source
                or (scope != "module" and (surface in scope or scope_tail in surface))
            )
            if not surface_is_anchored:
                findings.append(
                    DeterministicSemanticFinding(
                        "scope_boundary_surface_unanchored",
                        "surface",
                        "scope boundary surface must name the target source, scope, or structural node",
                    )
                )
        elif surface != extracted:
            findings.append(
                DeterministicSemanticFinding(
                    "surface_span_mismatch",
                    "surface",
                    "canonical surface differs from the exact source span",
                )
            )

        allowed_ir = _SEMANTIC_CONTRACT["event_ir_kinds"].get(event)
        if type(allowed_ir) is not list or ir.get("kind") not in allowed_ir:
            findings.append(
                DeterministicSemanticFinding(
                    "event_ir_contract_mismatch",
                    "ir.kind",
                    "event and IR kind violate the pinned Common Coding Core contract",
                )
            )
        ir_name = ir.get("name")
        if (
            ir.get("kind") != "extension"
            and ir_name is not None
            and ir_name != surface
        ):
            findings.append(
                DeterministicSemanticFinding(
                    "ir_name_surface_mismatch",
                    "ir.name",
                    "non-extension IR names must match the claimed semantic surface",
                )
            )
        if scope != "module" and not scope.startswith("module."):
            findings.append(
                DeterministicSemanticFinding(
                    "scope_root_mismatch",
                    "scope",
                    "scope path must be rooted at module",
                )
            )
        if event == "UNKNOWN_SEMANTIC" and (
            metadata.get("knowledge_state") == "known"
            or metadata.get("uncertainty_signals") == []
            or metadata.get("uyr_rounds") == 0
            or record.get("confidence_band") == "high"
            or record.get("uncertainty_band") == "low"
        ):
            findings.append(
                DeterministicSemanticFinding(
                    "unknown_semantic_uncertainty_inconsistent",
                    "metadata",
                    "UNKNOWN_SEMANTIC must preserve an honest unresolved state",
                )
            )
        return DeterministicSemanticAuditResult(
            review_id=review_id,
            passed=not findings,
            extracted_surface=extracted,
            findings=tuple(findings),
        )


class DeterministicSemanticAuditReport:
    """Freeze and verify objective pre-audit evidence for a semantic queue."""

    def materialize(
        self, source_root: str | Path, output_root: str | Path
    ) -> DeterministicSemanticAuditReportResult:
        output = Path(output_root).resolve()
        findings_text, manifest = self._evaluate(Path(source_root).resolve())
        _atomic_write(output / "findings.jsonl", findings_text.encode("utf-8"))
        _atomic_write(
            output / "manifest.json", f"{_canonical(manifest)}\n".encode("utf-8")
        )
        return self._result(output, manifest)

    def verify(
        self, source_root: str | Path, output_root: str | Path
    ) -> DeterministicSemanticAuditReportResult:
        output = Path(output_root).resolve()
        expected_findings, expected_manifest = self._evaluate(
            Path(source_root).resolve()
        )
        try:
            actual_findings = (output / "findings.jsonl").read_text(encoding="utf-8")
            actual_manifest = (output / "manifest.json").read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                "deterministic semantic audit evidence is stale or tampered"
            ) from exc
        if (
            actual_findings != expected_findings
            or actual_manifest != f"{_canonical(expected_manifest)}\n"
        ):
            raise ValueError(
                "deterministic semantic audit evidence is stale or tampered"
            )
        return self._result(output, expected_manifest)

    def _evaluate(self, source_root: Path) -> tuple[str, dict[str, object]]:
        manifest = self._load_source_manifest(source_root / "manifest.json")
        try:
            queue_text = (
                source_root / str(manifest.get("queue_file"))
            ).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError("semantic source queue cannot be read") from exc
        if _sha256_text(queue_text) != manifest.get("queue_sha256"):
            raise ValueError("semantic source queue SHA-256 mismatch")
        rows: list[dict[str, object]] = []
        for line in queue_text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("semantic source queue JSON is invalid") from exc
            if type(row) is not dict or line != _canonical(row):
                raise ValueError("semantic source queue row is not canonical")
            rows.append(row)
        review_ids: set[str] = set()
        parser_identity = manifest.get("parser_runtime_identity_sha256")
        for row in rows:
            review_id = row.get("review_id")
            if (
                row.get("schema_version") != _SOURCE_ITEM_SCHEMA
                or type(review_id) is not str
                or review_id in review_ids
                or row.get("parser_runtime_identity_sha256") != parser_identity
                or type(row.get("language")) is not str
                or not row.get("language")
            ):
                raise ValueError("semantic source queue item is invalid")
            review_ids.add(review_id)
        if len(rows) != manifest.get("item_count"):
            raise ValueError("semantic source queue item count mismatch")
        represented = sorted({str(row["language"]) for row in rows})
        if represented != manifest.get("represented_languages"):
            raise ValueError("semantic source queue language inventory mismatch")

        auditor = DeterministicSemanticAuditor()
        finding_rows: list[dict[str, object]] = []
        failed_ids: set[str] = set()
        counts_by_code: dict[str, int] = {}
        counts_by_language: dict[str, int] = {}
        for row in rows:
            result = auditor.audit_item(row)
            if result.passed:
                continue
            review_id = str(row["review_id"])
            language = str(row["language"])
            failed_ids.add(review_id)
            counts_by_language[language] = counts_by_language.get(language, 0) + 1
            for finding in result.findings:
                counts_by_code[finding.code] = counts_by_code.get(finding.code, 0) + 1
                finding_rows.append(
                    {
                        "schema_version": _FINDING_SCHEMA,
                        "review_id": review_id,
                        "language": language,
                        "code": finding.code,
                        "field": finding.field,
                        "message": finding.message,
                        "extracted_surface": result.extracted_surface,
                        "parser_runtime_identity_sha256": parser_identity,
                    }
                )
        findings_text = "".join(f"{_canonical(row)}\n" for row in finding_rows)
        findings_sha256 = _sha256_text(findings_text)
        audit_passed = not failed_ids
        blockers = ["deterministic semantic audit alone does not authorize training"]
        if not audit_passed:
            blockers.insert(0, "deterministic semantic audit found objective defects")
        unsigned: dict[str, object] = {
            "schema_version": _REPORT_SCHEMA,
            "source_review_manifest_sha256": manifest["manifest_sha256"],
            "source_queue_sha256": manifest["queue_sha256"],
            "parser_runtime_identity_sha256": parser_identity,
            "item_count": len(rows),
            "passed_item_count": len(rows) - len(failed_ids),
            "failed_item_count": len(failed_ids),
            "finding_count": len(finding_rows),
            "counts_by_code": dict(sorted(counts_by_code.items())),
            "counts_by_language": dict(sorted(counts_by_language.items())),
            "findings_file": "findings.jsonl",
            "findings_sha256": findings_sha256,
            "deterministic_audit_passed": audit_passed,
            "training_ready": False,
            "training_blockers": blockers,
        }
        return findings_text, {
            **unsigned,
            "manifest_sha256": _sha256_text(_canonical(unsigned)),
        }

    @staticmethod
    def _load_source_manifest(path: Path) -> dict[str, object]:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("semantic source manifest is invalid") from exc
        if type(manifest) is not dict or manifest.get("schema_version") != _SOURCE_MANIFEST_SCHEMA:
            raise ValueError("semantic source manifest schema mismatch")
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if claimed != _sha256_text(_canonical(unsigned)):
            raise ValueError("semantic source manifest SHA-256 mismatch")
        return manifest

    @staticmethod
    def _result(
        output: Path, manifest: dict[str, object]
    ) -> DeterministicSemanticAuditReportResult:
        return DeterministicSemanticAuditReportResult(
            output_root=output,
            item_count=int(manifest["item_count"]),
            failed_item_count=int(manifest["failed_item_count"]),
            finding_count=int(manifest["finding_count"]),
            training_ready=bool(manifest["training_ready"]),
            findings_sha256=str(manifest["findings_sha256"]),
            manifest_sha256=str(manifest["manifest_sha256"]),
        )

