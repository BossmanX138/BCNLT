from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from bosscrafts_model.heldout_semantic_review import HeldoutSemanticReviewQueue


def _feature():
    if importlib.util.find_spec("bosscrafts_model.deterministic_semantic_audit") is None:
        pytest.fail("deterministic semantic pre-auditor is not implemented")
    return importlib.import_module("bosscrafts_model.deterministic_semantic_audit")


def _item(
    *,
    review_id="1" * 64,
    language="python",
    event="REFERENCE",
    surface="value",
    span=None,
    ir_kind="variable",
):
    return {
        "schema_version": "bosscrafts-heldout-semantic-review-item-v2",
        "review_id": review_id,
        "language": language,
        "parser_runtime_identity_sha256": "a" * 64,
        "source_excerpt": (
            "[source context lines 1-1; target span 1:0-1:5]\n"
            ">000001| value = 1\n"
        ),
        "canonical_record": {
            "event": event,
            "surface": surface,
            "span": span or [1, 0, 1, 5],
            "scope": "module",
            "ir": {
                "kind": ir_kind,
                "name": surface,
                "value": None,
                "children": [],
                "metadata": {},
            },
            "metadata": {
                "knowledge_state": "known",
                "uncertainty_signals": [],
                "uyr_rounds": 0,
            },
            "confidence_band": "high",
            "uncertainty_band": "low",
        },
    }


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_source_queue(root: Path):
    rows = [
        _item(review_id="1" * 64),
        _item(review_id="2" * 64, language="javascript", surface="other"),
    ]
    queue_text = "".join(f"{_canonical(row)}\n" for row in rows)
    unsigned = {
        "schema_version": "bosscrafts-heldout-semantic-review-manifest-v2",
        "item_count": 2,
        "represented_languages": ["javascript", "python"],
        "queue_file": "review_queue.jsonl",
        "queue_sha256": _sha256(queue_text),
        "parser_runtime_identity_sha256": "a" * 64,
        "training_ready": False,
    }
    manifest = {**unsigned, "manifest_sha256": _sha256(_canonical(unsigned))}
    root.mkdir(parents=True)
    (root / "review_queue.jsonl").write_text(
        queue_text, encoding="utf-8", newline="\n"
    )
    (root / "manifest.json").write_text(
        f"{_canonical(manifest)}\n", encoding="utf-8", newline="\n"
    )
    return manifest, queue_text


def test_objective_audit_passes_an_exact_common_core_record():
    result = _feature().DeterministicSemanticAuditor().audit_item(_item())
    assert result.passed is True
    assert result.findings == ()
    assert result.extracted_surface == "value"


def test_objective_audit_finds_surface_contract_and_uncertainty_defects():
    feature = _feature()
    auditor = feature.DeterministicSemanticAuditor()

    surface = auditor.audit_item(_item(surface="other"))
    contract = auditor.audit_item(_item(ir_kind="call"))
    unknown_item = _item(event="UNKNOWN_SEMANTIC", ir_kind="extension")
    unknown_item["canonical_record"]["ir"]["name"] = "python.identifier"
    unknown_item["canonical_record"]["metadata"]["unknown_node"] = "identifier"
    unknown = auditor.audit_item(unknown_item)

    assert {finding.code for finding in surface.findings} == {
        "surface_span_mismatch"
    }
    assert {finding.code for finding in contract.findings} == {
        "event_ir_contract_mismatch"
    }
    assert {finding.code for finding in unknown.findings} == {
        "unknown_semantic_uncertainty_inconsistent"
    }


def test_non_extension_ir_name_must_match_its_claimed_surface():
    item = _item()
    item["canonical_record"]["ir"]["name"] = "different_value"

    result = _feature().DeterministicSemanticAuditor().audit_item(item)

    assert {finding.code for finding in result.findings} == {
        "ir_name_surface_mismatch"
    }


def test_truncated_surface_must_be_an_exact_bounded_prefix():
    exact = "x" * 4100
    item = _item(surface="x" * 4096, span=[1, 0, 1, 4100])
    item["source_excerpt"] = (
        "[source context lines 1-1; target span 1:0-1:4100]\n"
        f">000001| {exact}\n"
    )
    item["canonical_record"]["metadata"]["surface_truncated"] = True
    item["canonical_record"]["ir"]["name"] = exact

    result = _feature().DeterministicSemanticAuditor().audit_item(item)

    assert result.passed is True
    assert result.extracted_surface == exact


def test_truncated_surface_rejects_a_nonprefix_claim():
    exact = "x" * 4100
    item = _item(surface="y" * 4096, span=[1, 0, 1, 4100])
    item["source_excerpt"] = (
        "[source context lines 1-1; target span 1:0-1:4100]\n"
        f">000001| {exact}\n"
    )
    item["canonical_record"]["metadata"]["surface_truncated"] = True
    item["canonical_record"]["ir"]["name"] = exact

    result = _feature().DeterministicSemanticAuditor().audit_item(item)

    assert {finding.code for finding in result.findings} == {
        "surface_span_mismatch",
        "ir_name_surface_mismatch",
    }


def test_span_evidence_preserves_and_canonicalizes_crlf_bytes():
    item = _item(
        event="UNKNOWN_SEMANTIC",
        surface="Logging\n",
        span=[1, 8, 1, 16],
        ir_kind="extension",
    )
    item["source_excerpt"] = HeldoutSemanticReviewQueue._review_source_context(
        "#region Logging\r\n",
        item["canonical_record"],
    )
    item["canonical_record"]["ir"]["name"] = "c_sharp.preproc_arg"
    item["canonical_record"]["metadata"] = {
        "knowledge_state": "provisional",
        "unknown_node": "preproc_arg",
        "uncertainty_signals": ["unknown semantic"],
        "uyr_rounds": 1,
    }
    item["canonical_record"]["confidence_band"] = "medium"
    item["canonical_record"]["uncertainty_band"] = "medium"

    result = _feature().DeterministicSemanticAuditor().audit_item(item)

    assert result.extracted_surface == "Logging\n"
    assert result.passed is True


def test_zero_width_scope_boundary_uses_structural_surface_by_contract():
    item = _item(
        event="SCOPE_EXIT",
        surface="compound_statement",
        span=[1, 5, 1, 5],
        ir_kind="extension",
    )
    item["canonical_record"]["ir"]["name"] = "python.scope_exit"
    item["canonical_record"]["scope"] = "module.fn.compound_statement"
    item["canonical_record"]["ast_relation"] = "compound_statement.scope"

    result = _feature().DeterministicSemanticAuditor().audit_item(item)

    assert result.passed is True
    assert result.extracted_surface == ""


def test_decorated_scope_entry_accepts_a_nonempty_ast_owned_name():
    item = _item(
        event="SCOPE_ENTER",
        surface="serialize",
        span=[1, 0, 1, 0],
        ir_kind="extension",
    )
    item["source_excerpt"] = (
        "[source context lines 1-2; target span 1:0-1:0]\n"
        ">000001| @Override\n"
        " 000002| public void serialize() {}\n"
    )
    item["canonical_record"]["ir"]["name"] = "java.scope_enter"
    item["canonical_record"]["scope"] = "module.Serializer"
    item["canonical_record"]["ast_relation"] = "method_declaration.scope"

    result = _feature().DeterministicSemanticAuditor().audit_item(item)

    assert result.passed is True


def test_scope_boundary_surface_must_anchor_to_target_source_scope_or_node_kind():
    item = _item(
        event="SCOPE_ENTER",
        surface="unrelated_name",
        span=[1, 0, 1, 0],
        ir_kind="extension",
    )
    item["canonical_record"]["ir"]["name"] = "python.scope_enter"
    item["canonical_record"]["ast_relation"] = "function_definition.scope"

    result = _feature().DeterministicSemanticAuditor().audit_item(item)

    assert {finding.code for finding in result.findings} == {
        "scope_boundary_surface_unanchored"
    }


def test_report_materializes_content_addressed_findings_without_authorizing_training(
    tmp_path,
):
    feature = _feature()
    source = tmp_path / "source"
    output = tmp_path / "output"
    source_manifest, source_queue = _write_source_queue(source)

    result = feature.DeterministicSemanticAuditReport().materialize(source, output)

    assert result.item_count == 2
    assert result.failed_item_count == 1
    assert result.finding_count == 1
    assert result.training_ready is False
    report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report["source_review_manifest_sha256"] == source_manifest["manifest_sha256"]
    assert report["source_queue_sha256"] == source_manifest["queue_sha256"]
    assert report["parser_runtime_identity_sha256"] == "a" * 64
    assert report["counts_by_code"] == {"surface_span_mismatch": 1}
    assert report["counts_by_language"] == {"javascript": 1}
    assert report["deterministic_audit_passed"] is False
    assert report["training_ready"] is False
    assert (source / "review_queue.jsonl").read_text(encoding="utf-8") == source_queue
    assert feature.DeterministicSemanticAuditReport().verify(source, output) == result


def test_report_verifier_rejects_tampered_or_stale_evidence(tmp_path):
    feature = _feature()
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source_queue(source)
    reporter = feature.DeterministicSemanticAuditReport()
    reporter.materialize(source, output)
    findings_path = output / "findings.jsonl"
    findings_path.write_text(
        findings_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="stale or tampered"):
        reporter.verify(source, output)

