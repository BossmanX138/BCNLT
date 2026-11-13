from __future__ import annotations

import importlib
import importlib.util
import json

import pytest

from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _feature():
    if importlib.util.find_spec("bosscrafts_model.raw_ast_semantic_audit") is None:
        pytest.fail("raw AST semantic audit report is not implemented")
    return importlib.import_module("bosscrafts_model.raw_ast_semantic_audit")


def test_raw_ast_report_freezes_source_relation_and_scope_witnesses(
    tmp_path,
    current_runtime_semantic_evidence,
):
    feature = _feature()
    output = tmp_path / "ast-audit"
    auditor = feature.RawAstSemanticAuditReport(NativeTokenizerBinding.load())

    result = auditor.materialize(
        current_runtime_semantic_evidence.admission_root,
        current_runtime_semantic_evidence.review_root,
        output,
        review_ids=(current_runtime_semantic_evidence.selected_review_id,),
    )

    assert result.item_count == 1
    assert result.failed_item_count == 0
    assert result.complete_source_queue is False
    assert result.training_ready is False
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_review_item_count"] == 2
    assert manifest["witness_item_count"] == 1
    assert manifest["failed_item_count"] == 0
    assert manifest["counts_by_code"] == {}
    assert manifest["raw_ast_audit_passed"] is True
    assert manifest["complete_source_queue"] is False
    assert manifest["training_ready"] is False
    assert auditor.verify(
        current_runtime_semantic_evidence.admission_root,
        current_runtime_semantic_evidence.review_root,
        output,
        review_ids=(current_runtime_semantic_evidence.selected_review_id,),
    ) == result


def test_raw_ast_report_rejects_preserved_stale_runtime_evidence(
    tmp_path,
    stale_runtime_semantic_evidence,
):
    auditor = _feature().RawAstSemanticAuditReport(NativeTokenizerBinding.load())

    with pytest.raises(ValueError, match="lineage mismatch"):
        auditor.materialize(
            stale_runtime_semantic_evidence.admission_root,
            stale_runtime_semantic_evidence.review_root,
            tmp_path / "stale-ast-audit",
            review_ids=(stale_runtime_semantic_evidence.selected_review_id,),
        )
