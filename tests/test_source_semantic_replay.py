from __future__ import annotations

import importlib
import importlib.util
import json

import pytest

from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _feature():
    if importlib.util.find_spec("bosscrafts_model.source_semantic_replay") is None:
        pytest.fail("full-source semantic replay auditor is not implemented")
    return importlib.import_module("bosscrafts_model.source_semantic_replay")


def test_live_replay_rebuilds_the_selected_record_from_full_source(
    tmp_path,
    current_runtime_semantic_evidence,
):
    feature = _feature()
    output = tmp_path / "replay"
    auditor = feature.SourceSemanticReplayAuditor(NativeTokenizerBinding.load())

    result = auditor.materialize(
        current_runtime_semantic_evidence.admission_root,
        current_runtime_semantic_evidence.review_root,
        output,
        review_ids=(current_runtime_semantic_evidence.selected_review_id,),
    )

    assert result.item_count == 1
    assert result.document_count == 1
    assert result.matched_item_count == 1
    assert result.mismatched_item_count == 0
    assert result.complete_source_queue is False
    assert result.training_ready is False
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_review_item_count"] == 2
    assert manifest["replayed_item_count"] == 1
    assert manifest["replayed_document_count"] == 1
    assert manifest["record_match_count"] == 1
    assert manifest["record_mismatch_count"] == 0
    assert manifest["document_stream_mismatch_count"] == 0
    assert manifest["semantic_replay_passed"] is True
    assert manifest["complete_source_queue"] is False
    assert manifest["training_ready"] is False
    assert auditor.verify(
        current_runtime_semantic_evidence.admission_root,
        current_runtime_semantic_evidence.review_root,
        output,
        review_ids=(current_runtime_semantic_evidence.selected_review_id,),
    ) == result


def test_live_replay_rejects_preserved_stale_runtime_evidence(
    tmp_path,
    stale_runtime_semantic_evidence,
):
    auditor = _feature().SourceSemanticReplayAuditor(NativeTokenizerBinding.load())

    with pytest.raises(ValueError, match="lineage mismatch"):
        auditor.materialize(
            stale_runtime_semantic_evidence.admission_root,
            stale_runtime_semantic_evidence.review_root,
            tmp_path / "stale-replay",
            review_ids=(stale_runtime_semantic_evidence.selected_review_id,),
        )
