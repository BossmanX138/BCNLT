from __future__ import annotations

import importlib
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest
from test_semantic_challenge_review import _reviewed_source

from bosscrafts_model.semantic_challenge_review import SemanticChallengeReviewBuilder


def _feature():
    if importlib.util.find_spec("bosscrafts_model.semantic_challenge_canary") is None:
        pytest.fail("balanced semantic challenge canary selector is not implemented")
    return importlib.import_module("bosscrafts_model.semantic_challenge_canary")


def test_selects_five_blinded_pairs_covering_every_defect_type():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as natural_review_dir,
        tempfile.TemporaryDirectory() as labels_dir,
        tempfile.TemporaryDirectory() as challenge_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _reviewed_source(
            Path(queue_dir), Path(natural_review_dir), Path(labels_dir)
        )
        SemanticChallengeReviewBuilder(pairs_per_language=5).materialize(
            queue_dir, labels_dir, challenge_dir
        )
        result = feature.SemanticChallengeCanarySelector().materialize(
            challenge_dir, output_dir
        )
        verified = feature.SemanticChallengeCanarySelector().verify(
            challenge_dir, output_dir
        )
        selection = json.loads(
            (Path(output_dir) / "selection.json").read_text(encoding="utf-8")
        )
        answers = {
            row["review_id"]: row
            for row in (
                json.loads(line)
                for line in (Path(challenge_dir) / "control_answer_key.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        selected_answers = [
            answers[review_id] for review_id in selection["selected_review_ids"]
        ]
        queue = {
            row["review_id"]: row
            for row in (
                json.loads(line)
                for line in (Path(challenge_dir) / "review_queue.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }

        assert result.item_count == verified.item_count == 10
        assert len(set(selection["selected_review_ids"])) == 10
        assert sum(row["expected_decision"] == "keep" for row in selected_answers) == 5
        assert sum(row["expected_decision"] == "reject" for row in selected_answers) == 5
        assert {
            row["defect_type"]
            for row in selected_answers
            if row["expected_decision"] == "reject"
        } == {
            "event_ir_mismatch",
            "scope_mismatch",
            "span_mismatch",
            "surface_mismatch",
            "uncertainty_understated",
        }
        assert len({row["pair_identity_sha256"] for row in selected_answers}) == 5
        scope_negative = next(
            row for row in selected_answers if row["defect_type"] == "scope_mismatch"
        )
        scope_positive = next(
            row
            for row in selected_answers
            if row["pair_identity_sha256"]
            == scope_negative["pair_identity_sha256"]
            and row["expected_decision"] == "keep"
        )
        positive_item = queue[scope_positive["review_id"]]
        scope_parts = [
            part
            for part in positive_item["canonical_record"]["scope"].split(".")
            if part not in {"module", "compound_statement", "block"}
        ]
        assert any(part in positive_item["source_excerpt"] for part in scope_parts)
        assert "expected_decision" not in selection
        assert "defect_type" not in selection

