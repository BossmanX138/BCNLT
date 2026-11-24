from __future__ import annotations

import importlib
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest
from test_local_semantic_review import _decision, _review_server, _reviewers
from test_semantic_challenge_review import _reviewed_source

import bosscrafts_model.local_semantic_review as local_review
from bosscrafts_model.local_semantic_review import DualLocalSemanticReviewer
from bosscrafts_model.semantic_challenge_review import SemanticChallengeReviewBuilder


def _feature():
    if importlib.util.find_spec("bosscrafts_model.semantic_challenge_evaluation") is None:
        pytest.fail("semantic challenge qualification gate is not implemented")
    return importlib.import_module("bosscrafts_model.semantic_challenge_evaluation")


def test_qualification_requires_reviewer_to_identify_the_planted_defect():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as natural_review_dir,
        tempfile.TemporaryDirectory() as labels_dir,
        tempfile.TemporaryDirectory() as challenge_dir,
        tempfile.TemporaryDirectory() as reviewer_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _reviewed_source(
            Path(queue_dir), Path(natural_review_dir), Path(labels_dir)
        )
        SemanticChallengeReviewBuilder(pairs_per_language=5).materialize(
            queue_dir, labels_dir, challenge_dir
        )
        challenge_root = Path(challenge_dir)
        queue = [
            json.loads(line)
            for line in (challenge_root / "review_queue.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        answers = {
            row["review_id"]: row
            for row in (
                json.loads(line)
                for line in (challenge_root / "control_answer_key.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        right_label_wrong_reason = [
            _decision(answers[item["review_id"]]["expected_decision"])
            for item in queue
        ]
        misses_every_defect = [_decision("keep") for _item in queue]
        with _review_server(
            {"model-a": right_label_wrong_reason, "model-b": misses_every_defect}
        ) as (base_url, _state):
            DualLocalSemanticReviewer(
                challenge_dir,
                reviewer_dir,
                _reviewers(local_review, base_url),
                review_protocol="contract-aware-v3",
            ).run()

        policy = feature.SemanticChallengeQualificationPolicy(
            minimum_terminal_items=10,
            minimum_positive_controls=5,
            minimum_negative_controls=5,
            minimum_accuracy=1.0,
            minimum_negative_sensitivity=1.0,
            minimum_positive_specificity=1.0,
            require_no_exhausted_failures=True,
        )
        result = feature.SemanticChallengeEvaluator(policy).materialize(
            challenge_dir, reviewer_dir, output_dir
        )
        report = json.loads(
            (Path(output_dir) / "qualification.json").read_text(encoding="utf-8")
        )
        metrics = {
            row["reviewer_id"]: row for row in report["reviewer_metrics"]
        }

        assert result.qualification_passed is False
        assert metrics["reviewer-a"]["correct_items"] == 7
        assert metrics["reviewer-a"]["negative_sensitivity"] == 0.4
        assert metrics["reviewer-a"]["positive_specificity"] == 1.0
        assert metrics["reviewer-a"]["qualification_passed"] is False
        assert metrics["reviewer-b"]["correct_items"] == 5
        assert metrics["reviewer-b"]["negative_sensitivity"] == 0.0
        assert metrics["reviewer-b"]["positive_specificity"] == 1.0
        assert metrics["reviewer-b"]["qualification_passed"] is False
        assert "one or more reviewers failed qualification" in report["blockers"]
        assert report["training_ready"] is False


def test_verifier_rejects_a_modified_qualification_report():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as natural_review_dir,
        tempfile.TemporaryDirectory() as labels_dir,
        tempfile.TemporaryDirectory() as challenge_dir,
        tempfile.TemporaryDirectory() as reviewer_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _reviewed_source(
            Path(queue_dir), Path(natural_review_dir), Path(labels_dir)
        )
        SemanticChallengeReviewBuilder(pairs_per_language=5).materialize(
            queue_dir, labels_dir, challenge_dir
        )
        challenge_root = Path(challenge_dir)
        queue = [
            json.loads(line)
            for line in (challenge_root / "review_queue.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        answers = {
            row["review_id"]: row
            for row in (
                json.loads(line)
                for line in (challenge_root / "control_answer_key.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        perfect = [
            _decision(answers[item["review_id"]]["expected_decision"])
            for item in queue
        ]
        with _review_server({"model-a": perfect, "model-b": perfect}) as (
            base_url,
            _state,
        ):
            DualLocalSemanticReviewer(
                challenge_dir,
                reviewer_dir,
                _reviewers(local_review, base_url),
            ).run()
        evaluator = feature.SemanticChallengeEvaluator()
        evaluator.materialize(challenge_dir, reviewer_dir, output_dir)
        report_path = Path(output_dir) / "qualification.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["qualification_passed"] = not report["qualification_passed"]
        report_path.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        with pytest.raises(ValueError, match="stale or tampered"):
            evaluator.verify(challenge_dir, reviewer_dir, output_dir)

