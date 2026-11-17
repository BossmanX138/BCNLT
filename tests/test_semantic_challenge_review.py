from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest
from test_local_semantic_review import (
    _canonical,
    _decision,
    _queue,
    _review_server,
    _reviewers,
)

import bosscrafts_model.local_semantic_review as local_review
from bosscrafts_model.local_semantic_review import DualLocalSemanticReviewer
from bosscrafts_model.semantic_calibration_dataset import (
    SemanticCalibrationDatasetBuilder,
)


def _feature():
    if importlib.util.find_spec("bosscrafts_model.semantic_challenge_review") is None:
        pytest.fail("semantic challenge review feature is not implemented")
    return importlib.import_module("bosscrafts_model.semantic_challenge_review")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reviewed_source(queue_dir: Path, dual_dir: Path, labels_dir: Path) -> None:
    _queue(queue_dir, item_count=10)
    queue_path = queue_dir / "review_queue.jsonl"
    rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows):
        record = row["canonical_record"]
        record["surface"] = f"value_{index}"
        record["scope"] = "module.value"
        record["metadata"] = {
            "knowledge_state": "known",
            "uncertainty_signals": [],
            "uyr_rounds": 0,
        }
        record["confidence_band"] = "high"
        record["uncertainty_band"] = "low"
        if index in {4, 9}:
            record["event"] = "UNKNOWN_SEMANTIC"
            record["ir"] = {
                "kind": "extension",
                "name": "python.fixture_unknown",
                "value": None,
                "children": [],
                "metadata": {"node_type": "fixture_unknown"},
            }
            record["metadata"] = {
                "knowledge_state": "provisional",
                "uncertainty_signals": ["unknown semantic"],
                "uyr_rounds": 2,
            }
            record["confidence_band"] = "low"
            record["uncertainty_band"] = "high"
    queue_text = "".join(f"{_canonical(row)}\n" for row in rows)
    queue_path.write_text(queue_text, encoding="utf-8", newline="\n")
    manifest_path = queue_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    manifest["queue_sha256"] = _sha(queue_text)
    manifest["manifest_sha256"] = _sha(_canonical(manifest))
    manifest_path.write_text(
        f"{_canonical(manifest)}\n", encoding="utf-8", newline="\n"
    )
    with _review_server(
        {"model-a": _decision("keep"), "model-b": _decision("keep")}
    ) as (base_url, _state):
        DualLocalSemanticReviewer(
            queue_dir, dual_dir, _reviewers(local_review, base_url)
        ).run()
    SemanticCalibrationDatasetBuilder(
        calibration_per_language=5,
        heldout_per_language=5,
    ).materialize(queue_dir, dual_dir, (), labels_dir)


def test_builds_blinded_paired_controls_with_separate_hashed_answer_key():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as dual_dir,
        tempfile.TemporaryDirectory() as labels_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _reviewed_source(Path(queue_dir), Path(dual_dir), Path(labels_dir))
        result = feature.SemanticChallengeReviewBuilder(
            pairs_per_language=5
        ).materialize(queue_dir, labels_dir, output_dir)
        verified = feature.SemanticChallengeReviewBuilder(
            pairs_per_language=5
        ).verify(output_dir)

        root = Path(output_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        queue = [
            json.loads(line)
            for line in (root / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        answers = [
            json.loads(line)
            for line in (root / "control_answer_key.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert result.item_count == verified.item_count == 10
        assert manifest["expected_counts"] == {"keep": 5, "reject": 5}
        assert {row["expected_decision"] for row in answers} == {"keep", "reject"}
        assert {
            row["defect_type"] for row in answers if row["expected_decision"] == "reject"
        } == {
            "event_ir_mismatch",
            "scope_mismatch",
            "span_mismatch",
            "surface_mismatch",
            "uncertainty_understated",
        }
        assert all(
            not ({"expected_decision", "defect_type", "control_kind"} & set(row))
            for row in queue
        )
        assert manifest["answer_key_sha256"] == hashlib.sha256(
            (root / "control_answer_key.jsonl").read_bytes()
        ).hexdigest()
        assert manifest["training_ready"] is False


def test_challenge_verifier_rejects_queue_tampering():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as dual_dir,
        tempfile.TemporaryDirectory() as labels_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _reviewed_source(Path(queue_dir), Path(dual_dir), Path(labels_dir))
        builder = feature.SemanticChallengeReviewBuilder(pairs_per_language=5)
        builder.materialize(queue_dir, labels_dir, output_dir)
        with (Path(output_dir) / "review_queue.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with pytest.raises(ValueError, match="challenge queue SHA-256"):
            builder.verify(output_dir)
