from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_local_semantic_review import (  # noqa: E402
    _decision,
    _queue,
    _review_server,
    _reviewers,
)

import bosscrafts_model.local_semantic_review as local_review
from bosscrafts_model.local_semantic_review import DualLocalSemanticReviewer


def _feature():
    if importlib.util.find_spec("bosscrafts_model.semantic_calibration_dataset") is None:
        pytest.fail("semantic calibration dataset feature is not implemented")
    return importlib.import_module("bosscrafts_model.semantic_calibration_dataset")


def test_materializes_disjoint_balanced_observations_from_verified_consensus():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as dual_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _queue(Path(queue_dir), item_count=4)
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")}
        ) as (base_url, _state):
            DualLocalSemanticReviewer(
                queue_dir, dual_dir, _reviewers(local_review, base_url)
            ).run()

        result = feature.SemanticCalibrationDatasetBuilder(
            calibration_per_language=2,
            heldout_per_language=2,
        ).materialize(queue_dir, dual_dir, (), output_dir)

        calibration = [
            json.loads(line)
            for line in (Path(output_dir) / "calibration_observations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        heldout = [
            json.loads(line)
            for line in (Path(output_dir) / "heldout_observations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert result.calibration_count == 2
        assert result.heldout_count == 2
        assert {row["sample_id"] for row in calibration}.isdisjoint(
            row["sample_id"] for row in heldout
        )
        assert all(row["language"] == "python" for row in calibration + heldout)
        assert all(row["raw_uncertainty"] == 0.05 for row in calibration + heldout)
        assert all(row["incorrect"] is False for row in calibration + heldout)
        manifest = json.loads((Path(output_dir) / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["calibration_ready"] is False
        assert "calibration_split_missing_reject:calibration" in manifest[
            "calibration_blockers"
        ]
        assert "language_split_missing_reject:python:heldout" in manifest[
            "calibration_blockers"
        ]
        assert manifest["training_ready"] is False
        assert manifest["language_counts"] == {"python": {"calibration": 2, "heldout": 2}}


def test_each_language_and_split_needs_both_label_classes():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as dual_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _queue(Path(queue_dir), item_count=8)
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("reject")}
        ) as (base_url, _state):
            DualLocalSemanticReviewer(
                queue_dir, dual_dir, _reviewers(local_review, base_url)
            ).run()
        review_ids = [
            json.loads(line)["review_id"]
            for line in (Path(queue_dir) / "review_queue.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        decisions = tuple(
            {
                "review_id": review_id,
                "decision": "reject" if index % 2 == 0 else "keep",
                "reviewer": "manual-fixture",
                "notes": "Checked exact source and semantic record evidence.",
            }
            for index, review_id in enumerate(review_ids)
        )
        feature.SemanticCalibrationDatasetBuilder(
            calibration_per_language=4,
            heldout_per_language=4,
        ).materialize(queue_dir, dual_dir, decisions, output_dir)

        manifest = json.loads(
            (Path(output_dir) / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["calibration_ready"] is True
        assert manifest["calibration_blockers"] == []
        assert manifest["label_counts"]["per_language"]["python"] == {
            "calibration": {"keep": 2, "reject": 2},
            "heldout": {"keep": 2, "reject": 2},
        }


def test_manual_disagreement_requires_exact_keep_or_reject_resolution():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as dual_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _queue(Path(queue_dir), item_count=2)
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("reject")}
        ) as (base_url, _state):
            DualLocalSemanticReviewer(
                queue_dir, dual_dir, _reviewers(local_review, base_url)
            ).run()

        builder = feature.SemanticCalibrationDatasetBuilder(
            calibration_per_language=1,
            heldout_per_language=1,
        )
        with pytest.raises(ValueError, match="missing manual decisions"):
            builder.materialize(queue_dir, dual_dir, (), output_dir)

        review_ids = [
            json.loads(line)["review_id"]
            for line in (Path(queue_dir) / "review_queue.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        decisions = tuple(
            {
                "review_id": review_id,
                "decision": "reject" if index == 0 else "keep",
                "reviewer": "manual-fixture",
                "notes": "Checked the marked span against the semantic record.",
            }
            for index, review_id in enumerate(review_ids)
        )
        result = builder.materialize(queue_dir, dual_dir, decisions, output_dir)
        assert result.calibration_count == 1
        assert result.heldout_count == 1
        rows = [
            json.loads(line)
            for name in ("calibration_observations.jsonl", "heldout_observations.jsonl")
            for line in (Path(output_dir) / name).read_text(encoding="utf-8").splitlines()
        ]
        assert {row["incorrect"] for row in rows} == {False, True}
        assert all(row["resolution"] == "manual" for row in rows)


def test_cli_materializes_verified_dataset_without_network_calls():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "build_semantic_calibration_dataset.py"
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as dual_dir,
        tempfile.TemporaryDirectory() as output_dir,
        tempfile.TemporaryDirectory() as decisions_dir,
    ):
        decisions_path = Path(decisions_dir) / "manual-decisions.jsonl"
        decisions_path.write_text("", encoding="utf-8")
        _queue(Path(queue_dir), item_count=2)
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")}
        ) as (base_url, _state):
            DualLocalSemanticReviewer(
                queue_dir, dual_dir, _reviewers(local_review, base_url)
            ).run()
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--queue",
                queue_dir,
                "--dual-review",
                dual_dir,
                "--manual-decisions",
                str(decisions_path),
                "--output",
                output_dir,
                "--calibration-per-language",
                "1",
                "--heldout-per-language",
                "1",
            ],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        assert report["calibration_count"] == 1
        assert report["heldout_count"] == 1
