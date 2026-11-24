from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bosscrafts_model.gate_category_case_sources import freeze_gate_category_cases


def test_freezes_source_controlled_regression_and_resource_cases(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    policy = project_root / "benchmarks" / "bosscrafts-acceptance-policy-v1.json"
    source_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
    regression = tmp_path / "regression.jsonl"
    resources = tmp_path / "resources.jsonl"

    result = freeze_gate_category_cases(
        policy,
        regression,
        resources,
        revision="a" * 40,
        expected_source_sha256=source_sha256,
    )

    regression_record = json.loads(regression.read_text(encoding="utf-8"))
    resource_record = json.loads(resources.read_text(encoding="utf-8"))
    assert regression_record["case"] == {
        "suite_id": "bosscrafts-pre-model-full-regression-v1",
        "minimum_test_count": 631,
        "minimum_subtest_count": 195,
    }
    assert resource_record["case"] == {
        "vram_limit_bytes": 12 * 1024**3,
        "ram_limit_bytes": 64 * 1024**3,
        "minimum_tokens_per_second": 1.0,
    }
    assert result.policy_sha256 == source_sha256
    assert result.regression_example_id == regression_record["example_id"]
    assert result.resources_example_id == resource_record["example_id"]
    with pytest.raises(ValueError, match="already exists"):
        freeze_gate_category_cases(
            policy,
            regression,
            resources,
            revision="a" * 40,
            expected_source_sha256=source_sha256,
        )


def test_gate_policy_digest_and_revision_are_mandatory(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    policy = project_root / "benchmarks" / "bosscrafts-acceptance-policy-v1.json"
    with pytest.raises(ValueError, match="SHA-256"):
        freeze_gate_category_cases(
            policy,
            tmp_path / "regression.jsonl",
            tmp_path / "resources.jsonl",
            revision="a" * 40,
            expected_source_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="revision"):
        freeze_gate_category_cases(
            policy,
            tmp_path / "regression.jsonl",
            tmp_path / "resources.jsonl",
            revision="main",
            expected_source_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        )
