from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEWS = PROJECT_ROOT / "artifacts" / "reviews"
CHALLENGE = REVIEWS / "bcnlt-v10-semantic-challenge-v1"
DETERMINISTIC = REVIEWS / "bcnlt-v10-semantic-challenge-deterministic-audit-v1"
RAW_AST = REVIEWS / "bcnlt-v10-semantic-challenge-raw-ast-audit-v1"


def _feature():
    if importlib.util.find_spec("bosscrafts_model.semantic_audit_qualification") is None:
        pytest.fail("combined semantic audit qualification is not implemented")
    return importlib.import_module("bosscrafts_model.semantic_audit_qualification")


def test_combined_audits_qualify_on_every_blinded_defect_without_false_flags(
    tmp_path,
):
    feature = _feature()
    output = tmp_path / "qualification"
    evaluator = feature.SemanticAuditQualificationEvaluator()

    result = evaluator.materialize(
        CHALLENGE,
        DETERMINISTIC,
        RAW_AST,
        output,
    )

    assert result.audit_qualification_passed is True
    assert result.positive_control_count == 140
    assert result.positive_false_flag_count == 0
    assert result.negative_control_count == 140
    assert result.negative_detected_count == 140
    assert result.training_ready is False
    report = json.loads((output / "qualification.json").read_text(encoding="utf-8"))
    assert report["defect_metrics"] == {
        "event_ir_mismatch": {"detected": 28, "total": 28},
        "scope_mismatch": {"detected": 28, "total": 28},
        "span_mismatch": {"detected": 28, "total": 28},
        "surface_mismatch": {"detected": 42, "total": 42},
        "uncertainty_understated": {"detected": 14, "total": 14},
    }
    assert report["audit_qualification_passed"] is True
    assert report["training_ready"] is False
    assert evaluator.verify(
        CHALLENGE,
        DETERMINISTIC,
        RAW_AST,
        output,
    ) == result
