from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_full_model_acceptance_categories import (
    CATEGORIES,
    CHECKPOINT_ID,
    DATABASE_ID,
    HELDOUT_MANIFEST,
    LANGUAGES,
    _category,
    _evaluation,
    _inference,
    _records,
    verify_model_acceptance,
)

from bosscrafts_model.bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from bosscrafts_model.model_category_evidence import (
    checkpoint_sha256,
    compile_model_category_evidence,
    validate_strict_uncertainty_receipt,
)
from bosscrafts_model.strict_uncertainty_acceptance import (
    StrictUncertaintyAcceptanceReceipt,
)


def _strict_receipt() -> StrictUncertaintyAcceptanceReceipt:
    policy = {
        "minimum_global_selective_coverage_at_025": 0.5,
        "maximum_global_selective_error_rate_at_025": 0.1,
        "minimum_per_language_selective_coverage_at_025": 0.5,
        "maximum_per_language_selective_error_rate_at_025": 0.1,
    }
    metric = {
        "sample_count": 1,
        "brier": 0.1,
        "ece": 0.1,
        "selective_coverage_at_025": 1.0,
        "selective_error_rate_at_025": 0.0,
    }
    provisional = StrictUncertaintyAcceptanceReceipt(
        schema_version="bosscrafts-strict-uncertainty-acceptance-v1",
        calibration_artifact_sha256="8" * 64,
        parser_runtime_identity_sha256=PARSER_RUNTIME_IDENTITY_SHA256,
        policy=policy,
        global_metrics={**metric, "sample_count": len(LANGUAGES)},
        per_language_metrics={language: dict(metric) for language in LANGUAGES},
        upstream_promoted=True,
        passed=True,
        failure_codes=(),
        receipt_sha256="",
    )
    return replace(
        provisional,
        receipt_sha256=provisional.compute_sha256(),
    )


def _compile(category: str, *, receipt: dict[str, object] | None = None):
    return compile_model_category_evidence(
        category=category,
        records=_records(category),
        heldout_manifest=HELDOUT_MANIFEST,
        task_database_identity_sha256=DATABASE_ID,
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
        strict_uncertainty_receipt=receipt,
    )


def test_compiler_recomputes_records_and_produces_accepted_artifact():
    compiled = _compile("tokenizer_fidelity")
    assert compiled["passed"] is True
    assert compiled["metrics"] == {
        "exact_roundtrip_rate": 1.0,
        "id_stability_rate": 1.0,
        "reserved_control_violation_count": 0,
    }
    evidence = [_category(category) for category in CATEGORIES]
    evidence[0] = compiled
    result = verify_model_acceptance(
        (_evaluation("validation"), _evaluation("test")),
        tuple(
            _inference(task)
            for task in (
                "code_to_semantic",
                "semantic_continuation",
                "semantic_to_code",
            )
        ),
        tuple(evidence),
    )
    assert result.acceptance_passed is True


def test_semantic_compiler_requires_and_validates_current_runtime_receipt():
    receipt = _strict_receipt()
    compiled = _compile("semantic_fidelity", receipt=receipt.to_dict())
    assert compiled["passed"] is True
    assert compiled["metrics"]["strict_uncertainty_receipt_sha256"] == (
        receipt.receipt_sha256
    )
    assert validate_strict_uncertainty_receipt(receipt.to_dict()) == receipt

    weak = replace(
        receipt,
        global_metrics={
            **receipt.global_metrics,
            "selective_coverage_at_025": 0.0,
        },
        receipt_sha256="",
    )
    weak = replace(weak, receipt_sha256=weak.compute_sha256())
    with pytest.raises(ValueError, match="current runtime gates"):
        _compile("semantic_fidelity", receipt=weak.to_dict())


def test_failed_case_outcome_is_published_as_failed_not_promoted():
    records = _records("tokenizer_fidelity")
    records[0]["exact_roundtrip"] = False
    payload = compile_model_category_evidence(
        category="tokenizer_fidelity",
        records=records,
        heldout_manifest=HELDOUT_MANIFEST,
        task_database_identity_sha256=DATABASE_ID,
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
    )
    assert payload["passed"] is False
    assert payload["failure_codes"] == ["category_gate_failed"]
    assert payload["metrics"]["exact_roundtrip_rate"] < 1.0


def test_compiler_rejects_substituted_case_and_unrelated_receipt():
    records = _records("safety")
    records[0]["example_id"] = "f" * 64
    with pytest.raises(ValueError, match="inventory mismatch"):
        compile_model_category_evidence(
            category="safety",
            records=records,
            heldout_manifest=HELDOUT_MANIFEST,
            task_database_identity_sha256=DATABASE_ID,
            checkpoint_sha256=CHECKPOINT_ID,
            checkpoint_step=86_016,
        )
    with pytest.raises(ValueError, match="only valid for semantic"):
        _compile("safety", receipt=_strict_receipt().to_dict())


def test_checkpoint_digest_and_script_entrypoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-fixture")
    assert len(checkpoint_sha256(checkpoint)) == 64
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_model_category_evidence.py"),
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--heldout-manifest" in completed.stdout
    assert "--strict-uncertainty-receipt" in completed.stdout
