from __future__ import annotations

import json

from bcnlt.scientific_learning import (
    BoundedProjectionVerifier,
    CodingExperimentFixture,
    HypothesisProjection,
    VerificationStatus,
    load_experiment_fixture,
)


SPEC = "a" * 64


def _projection(event: str = "BRANCH", ir_kind: str = "branch") -> HypothesisProjection:
    return HypothesisProjection(
        signature="python|spec|unknown|match|match value",
        revision=1,
        target_event=event,
        target_ir_kind=ir_kind,
        language="python",
        semantic_spec_sha256=SPEC,
    )


def _branch_fixture(*, expected: object = "yes") -> CodingExperimentFixture:
    return CodingExperimentFixture(
        experiment_id="branch-true-001",
        fixture_id="trusted/match/true",
        fixture_version=1,
        domain="python.match.branch-selection",
        document_identity="first.py",
        oracle_id="python-language-reference-match",
        oracle_kind="language_spec",
        oracle_version="3.12",
        operation="branch_select",
        inputs={"condition": True, "when_true": "yes", "when_false": "no"},
        expected_output=expected,
    )


def test_candidate_projection_causally_passes_and_wrong_projection_is_rejected():
    verifier = BoundedProjectionVerifier()

    result = verifier.verify(
        _projection(),
        _branch_fixture(),
        source="match value:\n    case _: result = 'yes'\n",
        context_fingerprint="context-001",
        negative_control=_projection("LOOP", "loop"),
    )

    assert result.status is VerificationStatus.PASSED
    assert result.intended_passed is True
    assert result.negative_control_rejected is True
    assert result.actual_output == "yes"
    assert result.source_sha256 != result.fixture_sha256
    assert len(result.evidence_sha256) == 64


def test_nondiscriminating_negative_control_invalidates_the_experiment():
    result = BoundedProjectionVerifier().verify(
        _projection(),
        _branch_fixture(),
        source="match value:\n    case _: result = 'yes'\n",
        context_fingerprint="context-001",
        negative_control=_projection(),
    )

    assert result.status is VerificationStatus.CONTROL_FAILED
    assert result.intended_passed is True
    assert result.negative_control_rejected is False


def test_wrong_expected_result_records_a_candidate_failure():
    result = BoundedProjectionVerifier().verify(
        _projection(),
        _branch_fixture(expected="no"),
        source="match value:\n    case _: result = 'yes'\n",
        context_fingerprint="context-001",
        negative_control=_projection("LOOP", "loop"),
    )

    assert result.status is VerificationStatus.CANDIDATE_FAILED
    assert result.intended_passed is False


def test_unsupported_and_unsafe_programs_fail_closed():
    unsupported = CodingExperimentFixture(
        **{
            **_branch_fixture().__dict__,
            "experiment_id": "unsafe-001",
            "operation": "python_eval",
            "inputs": {"code": "__import__('os').system('echo unsafe')"},
        }
    )

    result = BoundedProjectionVerifier().verify(
        _projection(),
        unsupported,
        source="match value:\n    case _: pass\n",
        context_fingerprint="context-unsafe",
        negative_control=_projection("LOOP", "loop"),
    )

    assert result.status is VerificationStatus.UNSUPPORTED
    assert result.actual_output is None


def test_unavailable_verifier_never_reports_success():
    result = BoundedProjectionVerifier(available=False).verify(
        _projection(),
        _branch_fixture(),
        source="match value:\n    case _: pass\n",
        context_fingerprint="context-001",
        negative_control=_projection("LOOP", "loop"),
    )

    assert result.status is VerificationStatus.VERIFIER_UNAVAILABLE
    assert result.intended_passed is False


def test_external_fixture_file_loads_as_the_same_auditable_identity(tmp_path):
    fixture = _branch_fixture()
    path = tmp_path / "branch-fixture.json"
    path.write_text(json.dumps(fixture.__dict__), encoding="utf-8")

    loaded = load_experiment_fixture(path)

    assert loaded == fixture
    assert loaded.fingerprint == fixture.fingerprint


def test_bounded_operator_rejects_resource_amplifying_string_multiplication():
    fixture = CodingExperimentFixture(
        **{
            **_branch_fixture().__dict__,
            "experiment_id": "operator-unsafe-001",
            "fixture_id": "trusted/operator/unsafe",
            "operation": "binary_operator",
            "inputs": {"operator": "*", "left": "x", "right": 5000},
            "expected_output": "x",
        }
    )
    projection = _projection("OPERATOR", "operator")

    result = BoundedProjectionVerifier().verify(
        projection,
        fixture,
        source="result = 'x' * 5000\n",
        context_fingerprint="context-operator",
        negative_control=_projection("BRANCH", "branch"),
    )

    assert result.status is VerificationStatus.INVALID
    assert result.actual_output is None

