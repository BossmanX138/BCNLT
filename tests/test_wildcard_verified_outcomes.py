from __future__ import annotations

from pathlib import Path

import pytest

from bcnlt.cache import ProjectIndex
from bcnlt.core import ContextualUnit
from bcnlt.ir import IRKind
from bcnlt.wildcard import DefinitionState, WildcardMemory, WildcardResolver


def _unknown(*, scope: str, line: int) -> ContextualUnit:
    return ContextualUnit(
        surface="match value",
        event="UNKNOWN_SEMANTIC",
        scope=scope,
        span=(line, 0, line, 11),
        ast_relation="match_statement.self",
        metadata={"unknown_node": "match_statement"},
    )


def _observe_two(memory: WildcardMemory):
    first = _unknown(scope="module", line=1)
    second = _unknown(scope="module.worker", line=20)
    resolver = WildcardResolver(memory)
    resolver.resolve_and_remember(first, ProjectIndex(), document_identity="first.py")
    resolver.resolve_and_remember(second, ProjectIndex(), document_identity="second.py")
    return first, second


def _record_success(memory: WildcardMemory, unit: ContextualUnit, *, document_identity: str):
    return memory.record_coding_outcome(
        unit,
        target_event="BRANCH",
        target_ir_kind=IRKind.BRANCH.value,
        evidence=("the projected coding fixture reportedly passed",),
        document_identity=document_identity,
        verification_oracle=f"pytest:{document_identity}",
        result_fingerprint=f"result:{document_identity}",
        correct=True,
    )


def test_repetition_without_coding_outcomes_remains_provisional(tmp_path: Path):
    memory = WildcardMemory(tmp_path / "learned.jsonl")
    _, second = _observe_two(memory)
    assert memory.lookup(second).definition_state is DefinitionState.PROVISIONAL


def test_repeated_claimed_outcomes_stay_provisional_without_an_executable_verifier(tmp_path: Path):
    path = tmp_path / "learned.jsonl"
    memory = WildcardMemory(path)
    first, second = _observe_two(memory)
    _record_success(memory, first, document_identity="first.py")
    learned = _record_success(memory, second, document_identity="second.py")
    reloaded = WildcardMemory(path)
    assert learned.definition_state is DefinitionState.PROVISIONAL
    assert learned.validated_by is None
    assert "reproduced_claim_pending_verifier" in learned.evidence_kinds
    assert reloaded.lookup(second).definition_state is DefinitionState.PROVISIONAL


def test_failed_outcome_remains_provisional(tmp_path: Path):
    memory = WildcardMemory(tmp_path / "learned.jsonl")
    first, second = _observe_two(memory)
    _record_success(memory, first, document_identity="first.py")
    _record_success(memory, second, document_identity="second.py")
    failed = memory.record_coding_outcome(
        second,
        target_event="BRANCH",
        target_ir_kind=IRKind.BRANCH.value,
        evidence=("the regression fixture failed",),
        document_identity="second.py",
        verification_oracle="pytest:second-regression",
        result_fingerprint="failed:second.py",
        correct=False,
    )
    assert failed.definition_state is DefinitionState.PROVISIONAL


def test_changing_only_document_name_does_not_create_independent_context(tmp_path: Path):
    unit = _unknown(scope="module", line=1)
    memory = WildcardMemory(tmp_path / "learned.jsonl")
    resolver = WildcardResolver(memory)
    resolver.resolve_and_remember(unit, ProjectIndex(), document_identity="first.py")
    resolver.resolve_and_remember(unit, ProjectIndex(), document_identity="renamed.py")
    assert len(memory.lookup(unit).context_fingerprints) == 1


def test_truthy_string_is_not_a_correct_result(tmp_path: Path):
    memory = WildcardMemory(tmp_path / "learned.jsonl")
    first, _ = _observe_two(memory)
    with pytest.raises(TypeError, match="real boolean"):
        memory.record_coding_outcome(
            first,
            target_event="BRANCH",
            target_ir_kind=IRKind.BRANCH.value,
            evidence=("claimed pass",),
            document_identity="first.py",
            verification_oracle="pytest:first",
            result_fingerprint="result:first",
            correct="false",  # type: ignore[arg-type]
        )


def test_direct_metadata_only_validation_is_disabled(tmp_path: Path):
    memory = WildcardMemory(tmp_path / "learned.jsonl")
    first, _ = _observe_two(memory)
    with pytest.raises(RuntimeError, match="executable"):
        memory.validate(
            first,
            target_event="BRANCH",
            target_ir_kind=IRKind.BRANCH.value,
            evidence=("claimed pass",),
            document_identity="first.py",
            verification_oracle="pytest:first",
            result_fingerprint="result:first",
        )
