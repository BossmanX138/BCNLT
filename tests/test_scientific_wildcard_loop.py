from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcnlt.cache import ProjectIndex
from bcnlt.core import ContextualUnit, SourceDocument
from bcnlt.ir import IRKind
from bcnlt.preprocessing import SemanticPreprocessor
from bcnlt.scientific_learning import (
    BoundedProjectionVerifier,
    CodingExperimentFixture,
    HypothesisProjection,
)
from bcnlt.wildcard import (
    DefinitionState,
    ProvisionalMeaning,
    WildcardMemory,
    WildcardResolver,
)


SPEC = "f" * 64


def _unit(index: int) -> ContextualUnit:
    return ContextualUnit(
        surface="match value",
        event="UNKNOWN_SEMANTIC",
        scope=f"module.case_{index}",
        span=(index * 10 + 1, 0, index * 10 + 1, 11),
        ast_relation="match_statement.self",
        recent_history=(f"DEFINE:case_{index}",),
        metadata={"unknown_node": "match_statement"},
    )


def _observe(memory: WildcardMemory, unit: ContextualUnit, document: str) -> None:
    WildcardResolver(memory).resolve_and_remember(
        unit, ProjectIndex(), document_identity=document
    )


def _fixture(
    index: int,
    document: str,
    *,
    expected: object | None = None,
) -> CodingExperimentFixture:
    condition = index % 2 == 1
    when_true = f"true-{index}"
    when_false = f"false-{index}"
    return CodingExperimentFixture(
        experiment_id=f"match-exp-{index}",
        fixture_id=f"trusted/match/{index}",
        fixture_version=1,
        domain="python.match.branch-selection",
        document_identity=document,
        oracle_id="python-language-reference-match",
        oracle_kind="language_spec",
        oracle_version="3.12",
        operation="branch_select",
        inputs={
            "condition": condition,
            "when_true": when_true,
            "when_false": when_false,
        },
        expected_output=(
            expected
            if expected is not None
            else (when_true if condition else when_false)
        ),
    )


def _wrong_control(memory: WildcardMemory, unit: ContextualUnit) -> HypothesisProjection:
    record = memory.lookup(unit)
    assert record is not None
    return HypothesisProjection(
        signature=record.signature,
        revision=record.hypothesis_revision,
        target_event="LOOP",
        target_ir_kind=IRKind.LOOP.value,
        language=record.language,
        semantic_spec_sha256=record.semantic_spec_sha256,
    )


def _run(
    memory: WildcardMemory,
    unit: ContextualUnit,
    index: int,
    document: str,
    *,
    expected: object | None = None,
    verifier: BoundedProjectionVerifier | None = None,
):
    return memory.run_scientific_experiment(
        unit,
        source=f"match value fixture {index}\n",
        fixture=_fixture(index, document, expected=expected),
        negative_control=_wrong_control(memory, unit),
        verifier=verifier,
    )


def _validated_memory(path: Path) -> tuple[WildcardMemory, list[ContextualUnit]]:
    memory = WildcardMemory(
        path,
        language="python",
        semantic_spec_sha256=SPEC,
        minimum_reproductions=3,
    )
    units = [_unit(index) for index in (1, 2, 3)]
    for index, unit in enumerate(units, start=1):
        document = f"case-{index}.py"
        _observe(memory, unit, document)
        _run(memory, unit, index, document)
    return memory, units


def test_three_independent_experiments_validate_but_one_and_replay_do_not(tmp_path: Path):
    path = tmp_path / "learned.jsonl"
    memory = WildcardMemory(
        path,
        language="python",
        semantic_spec_sha256=SPEC,
        minimum_reproductions=3,
    )
    units = [_unit(index) for index in (1, 2, 3)]

    _observe(memory, units[0], "case-1.py")
    first = _run(memory, units[0], 1, "case-1.py")
    replay = _run(memory, units[0], 1, "case-1.py")

    assert first.definition_state is DefinitionState.PROVISIONAL
    assert replay.definition_state is DefinitionState.PROVISIONAL
    assert len(replay.scientific_experiments) == 1

    _observe(memory, units[1], "case-2.py")
    second = _run(memory, units[1], 2, "case-2.py")
    assert second.definition_state is DefinitionState.PROVISIONAL

    _observe(memory, units[2], "case-3.py")
    validated = _run(memory, units[2], 3, "case-3.py")
    reloaded = WildcardMemory(
        path,
        language="python",
        semantic_spec_sha256=SPEC,
        minimum_reproductions=3,
    )

    assert validated.definition_state is DefinitionState.VALIDATED
    assert validated.validated_domain == "python.match.branch-selection"
    assert validated.validated_by == "bcnlt-bounded-projection:1"
    assert reloaded.lookup(units[0]).definition_state is DefinitionState.VALIDATED


def test_later_verified_failure_revokes_acceptance_and_survives_reload(tmp_path: Path):
    path = tmp_path / "learned.jsonl"
    memory, units = _validated_memory(path)

    contradicted = _run(
        memory,
        units[0],
        4,
        "case-1.py",
        expected="independently-authored-wrong-result",
    )
    reloaded = WildcardMemory(
        path,
        language="python",
        semantic_spec_sha256=SPEC,
    )

    assert contradicted.definition_state is DefinitionState.PROVISIONAL
    assert contradicted.validated_by is None
    assert "scientific_contradiction" in contradicted.evidence_kinds
    assert "scientifically_reproduced" not in contradicted.evidence_kinds
    assert reloaded.lookup(units[0]).definition_state is DefinitionState.PROVISIONAL


def test_hypothesis_revision_requires_fresh_experiments(tmp_path: Path):
    memory, units = _validated_memory(tmp_path / "learned.jsonl")
    prior = memory.lookup(units[0])
    assert prior is not None

    revised = memory.remember(
        units[0],
        ProvisionalMeaning(
            category="control_flow",
            hypotheses=("the construct may be a loop instead",),
            evidence=("new grammar evidence",),
            confidence=0.6,
            target_event="LOOP",
            target_ir_kind=IRKind.LOOP.value,
        ),
        document_identity="case-1.py",
    )

    assert revised.hypothesis_revision == prior.hypothesis_revision + 1
    assert revised.hypothesis_fingerprint != prior.hypothesis_fingerprint
    assert revised.definition_state is DefinitionState.PROVISIONAL
    assert revised.validated_by is None
    assert len(revised.scientific_experiments) == 3


def test_loader_rejects_replayed_record_that_rolls_back_hypothesis_revision(
    tmp_path: Path,
):
    path = tmp_path / "learned.jsonl"
    memory, units = _validated_memory(path)
    revision_one_line = path.read_text(encoding="utf-8").splitlines()[-1]

    memory.remember(
        units[0],
        ProvisionalMeaning(
            category="control_flow",
            hypotheses=("the construct may be a loop instead",),
            evidence=("new grammar evidence",),
            confidence=0.6,
            target_event="LOOP",
            target_ir_kind=IRKind.LOOP.value,
        ),
        document_identity="case-1.py",
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(revision_one_line + "\n")

    with pytest.raises(ValueError, match="hypothesis revision rollback"):
        WildcardMemory(
            path,
            semantic_spec_sha256=memory.semantic_spec_sha256,
            minimum_reproductions=3,
        )


def test_unavailable_verifier_and_unsupported_fixture_remain_provisional(tmp_path: Path):
    memory = WildcardMemory(
        tmp_path / "learned.jsonl",
        language="python",
        semantic_spec_sha256=SPEC,
    )
    unit = _unit(1)
    _observe(memory, unit, "case-1.py")

    unavailable = _run(
        memory,
        unit,
        1,
        "case-1.py",
        verifier=BoundedProjectionVerifier(available=False),
    )

    assert unavailable.definition_state is DefinitionState.PROVISIONAL
    assert unavailable.scientific_experiments[-1].status.value == "verifier_unavailable"


def test_mismatched_fixture_and_stale_control_are_rejected_before_recording(tmp_path: Path):
    memory = WildcardMemory(
        tmp_path / "learned.jsonl",
        language="python",
        semantic_spec_sha256=SPEC,
    )
    unit = _unit(1)
    _observe(memory, unit, "case-1.py")
    before = memory.lookup(unit)

    with pytest.raises(ValueError, match="document identity"):
        memory.run_scientific_experiment(
            unit,
            source="match value fixture 1\n",
            fixture=_fixture(1, "forged-name.py"),
            negative_control=_wrong_control(memory, unit),
        )

    stale = HypothesisProjection(
        **{**_wrong_control(memory, unit).__dict__, "revision": 99}
    )
    with pytest.raises(ValueError, match="current hypothesis revision"):
        memory.run_scientific_experiment(
            unit,
            source="match value fixture 1\n",
            fixture=_fixture(1, "case-1.py"),
            negative_control=stale,
        )

    assert memory.lookup(unit) == before


def test_forged_validated_flag_with_legacy_claims_is_demoted_on_load(tmp_path: Path):
    path = tmp_path / "learned.jsonl"
    memory = WildcardMemory(path)
    unit = _unit(1)
    _observe(memory, unit, "case-1.py")
    memory.record_coding_outcome(
        unit,
        target_event="BRANCH",
        target_ir_kind=IRKind.BRANCH.value,
        evidence=("caller claimed pass",),
        document_identity="case-1.py",
        verification_oracle="caller:untrusted",
        result_fingerprint="claimed-result",
        correct=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    payload["definition_state"] = "validated"
    payload["validated_by"] = "forged"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")

    loaded = WildcardMemory(path)

    assert loaded.lookup(unit).definition_state is DefinitionState.PROVISIONAL
    assert loaded.lookup(unit).validated_by is None


def test_loader_recomputes_hypothesis_fingerprint_after_projection_tampering(
    tmp_path: Path,
):
    path = tmp_path / "wildcard.jsonl"
    memory, units = _validated_memory(path)
    original = memory.lookup(units[0])
    assert original is not None
    assert original.definition_state is DefinitionState.VALIDATED

    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    old_fingerprint = payload["hypothesis_fingerprint"]
    payload["target_event"] = "RETURN"
    payload["target_ir_kind"] = "return"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    reloaded = WildcardMemory(
        path,
        semantic_spec_sha256=memory.semantic_spec_sha256,
        minimum_reproductions=3,
    )
    record = reloaded.lookup(units[0])
    assert record is not None
    assert record.target_event == "RETURN"
    assert record.hypothesis_fingerprint != old_fingerprint
    assert record.definition_state is DefinitionState.PROVISIONAL
    assert len(reloaded.validated_extended_core()) == 0


def test_tampered_scientific_evidence_fails_closed_on_load(tmp_path: Path):
    path = tmp_path / "learned.jsonl"
    memory = WildcardMemory(
        path,
        language="python",
        semantic_spec_sha256=SPEC,
    )
    unit = _unit(1)
    _observe(memory, unit, "case-1.py")
    _run(memory, unit, 1, "case-1.py")

    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[-1])
    payload["scientific_experiments"][0]["actual_output"] = "tampered"
    lines[-1] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity mismatch"):
        WildcardMemory(
            path,
            language="python",
            semantic_spec_sha256=SPEC,
        )


def test_extended_core_library_tracks_acceptance_and_revocation(tmp_path: Path):
    path = tmp_path / "learned.jsonl"
    memory, units = _validated_memory(path)
    library = memory.validated_extended_core()
    assert len(library) == 1
    entry = library.by_signature(memory.signature_for(units[0]))
    assert entry is not None
    assert entry.common_core_projection.event == "BRANCH"
    assert entry.common_core_projection.ir_kind == IRKind.BRANCH.value
    assert entry.validated_domain == "python.match.branch-selection"
    _run(memory, units[0], 4, "case-1.py", expected="contradiction")
    assert len(memory.validated_extended_core()) == 0


def test_validated_meaning_enters_extended_core_without_reclassifying_common_core(
    tmp_path: Path,
):
    path = tmp_path / "learned.jsonl"
    memory, units = _validated_memory(path)

    library = memory.validated_extended_core()

    assert len(library) == 1
    entry = library.by_signature(memory.signature_for(units[0]))
    assert entry is not None
    assert entry.extended_core_id.startswith("bcnlt.extended.python.")
    assert entry.common_core_projection.event == "BRANCH"
    assert entry.common_core_projection.ir_kind == IRKind.BRANCH.value

    result = SemanticPreprocessor(
        memory_path=path,
        memory_read_only=True,
    ).process(
        SourceDocument(
            "runtime.py",
            units[0].surface,
            None,
            ProjectIndex(),
            language="python",
        ),
        [units[0]],
        semantic_spec_sha256=SPEC,
    )
    resolved = result.units[0]

    assert resolved.event == "UNKNOWN_SEMANTIC"
    assert resolved.ir is not None
    assert resolved.ir.kind is IRKind.EXTENSION
    assert resolved.metadata["extended_core_id"] == entry.extended_core_id
    assert resolved.metadata["extended_core_state"] == "validated"
    assert resolved.metadata["extended_core_projection"] == {
        "event": "BRANCH",
        "ir_kind": IRKind.BRANCH.value,
    }
    assert "learned_definition_state" not in resolved.metadata


def test_sqlite_scientific_state_reloads_with_same_decision(tmp_path: Path):
    path = tmp_path / "learned.sqlite3"
    memory = WildcardMemory(
        path,
        language="python",
        semantic_spec_sha256=SPEC,
        storage="sqlite",
    )
    units = [_unit(index) for index in (1, 2, 3)]
    for index, unit in enumerate(units, start=1):
        document = f"case-{index}.py"
        _observe(memory, unit, document)
        _run(memory, unit, index, document)
    assert memory.lookup(units[0]).definition_state is DefinitionState.VALIDATED
    memory.close()

    reloaded = WildcardMemory(
        path,
        language="python",
        semantic_spec_sha256=SPEC,
        storage="sqlite",
    )
    assert reloaded.lookup(units[0]).definition_state is DefinitionState.VALIDATED
    assert len(reloaded.validated_extended_core()) == 1
    reloaded.close()



def test_renamed_duplicate_fixture_content_is_not_independent_reproduction(tmp_path: Path):
    memory = WildcardMemory(
        tmp_path / "learned.jsonl",
        language="python",
        semantic_spec_sha256=SPEC,
    )
    units = [_unit(index) for index in (1, 2, 3)]
    template = _fixture(1, "case-1.py")
    latest = None
    for index, unit in enumerate(units, start=1):
        document = f"case-{index}.py"
        _observe(memory, unit, document)
        duplicate = CodingExperimentFixture(
            **{
                **template.__dict__,
                "experiment_id": f"renamed-exp-{index}",
                "fixture_id": f"renamed/fixture/{index}",
                "document_identity": document,
            }
        )
        latest = memory.run_scientific_experiment(
            unit,
            source=f"distinct source {index}\n",
            fixture=duplicate,
            negative_control=_wrong_control(memory, unit),
        )

    assert latest is not None
    assert latest.definition_state is DefinitionState.PROVISIONAL

