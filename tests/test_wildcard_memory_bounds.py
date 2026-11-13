from __future__ import annotations

from pathlib import Path

from runtime_import import select_bcnlt_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = (
    PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate" / "source"
)
select_bcnlt_runtime(RUNTIME_SOURCE)

from bcnlt.core import ContextualUnit  # noqa: E402
from bcnlt.wildcard import (  # noqa: E402
    MAX_CODING_OUTCOMES_PER_RECORD,
    MAX_CONTEXT_FINGERPRINTS_PER_RECORD,
    MAX_DOCUMENT_IDENTITIES_PER_RECORD,
    MAX_EVIDENCE_ITEMS_PER_RECORD,
    ProvisionalMeaning,
    WildcardMemory,
)


def test_repeated_wildcard_learning_has_bounded_evidence_state() -> None:
    memory = WildcardMemory(language="c", semantic_spec_sha256="a" * 64)
    meaning = ProvisionalMeaning(
        category="call",
        hypotheses=("external call",),
        evidence=("base evidence",),
        confidence=0.45,
        target_event="CALL",
        target_ir_kind="call",
    )
    record = None
    for index in range(600):
        unit = ContextualUnit(
            surface="external_call",
            event="CALL",
            scope=f"function.block_{index}",
            span=(index + 1, 0, index + 1, 8),
            ast_relation="call_expression.function",
            recent_history=(f"history-{index}",),
        )
        record = memory.remember(
            unit,
            ProvisionalMeaning(
                **{
                    **meaning.__dict__,
                    "evidence": ("base evidence", f"evidence-{index}"),
                }
            ),
            document_identity=f"document-{index}",
        )

    assert record is not None
    assert record.seen_count == 600
    assert len(record.evidence) == MAX_EVIDENCE_ITEMS_PER_RECORD
    assert len(record.context_fingerprints) == MAX_CONTEXT_FINGERPRINTS_PER_RECORD
    assert len(record.document_identities) == MAX_DOCUMENT_IDENTITIES_PER_RECORD


def test_coding_outcome_evidence_is_bounded() -> None:
    memory = WildcardMemory(language="c", semantic_spec_sha256="b" * 64)
    unit = ContextualUnit(
        surface="external_call",
        event="CALL",
        scope="function.body",
        span=(1, 0, 1, 8),
        ast_relation="call_expression.function",
    )
    memory.remember(
        unit,
        ProvisionalMeaning(
            category="call",
            hypotheses=("external call",),
            evidence=("base evidence",),
            confidence=0.45,
            target_event="CALL",
            target_ir_kind="call",
        ),
        document_identity="document-1",
    )
    evidence = tuple(f"evidence-{index}" for index in range(200))
    record = memory.record_coding_outcome(
        unit,
        target_event="CALL",
        target_ir_kind="call",
        evidence=evidence,
        document_identity="document-1",
        verification_oracle="unit-test",
        result_fingerprint="c" * 64,
        correct=True,
    )

    assert len(record.evidence) == MAX_EVIDENCE_ITEMS_PER_RECORD
    assert len(record.coding_outcomes) <= MAX_CODING_OUTCOMES_PER_RECORD
    assert len(record.coding_outcomes[0].evidence) == MAX_EVIDENCE_ITEMS_PER_RECORD
