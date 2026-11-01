from __future__ import annotations

import json
from collections.abc import Sequence

from bcnlt.semantic_schema import KnowledgeState
from bcnlt.uncertainty import KnowledgeBoundaryAssessment

from ..curriculum import CurriculumRecord, make_curriculum_record
from ..database import SourceDocumentRecord


def generate(
    document: SourceDocumentRecord,
    assessments: Sequence[KnowledgeBoundaryAssessment],
    *,
    split: str,
) -> list[CurriculumRecord]:
    """Build evidence-bound targets that teach the model when it does not know."""

    if not assessments:
        return []
    selected = sorted(
        enumerate(assessments),
        key=lambda item: (-item[1].initial_score.uncertainty, item[0]),
    )[:8]
    generated: list[CurriculumRecord] = []
    for index, assessment in selected:
        state = assessment.state.value
        evidence = tuple(dict.fromkeys(assessment.evidence))
        supported = state == KnowledgeState.KNOWN.value or bool(evidence)
        resolution = "supported" if supported else "unresolved"
        target = {
            "initial_confidence": assessment.initial_score.confidence,
            "initial_uncertainty": assessment.initial_score.uncertainty,
            "final_confidence": assessment.final_score.confidence,
            "final_uncertainty": assessment.final_score.uncertainty,
            "state": state,
            "resolution": resolution,
            "uncertainty_signals": list(assessment.initial_score.signals),
            "final_signals": list(assessment.final_score.signals),
            "uyr_rounds": assessment.rounds,
            "retrievals": list(evidence),
            "wildcard_category": assessment.wildcard_category,
            "wildcard_hypotheses": list(assessment.wildcard_hypotheses),
            "supported_event": assessment.unit.event if supported else None,
        }
        prompt = (
            "Assess this semantic unit at the knowledge boundary. Use only recorded "
            "evidence; return an explicit unresolved result when evidence is absent.\n"
            f"Surface: {assessment.unit.surface}\n"
            f"Event candidate: {assessment.unit.event}\n"
            f"Scope: {assessment.unit.scope}\n"
            f"Initial signals: {json.dumps(list(assessment.initial_score.signals), separators=(',', ':'))}\n"
        )
        generated.append(
            make_curriculum_record(
                document,
                family="knowledge_boundary",
                template_id="evidence-bound-knowledge-state-v1",
                prompt_mode="text",
                target_mode="text",
                prompt=prompt,
                target=json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                split=split,
                extra_provenance={"assessment_index": str(index)},
            )
        )
    return generated
