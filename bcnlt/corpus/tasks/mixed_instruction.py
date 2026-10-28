from __future__ import annotations

from collections.abc import Sequence

from bcnlt.core import ContextualUnit
from bcnlt.semantic_record import SemanticTokenRecord

from ..curriculum import (
    CurriculumRecord,
    canonical_record_stream,
    make_curriculum_record,
    pair_within_character_budget,
)
from ..database import SourceDocumentRecord


def generate(
    document: SourceDocumentRecord,
    units: Sequence[ContextualUnit],
    records: Sequence[SemanticTokenRecord],
    *,
    split: str,
) -> list[CurriculumRecord]:
    del units
    instruction = (
        "Use the semantic context to return the original parseable Python.\n"
        "Semantic context:\n"
    )
    semantic = canonical_record_stream(records)
    if not pair_within_character_budget(instruction, semantic, document.content):
        return []
    return [
        make_curriculum_record(
            document,
            family="mixed_instruction",
            template_id="semantic-context-to-python-v1",
            prompt_mode="mixed",
            target_mode="raw_code",
            prompt=instruction + semantic,
            target=document.content,
            split=split,
        )
    ]
