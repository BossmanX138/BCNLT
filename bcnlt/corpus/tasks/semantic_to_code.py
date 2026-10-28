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
    instruction = "Reconstruct parseable Python from these canonical BCNLT records:\n"
    semantic = canonical_record_stream(records)
    if not pair_within_character_budget(instruction, semantic, document.content):
        return []
    return [
        make_curriculum_record(
            document,
            family="semantic_to_code",
            template_id="semantic-to-python-v1",
            prompt_mode="semantic_code",
            target_mode="raw_code",
            prompt=instruction + semantic,
            target=document.content,
            split=split,
        )
    ]
