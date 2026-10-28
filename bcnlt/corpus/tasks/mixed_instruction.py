from __future__ import annotations

from collections.abc import Sequence

from bcnlt.core import ContextualUnit
from bcnlt.semantic_record import SemanticTokenRecord

from ..curriculum import CurriculumRecord, canonical_record_stream, make_curriculum_record
from ..database import SourceDocumentRecord


def generate(
    document: SourceDocumentRecord,
    units: Sequence[ContextualUnit],
    records: Sequence[SemanticTokenRecord],
    *,
    split: str,
) -> list[CurriculumRecord]:
    del units
    return [
        make_curriculum_record(
            document,
            family="mixed_instruction",
            template_id="semantic-context-to-python-v1",
            prompt_mode="mixed",
            target_mode="raw_code",
            prompt=(
                "Use the semantic context to return the original parseable Python.\n"
                "Semantic context:\n"
                + canonical_record_stream(records)
            ),
            target=document.content,
            split=split,
        )
    ]
