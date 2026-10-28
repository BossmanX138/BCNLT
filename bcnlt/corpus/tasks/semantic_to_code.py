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
            family="semantic_to_code",
            template_id="semantic-to-python-v1",
            prompt_mode="semantic_code",
            target_mode="raw_code",
            prompt=(
                "Reconstruct parseable Python from these canonical BCNLT records:\n"
                + canonical_record_stream(records)
            ),
            target=document.content,
            split=split,
        )
    ]
