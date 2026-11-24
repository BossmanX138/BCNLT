from __future__ import annotations

from collections.abc import Sequence

from bcnlt.core import ContextualUnit
from bcnlt.semantic_record import SemanticTokenRecord

from ..curriculum import (
    MAX_CONTINUATION_CONTEXT_RECORDS,
    CurriculumRecord,
    canonical_record_stream,
    make_curriculum_record,
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
    if len(records) < 2:
        return []
    boundary = max(1, len(records) // 2)
    context_start = max(0, boundary - MAX_CONTINUATION_CONTEXT_RECORDS)
    return [
        make_curriculum_record(
            document,
            family="semantic_continuation",
            template_id="semantic-next-record-v1",
            prompt_mode="semantic_code",
            target_mode="semantic_code",
            prompt=(
                "Continue this canonical BCNLT semantic stream with the next record:\n"
                + canonical_record_stream(records[context_start:boundary])
            ),
            target=canonical_record_stream(records[boundary : boundary + 1]),
            split=split,
            extra_provenance={
                "record_boundary": str(boundary),
                "context_start": str(context_start),
            },
        )
    ]
