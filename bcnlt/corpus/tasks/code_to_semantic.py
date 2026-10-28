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
            family="code_to_semantic",
            template_id="python-to-semantic-v1",
            prompt_mode="raw_code",
            target_mode="semantic_code",
            prompt=(
                "Convert this Python source into canonical BCNLT semantic records:\n"
                + document.content
            ),
            target=canonical_record_stream(records),
            split=split,
        )
    ]
