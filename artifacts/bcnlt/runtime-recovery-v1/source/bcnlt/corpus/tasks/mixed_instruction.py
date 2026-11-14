from __future__ import annotations

from collections.abc import Sequence

from bcnlt.core import ContextualUnit
from bcnlt.semantic_record import SemanticTokenRecord

from ..curriculum import (
    CurriculumRecord,
    canonical_record_stream,
    make_curriculum_record,
    semantic_code_windows,
)
from ..database import SourceDocumentRecord


def generate(
    document: SourceDocumentRecord,
    units: Sequence[ContextualUnit],
    records: Sequence[SemanticTokenRecord],
    *,
    split: str,
) -> list[CurriculumRecord]:
    instruction = (
        "Use the semantic context to return the original parseable source.\n"
        "Semantic context:\n"
    )
    return [
        make_curriculum_record(
            document,
            family="mixed_instruction",
            template_id="semantic-context-to-source-v2",
            prompt_mode="mixed",
            target_mode="raw_code",
            prompt=instruction + canonical_record_stream(window.records),
            target=window.source,
            split=split,
            extra_provenance=dict(window.provenance),
        )
        for window in semantic_code_windows(
            document, units, records, instruction=instruction
        )
    ]
