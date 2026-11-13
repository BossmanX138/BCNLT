from __future__ import annotations

from collections.abc import Sequence

from bcnlt.canonical import canonical_record_bytes
from bcnlt.core import ContextualUnit
from bcnlt.semantic_record import SemanticTokenRecord

from ..curriculum import CurriculumRecord, make_curriculum_record
from ..database import SourceDocumentRecord


def generate(
    document: SourceDocumentRecord,
    units: Sequence[ContextualUnit],
    records: Sequence[SemanticTokenRecord],
    *,
    split: str,
) -> list[CurriculumRecord]:
    del units
    generated: list[CurriculumRecord] = []
    seen_definitions: set[str] = set()
    for index, record in enumerate(records):
        if record.defines is None or record.defines in seen_definitions:
            continue
        seen_definitions.add(record.defines)
        generated.append(
            make_curriculum_record(
                document,
                family="structural_reasoning",
                template_id="resolve-definition-link-v1",
                prompt_mode="mixed",
                target_mode="text",
                prompt=(
                    "Which exact symbol identity does this semantic reference resolve to?\n"
                    + canonical_record_bytes(record).decode("utf-8")
                ),
                target=record.defines,
                split=split,
                extra_provenance={"record_index": str(index)},
            )
        )
        if len(generated) >= 8:
            break
    return generated
