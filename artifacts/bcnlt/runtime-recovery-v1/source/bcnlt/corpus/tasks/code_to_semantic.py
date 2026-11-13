from __future__ import annotations

import json
from collections.abc import Sequence

from bcnlt.core import ContextualUnit
from bcnlt.semantic_record import SemanticTokenRecord

from ..curriculum import (
    CurriculumRecord,
    canonical_record_stream,
    make_curriculum_record,
    pair_within_character_budget,
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
    instruction = "Convert this source into canonical BCNLT semantic records.\n"
    generated = []
    for window in semantic_code_windows(
        document, units, records, instruction=instruction
    ):
        provenance = dict(window.provenance)
        source_line_start = provenance.get("source_line_start", "1")
        prior_history = json.dumps(
            list(window.records[0].recent_history),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = (
            instruction
            + f"Source path: {document.relative_path}\n"
            + f"Original line start: {source_line_start}\n"
            + f"Prior semantic history: {prior_history}\n"
            + "Source code:\n"
            + window.source
        )
        target = canonical_record_stream(window.records)
        if not pair_within_character_budget(prompt, target):
            continue
        generated.append(
            make_curriculum_record(
                document,
                family="code_to_semantic",
                template_id="language-context-to-semantic-v3",
                prompt_mode="raw_code",
                target_mode="semantic_code",
                prompt=prompt,
                target=target,
                split=split,
                extra_provenance=provenance,
            )
        )
    return generated
