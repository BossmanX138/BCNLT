from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from bcnlt.canonical import canonical_record_bytes
from bcnlt.core import ContextualUnit
from bcnlt.semantic_record import SemanticTokenRecord

from .database import SourceDocumentRecord


MAX_CURRICULUM_PAIR_CHARACTERS = 16_000
MAX_CONTINUATION_CONTEXT_RECORDS = 8


@dataclass(frozen=True)
class CurriculumRecord:
    record_id: str
    family: str
    template_id: str
    source_document_ids: tuple[str, ...]
    prompt_mode: str
    target_mode: str
    prompt: str
    target: str
    provenance: tuple[tuple[str, str], ...]
    split: str

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "family": self.family,
            "template_id": self.template_id,
            "source_document_ids": list(self.source_document_ids),
            "prompt_mode": self.prompt_mode,
            "target_mode": self.target_mode,
            "prompt": self.prompt,
            "target": self.target,
            "provenance": dict(self.provenance),
            "split": self.split,
        }


def canonical_record_stream(records: Sequence[SemanticTokenRecord]) -> str:
    return b"".join(canonical_record_bytes(record) for record in records).decode("utf-8")


def pair_within_character_budget(*parts: str) -> bool:
    return sum(len(part) for part in parts) <= MAX_CURRICULUM_PAIR_CHARACTERS


def make_curriculum_record(
    document: SourceDocumentRecord,
    *,
    family: str,
    template_id: str,
    prompt_mode: str,
    target_mode: str,
    prompt: str,
    target: str,
    split: str,
    extra_provenance: dict[str, str] | None = None,
) -> CurriculumRecord:
    provenance = {
        "source_provider": document.source_provider,
        "source_name": document.source_name,
        "source_revision": document.source_revision,
        "repository_identity": document.repository_identity,
        "relative_path": document.relative_path,
        "content_sha256": document.content_sha256,
        "license_spdx": document.license_spdx,
        "ownership": document.ownership,
        "attribution": document.attribution,
        "upstream_url": document.upstream_url,
    }
    if extra_provenance:
        provenance.update(extra_provenance)
    normalized_prompt = _normalize(prompt)
    normalized_target = _normalize(target)
    body = {
        "family": family,
        "template_id": template_id,
        "source_document_ids": [document.document_id],
        "prompt_mode": prompt_mode,
        "target_mode": target_mode,
        "prompt": normalized_prompt,
        "target": normalized_target,
        "provenance": provenance,
        "split": split,
    }
    record_id = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return CurriculumRecord(
        record_id=record_id,
        family=family,
        template_id=template_id,
        source_document_ids=(document.document_id,),
        prompt_mode=prompt_mode,
        target_mode=target_mode,
        prompt=normalized_prompt,
        target=normalized_target,
        provenance=tuple(sorted(provenance.items())),
        split=split,
    )


def generate_document_curriculum(
    document: SourceDocumentRecord,
    units: Sequence[ContextualUnit],
    records: Sequence[SemanticTokenRecord],
    *,
    split: str = "unassigned",
) -> Iterable[CurriculumRecord]:
    from .tasks.code_to_semantic import generate as code_to_semantic
    from .tasks.mixed_instruction import generate as mixed_instruction
    from .tasks.semantic_continuation import generate as semantic_continuation
    from .tasks.semantic_to_code import generate as semantic_to_code
    from .tasks.structural_reasoning import generate as structural_reasoning

    if len(units) != len(records):
        raise ValueError("Contextual units and canonical records must have equal length")
    for generator in (
        semantic_continuation,
        semantic_to_code,
        code_to_semantic,
        structural_reasoning,
        mixed_instruction,
    ):
        yield from generator(document, units, records, split=split)


def _normalize(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")
