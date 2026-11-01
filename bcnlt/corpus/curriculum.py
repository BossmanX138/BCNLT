from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from bcnlt.canonical import canonical_record_bytes
from bcnlt.core import ContextualUnit
from bcnlt.semantic_record import SemanticTokenRecord
from bcnlt.uncertainty import KnowledgeBoundaryAssessment

from .database import SourceDocumentRecord


MAX_CURRICULUM_PAIR_CHARACTERS = 16_000
MAX_CONTINUATION_CONTEXT_RECORDS = 8


@dataclass(frozen=True)
class SemanticCodeWindow:
    source: str
    records: tuple[SemanticTokenRecord, ...]
    provenance: tuple[tuple[str, str], ...] = ()


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
    language: str = "python"
    dialect: str | None = None

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
            "language": self.language,
            "dialect": self.dialect,
        }


def canonical_record_stream(records: Sequence[SemanticTokenRecord]) -> str:
    return b"".join(canonical_record_bytes(record) for record in records).decode("utf-8")


def pair_within_character_budget(*parts: str) -> bool:
    return sum(len(part) for part in parts) <= MAX_CURRICULUM_PAIR_CHARACTERS


def semantic_code_windows(
    document: SourceDocumentRecord,
    units: Sequence[ContextualUnit],
    records: Sequence[SemanticTokenRecord],
    *,
    instruction: str,
) -> tuple[SemanticCodeWindow, ...]:
    """Return full-document or top-level parseable semantic/code pairs."""

    if not records:
        return ()
    semantic = canonical_record_stream(records)
    if pair_within_character_budget(instruction, document.content, semantic):
        return (SemanticCodeWindow(document.content, tuple(records)),)
    lines = document.content.splitlines(keepends=True)
    windows: list[SemanticCodeWindow] = []
    candidate_spans = sorted(
        {
            (unit.span[0], unit.span[2])
            for unit in units
            if getattr(unit.event, "value", unit.event) == "SCOPE_ENTER"
            and unit.scope in {"module", "global", "program"}
        }
    )
    if not candidate_spans:
        candidate_spans = sorted(
            {
                (unit.span[0], unit.span[2])
                for unit in units
                if unit.span[0] > 0 and unit.span[2] >= unit.span[0]
            }
        )
    for start, end in candidate_spans:
        if start <= 0 or end < start:
            continue
        source = "".join(lines[start - 1 : end])
        window_records = tuple(
            record
            for unit, record in zip(units, records, strict=True)
            if unit.span[0] >= start and unit.span[2] <= end
        )
        if not window_records:
            continue
        window_semantic = canonical_record_stream(window_records)
        if not pair_within_character_budget(instruction, source, window_semantic):
            continue
        windows.append(
            SemanticCodeWindow(
                source=source,
                records=window_records,
                provenance=(
                    ("source_line_start", str(start)),
                    ("source_line_end", str(end)),
                ),
            )
        )
    return tuple(windows)


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
        "canonical_language": document.canonical_language,
        "repository_identity": document.repository_identity,
        "relative_path": document.relative_path,
        "content_sha256": document.content_sha256,
        "license_spdx": document.license_spdx,
        "ownership": document.ownership,
        "attribution": document.attribution,
        "upstream_url": document.upstream_url,
        "dialect": document.dialect or "",
        "registry_entry_sha256": document.registry_entry_sha256,
        "parser_bundle_sha256": document.parser_bundle_sha256,
        "grammar_revision": document.grammar_revision,
        "semantic_spec_sha256": document.semantic_spec_sha256,
        "extraction_rule_version": document.extraction_rule_version,
    }
    if extra_provenance:
        provenance.update(extra_provenance)
    language_line = f"Language: {document.canonical_language}\n"
    normalized_prompt = _normalize(prompt)
    if not normalized_prompt.startswith(language_line):
        normalized_prompt = language_line + normalized_prompt
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
        "language": document.canonical_language,
        "dialect": document.dialect,
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
        language=document.canonical_language,
        dialect=document.dialect,
    )


def generate_document_curriculum(
    document: SourceDocumentRecord,
    units: Sequence[ContextualUnit],
    records: Sequence[SemanticTokenRecord],
    *,
    split: str = "unassigned",
    assessments: Sequence[KnowledgeBoundaryAssessment] | None = None,
) -> Iterable[CurriculumRecord]:
    from .tasks.code_to_semantic import generate as code_to_semantic
    from .tasks.mixed_instruction import generate as mixed_instruction
    from .tasks.knowledge_boundary import generate as knowledge_boundary
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
    if assessments:
        yield from knowledge_boundary(document, assessments, split=split)


def _normalize(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")
