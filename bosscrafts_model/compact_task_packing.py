"""Record-independent task fragments for compact semantic wire V2."""
from __future__ import annotations

import hashlib
import json

from .bcnlt_adapter import BCNLTCanonicalRecord
from .compact_semantic_wire import COMPACT_SEMANTIC_WIRE_SCHEMA, CompactSemanticWireV2
from .current_task_packing import CurrentCorpusTaskPacker
from .jsonl import split_jsonl_records
from .tokenizer import NativeTokenizerBinding


class CompactCurrentCorpusTaskPacker(CurrentCorpusTaskPacker):
    """Pack each canonical record as an independently decodable V2 fragment."""

    def __init__(
        self,
        binding: NativeTokenizerBinding,
        *,
        max_sequence: int = 2048,
    ) -> None:
        super().__init__(binding, max_sequence=max_sequence)
        self._compact_codec = CompactSemanticWireV2(binding)

    @property
    def semantic_wire_schema_version(self) -> str:
        return COMPACT_SEMANTIC_WIRE_SCHEMA

    @staticmethod
    def _record(language: str, canonical: str) -> BCNLTCanonicalRecord:
        try:
            payload = json.loads(canonical)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("canonical task record JSON is invalid") from error
        if type(payload) is not dict:
            raise ValueError("canonical task record must be an object")
        span = payload.get("span")
        if type(span) is not list or len(span) != 4:
            raise ValueError("canonical task record span is invalid")
        return BCNLTCanonicalRecord(
            language=language,
            canonical_json=canonical,
            payload_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            event=str(payload.get("event")),
            surface=str(payload.get("surface")),
            confidence_band=str(payload.get("confidence_band")),
            uncertainty_band=str(payload.get("uncertainty_band")),
            span=tuple(span),  # type: ignore[arg-type]
        )

    def _semantic_record_fragments(self, row: dict[str, object]) -> list[list[int]]:
        canonical_jsonl = row.get("canonical_jsonl")
        language = row.get("language")
        record_count = row.get("semantic_record_count")
        if type(canonical_jsonl) is not str or type(language) is not str:
            raise ValueError("compact task row lacks canonical semantic data")
        lines = split_jsonl_records(canonical_jsonl)
        if type(record_count) is not int or len(lines) != record_count:
            raise ValueError("compact task canonical record count mismatch")
        records = tuple(self._record(language, line) for line in lines)
        fragments: list[list[int]] = []
        for fragment_values in self._compact_codec.encode_independent_fragments(records):
            fragment = list(fragment_values)
            if (
                not fragment
                or fragment[0] != self._id("record:begin")
                or fragment[-1] != self._id("record:end")
            ):
                raise ValueError("compact task record boundary mismatch")
            fragments.append(fragment)
        return fragments
