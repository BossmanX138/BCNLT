"""Measure serialized BCNLT context, rather than equating events with tokens.

This is a payload-size diagnostic, not a semantic-quality or promotion gate.
No model call, source execution, or learning-memory write is performed.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .adapters.python import PythonAdapter
from .core import ContextualUnit


PAYLOAD_FORMAT = "bcnlt-contextual-units-jsonl-v1"
AUGMENTATION_SEPARATOR = "\n\n# BCNLT semantic context\n"


def serialize_units(units: Iterable[ContextualUnit]) -> str:
    """Stable compact JSONL of complete contextual units, including metadata.

    This excludes enrichment added by the CLI (uncertainty, UYR, wildcards).
    It is not a reversible serialization of the original source document.
    """
    return "".join(
        json.dumps(unit.to_dict(), ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), allow_nan=False) + "\n"
        for unit in units
    )


@dataclass(frozen=True)
class PayloadSize:
    characters: int
    utf8_bytes: int
    units: int
    sha256: str


@dataclass(frozen=True)
class PayloadEvalReport:
    path: str
    event_count: int
    tokenizer_name: str
    count_unit: str
    fallback_reason: str | None
    raw_source: PayloadSize
    semantic_only: PayloadSize
    augmented_source: PayloadSize

    def to_dict(self) -> dict[str, object]:
        raw = self.raw_source.units
        return {
            "schema_version": "bcnlt-payload-evaluation-v1",
            "evaluation_kind": "serialized_context_cost_diagnostic",
            "promotion_eligible": False,
            "path": self.path,
            "payload_format": PAYLOAD_FORMAT,
            "digest_scope": "each_exact_string_encoded_as_utf8",
            "event_count": self.event_count,
            "tokenizer_name": self.tokenizer_name,
            "count_unit": self.count_unit,
            "model_token_counts_available": self.count_unit == "model_token",
            "fallback_reason": self.fallback_reason,
            "raw_source": asdict(self.raw_source),
            "semantic_only": asdict(self.semantic_only),
            "augmented_source": asdict(self.augmented_source),
            "semantic_to_raw_ratio": self.semantic_only.units / raw if raw else None,
            "augmented_to_raw_ratio": self.augmented_source.units / raw if raw else None,
            "semantic_minus_raw_units": self.semantic_only.units - raw,
            "augmentation_overhead_units": self.augmented_source.units - raw,
            "augmentation_separator": AUGMENTATION_SEPARATOR,
            "limitations": [
                "Semantic events are not model token IDs.",
                "Semantic-only context is not guaranteed to preserve all source information.",
                "Counts exclude chat wrappers, instructions, outputs, uncertainty, UYR, and wildcard enrichment.",
                "Regex fallback counts are estimates, not model tokens or evidence of token savings.",
                "Payload size does not measure answer quality or justify model or vocabulary promotion.",
            ],
        }


def _load_encoding(name: str):
    import tiktoken  # type: ignore

    return tiktoken.get_encoding(name)


def _count_payloads(texts: tuple[str, ...], encoding_name: str):
    try:
        encoding = _load_encoding(encoding_name)
        # Source code may literally contain strings such as <|endoftext|>.
        # Treat them as ordinary input, not special-token instructions.
        counts = tuple(len(encoding.encode(text, disallowed_special=())) for text in texts)
        return counts, f"tiktoken:{encoding_name}", "model_token", None
    except Exception as error:
        # Recount EVERY payload with one fallback. Mixing successful BPE counts
        # with fallback counts would make the comparison meaningless.
        counts = tuple(len(re.findall(r"\w+|[^\w\s]|\s+", text, flags=re.UNICODE)) for text in texts)
        return counts, "regex-piece-estimate", "regex_piece_estimate", type(error).__name__


def _size(text: str, count: int) -> PayloadSize:
    encoded = text.encode("utf-8")
    return PayloadSize(len(text), len(encoded), count, hashlib.sha256(encoded).hexdigest())


def evaluate_payload_source(
    source: str,
    path: str = "<memory>",
    encoding_name: str = "cl100k_base",
) -> PayloadEvalReport:
    """Compare raw, semantic-only, and raw-plus-semantic context consistently.

    The actual concatenated augmented string is encoded, rather than adding
    separate token counts (BPE token boundaries can change on concatenation).
    """
    adapter = PythonAdapter()
    document = adapter.parse(source, path=path)
    units = list(adapter.to_units(document))
    semantic = serialize_units(units)
    augmented = source + AUGMENTATION_SEPARATOR + semantic if semantic else source
    texts = (source, semantic, augmented)
    counts, tokenizer_name, count_unit, fallback_reason = _count_payloads(texts, encoding_name)
    return PayloadEvalReport(
        path=path,
        event_count=len(units),
        tokenizer_name=tokenizer_name,
        count_unit=count_unit,
        fallback_reason=fallback_reason,
        raw_source=_size(source, counts[0]),
        semantic_only=_size(semantic, counts[1]),
        augmented_source=_size(augmented, counts[2]),
    )
