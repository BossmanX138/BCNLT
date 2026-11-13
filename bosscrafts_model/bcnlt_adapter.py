"""Consumer-side adapter for pinned BCNLT multilingual parser output.

The external parser runtime is not copied into this project. This module only
validates its canonical JSONL output and projects each rich record into the
smaller frozen-tokenizer semantic handoff.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import unicodedata

from .semantic import NativeSemanticRecordCodec, SemanticRecord


PARSER_BUNDLE_SHA256 = "d7df9e034b19a2a459299cecb1e14c31e4bd7b9626247f3ebd56ba258d3fecd6"
EXTRACTION_RULE_VERSION = "bcnlt-multilingual-extraction-v1"
GRAMMAR_PACKAGE_VERSION = "tree-sitter-language-pack@1.22.1"
SOURCE_CANDIDATES_SHA256 = "af8ddc53af38edc3cb8d39e3f61cee01c1c387a449bf8be733365a64525daba4"
LANGUAGE_SPEC_SHA256 = {
    "bash": "2456bb03da817645a32fe197ddd4a7fa1f9d39aa962462ad8ec2d36495105d72",
    "c": "4074bac723d50ecd2046329c537bef89977144935a948fa19828e67ddc6d0deb",
    "c_sharp": "8ea03777924d8ae3a4a6a90396f72fb4027c44528c5fa9e46fc2b790db0676d3",
    "cpp": "342de562401c299eefc900ac9ac69fd4890b59379aa680baf2c0cd7e3f307a",
    "go": "2701f4ebcf1be3d199ade39b26f443d9309d2d03f433e6d4083876cadfa758ee",
    "java": "e21c3fda821a487ef25868175d21a3d1f569c8de362c3ad77efe89b23ade5cf8",
    "javascript": "1fe022c4990ff4293e367bc4761b949ca9c5b5ba56493dec4c4fd06beddc9cb4",
    "kotlin": "1dbcf1aa8d4e9277b60e08790e97e19182cbdab2299125d6fd4d6ced00329547",
    "php": "0aa553d70c7fd1364eb8e884942908828a9b18a1b198f94b4376bf4ab5b5fc54",
    "python": "fe0bb47da50274411299c39d0034eb434c2d36fc1d58d42e4e9b847403b83303",
    "ruby": "9251389d48aa0f94a9ae19d6605b8dc306b2c0f1e834014ae09de6cd37c88148",
    "rust": "345d4da1650dfc4020aed3d86205f428882bb2350974dc6c01bf1a84478deef1",
    "swift": "2cb2c75ffa4bfe53c2936dba00461ccd25fc04f3dd9d63f97b66869fa67ba398",
    "typescript": "45b4d2a264fd9629ae5c6b00567a3d0d523fc8ef13d8646e32e3aa811a76f190",
}

_RECORD_KEYS = frozenset(
    (
        "ast_relation",
        "confidence_band",
        "defines",
        "event",
        "ir",
        "metadata",
        "recent_history",
        "scope",
        "span",
        "surface",
        "symbol_id",
        "uncertainty_band",
    )
)
_IR_KEYS = frozenset(("children", "kind", "metadata", "name", "value"))
_METADATA_REQUIRED_KEYS = frozenset(("knowledge_state", "uncertainty_signals", "uyr_rounds"))
_METADATA_ALLOWED_KEYS = frozenset(
    (
        "dynamic_call",
        "dynamic_binding",
        "macro_expansion",
        "overload_unresolved",
        "preprocessor_condition",
        "unknown_node",
        "surface_truncated",
        "knowledge_state",
        "uyr_rounds",
        "uncertainty_signals",
        "wildcard_category",
        "learned_definition_signature",
        "learned_definition_state",
        "learned_extension_name",
        "extended_core_id",
        "extended_core_signature",
        "extended_core_revision",
        "extended_core_state",
        "extended_core_projection",
        "extended_core_name",
    )
)
_BANDS = frozenset(("low", "medium", "high"))
_KNOWLEDGE_STATES = frozenset(("known", "retrieved", "provisional", "unresolved"))
MAX_CANONICAL_OUTPUT_BYTES = 16 * 1024 * 1024
# Compatibility name retained for older callers; the enforced unit is bytes.
MAX_CANONICAL_OUTPUT_CHARS = MAX_CANONICAL_OUTPUT_BYTES
MAX_CANONICAL_RECORDS = 16_384


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate BCNLT record key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _is_string(value: object, *, allow_empty: bool = False, maximum: int = 4096) -> bool:
    return (
        type(value) is str
        and (allow_empty or bool(value))
        and len(value) <= maximum
        and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
        and "\r" not in value
        and unicodedata.normalize("NFC", value) == value
    )


def _validate_json_value(value: object, *, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("BCNLT JSON metadata nesting is too deep")
    if value is None or type(value) is bool or type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("BCNLT JSON metadata must be finite")
        return
    if type(value) is str:
        if not _is_string(value, allow_empty=True, maximum=8192):
            raise ValueError("BCNLT JSON metadata contains an invalid string")
        return
    if type(value) is list:
        if len(value) > 256:
            raise ValueError("BCNLT JSON metadata sequence is too large")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if type(value) is dict:
        if len(value) > 256:
            raise ValueError("BCNLT JSON metadata object is too large")
        for key, item in value.items():
            if not _is_string(key, maximum=1024):
                raise ValueError("BCNLT JSON metadata key is invalid")
            _validate_json_value(item, depth=depth + 1)
        return
    raise ValueError("BCNLT JSON metadata contains an unsupported value")


def _validate_ir_node(ir: object, *, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("BCNLT IR nesting is too deep")
    if type(ir) is not dict or set(ir) != _IR_KEYS:
        raise ValueError("BCNLT IR keys mismatch")
    if (
        not _is_string(ir["kind"])
        or not _is_string(ir["name"], allow_empty=True, maximum=8192)
        or ir["value"] is not None
        or type(ir["children"]) is not list
        or len(ir["children"]) > 256
        or type(ir["metadata"]) is not dict
    ):
        raise ValueError("BCNLT IR field contract mismatch")
    _validate_json_value(ir["metadata"], depth=depth)
    for child in ir["children"]:
        if type(child) is not dict:
            raise ValueError("BCNLT IR child must be an IR object")
        _validate_ir_node(child, depth=depth + 1)


@dataclass(frozen=True)
class BCNLTParserOutputProvenance:
    language: str
    parser_bundle_sha256: str
    semantic_spec_sha256: str
    grammar_revision: str
    extraction_rule_version: str

    @classmethod
    def expected(cls, language: str) -> "BCNLTParserOutputProvenance":
        if language not in LANGUAGE_SPEC_SHA256:
            raise ValueError("unsupported BCNLT parser language")
        return cls(
            language=language,
            parser_bundle_sha256=PARSER_BUNDLE_SHA256,
            semantic_spec_sha256=LANGUAGE_SPEC_SHA256[language],
            grammar_revision=f"{GRAMMAR_PACKAGE_VERSION}/{language}",
            extraction_rule_version=EXTRACTION_RULE_VERSION,
        )


@dataclass(frozen=True)
class BCNLTCanonicalRecord:
    language: str
    canonical_json: str
    payload_sha256: str
    event: str
    surface: str
    confidence_band: str
    uncertainty_band: str
    span: tuple[int, int, int, int]


def validate_canonical_record_payload(
    payload: object,
    *,
    allowed_events: frozenset[str],
) -> None:
    """Validate one decoded record independently of parser provenance."""
    if type(payload) is not dict or set(payload) != _RECORD_KEYS:
        raise ValueError("BCNLT canonical record keys mismatch")
    if payload["event"] not in allowed_events:
        raise ValueError("BCNLT event is not in the frozen event vocabulary")
    if not _is_string(payload["surface"]):
        raise ValueError("BCNLT surface must be a bounded non-empty string")
    for field in ("ast_relation", "scope"):
        if not _is_string(payload[field], maximum=8192):
            raise ValueError(f"BCNLT {field} must be a bounded non-empty string")
    for field in ("defines", "symbol_id"):
        if payload[field] is not None and not _is_string(payload[field], maximum=8192):
            raise ValueError(f"BCNLT {field} must be null or a bounded string")
    if payload["confidence_band"] not in _BANDS or payload["uncertainty_band"] not in _BANDS:
        raise ValueError("BCNLT confidence and uncertainty bands are invalid")

    span = payload["span"]
    if (
        type(span) is not list
        or len(span) != 4
        or any(type(value) is not int or value < 0 for value in span)
        or (span[0], span[1]) > (span[2], span[3])
    ):
        raise ValueError("BCNLT span must be four ordered non-negative integers")
    recent = payload["recent_history"]
    if (
        type(recent) is not list
        or len(recent) > 8
        or any(not _is_string(value, allow_empty=True, maximum=512) for value in recent)
    ):
        raise ValueError("BCNLT recent_history is invalid")

    _validate_ir_node(payload["ir"])

    metadata = payload["metadata"]
    if (
        type(metadata) is not dict
        or not _METADATA_REQUIRED_KEYS.issubset(metadata)
        or not set(metadata).issubset(_METADATA_ALLOWED_KEYS)
    ):
        raise ValueError("BCNLT metadata keys mismatch")
    if metadata["knowledge_state"] not in _KNOWLEDGE_STATES:
        raise ValueError("BCNLT knowledge_state is invalid")
    signals = metadata["uncertainty_signals"]
    if (
        type(signals) is not list
        or len(signals) > 16
        or any(not _is_string(value, maximum=1024) for value in signals)
        or type(metadata["uyr_rounds"]) is not int
        or not 0 <= metadata["uyr_rounds"] <= 4
    ):
        raise ValueError("BCNLT uncertainty metadata is invalid")
    for field in (
        "dynamic_call",
        "dynamic_binding",
        "macro_expansion",
        "overload_unresolved",
        "surface_truncated",
    ):
        if field in metadata and type(metadata[field]) is not bool:
            raise ValueError(f"BCNLT {field} metadata must be boolean")
    for field in (
        "preprocessor_condition",
        "unknown_node",
        "wildcard_category",
        "learned_definition_signature",
        "learned_extension_name",
        "extended_core_id",
        "extended_core_signature",
        "extended_core_name",
    ):
        if field in metadata and not _is_string(metadata[field], maximum=8192):
            raise ValueError(f"BCNLT {field} metadata must be a bounded string")
    for field in ("learned_definition_state", "extended_core_state"):
        if field in metadata and metadata[field] != "validated":
            raise ValueError(f"BCNLT {field} metadata must be validated")
    if "extended_core_revision" in metadata and (
        type(metadata["extended_core_revision"]) is not int
        or metadata["extended_core_revision"] < 1
    ):
        raise ValueError("BCNLT extended_core_revision must be a positive integer")
    if "extended_core_projection" in metadata:
        projection = metadata["extended_core_projection"]
        if (
            type(projection) is not dict
            or set(projection) != {"event", "ir_kind"}
            or projection["event"] not in allowed_events
            or not _is_string(projection["ir_kind"])
        ):
            raise ValueError("BCNLT extended_core_projection metadata is invalid")


class BCNLTMultilingualOutputAdapter:
    """Validate real canonical JSONL and project it into the native codec."""

    def __init__(
        self,
        semantic_codec: NativeSemanticRecordCodec,
        *,
        runtime_spec_sha256: dict[str, str] | None = None,
    ):
        if type(semantic_codec) is not NativeSemanticRecordCodec:
            raise TypeError("semantic_codec must be a NativeSemanticRecordCodec")
        self._semantic_codec = semantic_codec
        self._runtime_spec_sha256 = (
            dict(runtime_spec_sha256) if runtime_spec_sha256 is not None else None
        )
        if self._runtime_spec_sha256 is not None:
            if set(self._runtime_spec_sha256) != set(LANGUAGE_SPEC_SHA256):
                raise ValueError("runtime semantic spec inventory mismatch")
            if any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self._runtime_spec_sha256.values()
            ):
                raise ValueError("runtime semantic spec hashes are invalid")

    @property
    def consumer_contract_integrated(self) -> bool:
        return True

    @property
    def parser_execution_integrated(self) -> bool:
        return False

    @property
    def language_embedded_in_output(self) -> bool:
        return False

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset(LANGUAGE_SPEC_SHA256)

    def _validate_provenance(self, provenance: BCNLTParserOutputProvenance) -> None:
        if type(provenance) is not BCNLTParserOutputProvenance:
            raise TypeError("provenance must be BCNLTParserOutputProvenance")
        if self._runtime_spec_sha256 is None:
            expected = BCNLTParserOutputProvenance.expected(provenance.language)
        else:
            try:
                semantic_spec_sha256 = self._runtime_spec_sha256[provenance.language]
            except KeyError as error:
                raise ValueError("unsupported BCNLT parser language") from error
            expected = BCNLTParserOutputProvenance(
                language=provenance.language,
                parser_bundle_sha256=PARSER_BUNDLE_SHA256,
                semantic_spec_sha256=semantic_spec_sha256,
                grammar_revision=f"{GRAMMAR_PACKAGE_VERSION}/{provenance.language}",
                extraction_rule_version=EXTRACTION_RULE_VERSION,
            )
        if provenance != expected:
            raise ValueError("BCNLT parser provenance mismatch")

    def _validate_record(self, payload: object) -> None:
        validate_canonical_record_payload(
            payload,
            allowed_events=self._semantic_codec.allowed_events,
        )

    def parse_jsonl(
        self,
        output: str,
        provenance: BCNLTParserOutputProvenance,
    ) -> tuple[BCNLTCanonicalRecord, ...]:
        self._validate_provenance(provenance)
        if type(output) is not str:
            raise TypeError("BCNLT parser output must be a string")
        try:
            output_bytes = output.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("BCNLT parser output contains invalid Unicode") from error
        if not output or len(output_bytes) > MAX_CANONICAL_OUTPUT_CHARS or not output.endswith("\n"):
            raise ValueError("BCNLT parser output must be bounded newline-terminated JSONL")
        lines = output[:-1].split("\n")
        if not lines or len(lines) > MAX_CANONICAL_RECORDS or any(not line for line in lines):
            raise ValueError("BCNLT parser output contains an invalid record count or blank line")
        records = []
        for line in lines:
            if len(line) > 65_536:
                raise ValueError("BCNLT canonical record is too large")
            try:
                payload = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_nonfinite,
                )
            except (ValueError, RecursionError) as error:
                if "non-finite JSON constant" in str(error):
                    raise
                raise ValueError("invalid BCNLT canonical JSON") from error
            except TypeError as error:
                raise ValueError("invalid BCNLT canonical JSON") from error
            self._validate_record(payload)
            canonical = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if canonical != line:
                raise ValueError("BCNLT record is not canonical JSON")
            records.append(
                BCNLTCanonicalRecord(
                    language=provenance.language,
                    canonical_json=canonical,
                    payload_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    event=payload["event"],
                    surface=payload["surface"],
                    confidence_band=payload["confidence_band"],
                    uncertainty_band=payload["uncertainty_band"],
                    span=tuple(payload["span"]),
                )
            )
        return tuple(records)

    def project(self, record: BCNLTCanonicalRecord) -> SemanticRecord:
        if type(record) is not BCNLTCanonicalRecord:
            raise TypeError("record must be a BCNLTCanonicalRecord")
        return SemanticRecord(
            event=record.event,
            surface=record.surface,
            confidence_band=record.confidence_band,
        )

    def encode_handoff(self, record: BCNLTCanonicalRecord) -> list[int]:
        return self._semantic_codec.encode(self.project(record))
