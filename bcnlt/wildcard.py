from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from .cache import ProjectIndex
from .core import ContextualUnit
from .events import SemanticEvent
from .ir import COMMON_CODING_CONCEPTS, IRKind


class DefinitionState(str, Enum):
    PROVISIONAL = "provisional"
    VALIDATED = "validated"


@dataclass(frozen=True)
class ProvisionalMeaning:
    category: str
    hypotheses: tuple[str, ...]
    evidence: tuple[str, ...]
    confidence: float
    target_event: str | None = None
    target_ir_kind: str | None = None
    evidence_kinds: tuple[str, ...] = ()
    extension_name: str | None = None


@dataclass(frozen=True)
class WildcardMemoryRecord:
    signature: str
    surface: str
    relation: str
    category: str
    hypotheses: tuple[str, ...]
    evidence: tuple[str, ...]
    confidence: float
    seen_count: int = 1
    language: str = "python"
    semantic_spec_sha256: str = "legacy-python"
    definition_state: DefinitionState = DefinitionState.PROVISIONAL
    target_event: str | None = None
    target_ir_kind: str | None = None
    evidence_kinds: tuple[str, ...] = ()
    context_fingerprints: tuple[str, ...] = ()
    document_identities: tuple[str, ...] = ()
    validated_by: str | None = None
    extension_name: str | None = None


class WildcardMemory:
    """Append-only memory for unknown semantic constructs BCNLT has already seen."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        language: str = "python",
        semantic_spec_sha256: str = "legacy-python",
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.language = language
        self.semantic_spec_sha256 = semantic_spec_sha256
        self.records: dict[str, WildcardMemoryRecord] = {}
        self.load_warnings: list[str] = []
        if self.path is not None and self.path.exists():
            self._load()

    def signature_for(self, unit: ContextualUnit) -> str:
        surface = unit.surface.strip().lower()
        relation = (unit.ast_relation or "").strip().lower()
        event = unit.event.strip().upper()
        return f"{self.language}|{self.semantic_spec_sha256}|{event}|{relation}|{surface}"

    def lookup(self, unit: ContextualUnit) -> WildcardMemoryRecord | None:
        return self.records.get(self.signature_for(unit))

    def context_fingerprint_for(
        self,
        unit: ContextualUnit,
        document_identity: str | None = None,
    ) -> str:
        payload = {
            "document": document_identity or "<unknown>",
            "scope": unit.scope,
            "span": unit.span,
            "relation": unit.ast_relation,
            "history": unit.recent_history[-4:],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def remember(
        self,
        unit: ContextualUnit,
        meaning: ProvisionalMeaning,
        *,
        document_identity: str | None = None,
    ) -> WildcardMemoryRecord:
        signature = self.signature_for(unit)
        previous = self.records.get(signature)
        seen_count = 1 if previous is None else previous.seen_count + 1
        evidence = _merge_tuple(previous.evidence if previous else (), meaning.evidence)
        hypotheses = _merge_tuple(previous.hypotheses if previous else (), meaning.hypotheses)
        confidence = max(previous.confidence if previous else 0.0, meaning.confidence)
        definition_state = (
            previous.definition_state if previous else DefinitionState.PROVISIONAL
        )
        targets_agree = previous is None or (
            (meaning.target_event is None or meaning.target_event == previous.target_event)
            and (
                meaning.target_ir_kind is None
                or meaning.target_ir_kind == previous.target_ir_kind
            )
        )
        target_event = meaning.target_event or (previous.target_event if previous else None)
        target_ir_kind = meaning.target_ir_kind or (
            previous.target_ir_kind if previous else None
        )
        extension_name = meaning.extension_name or (
            previous.extension_name if previous else None
        )
        evidence_kinds = _merge_tuple(
            previous.evidence_kinds if previous else (), meaning.evidence_kinds
        )
        context_fingerprints = _merge_tuple(
            previous.context_fingerprints if previous else (),
            (self.context_fingerprint_for(unit, document_identity),),
        )
        document_identities = _merge_tuple(
            previous.document_identities if previous else (),
            (document_identity or "<unknown>",),
        )
        if len(context_fingerprints) >= 2 and targets_agree:
            evidence_kinds = _merge_tuple(evidence_kinds, ("context_agreement",))
        if len(context_fingerprints) >= 3 and targets_agree:
            evidence_kinds = _merge_tuple(evidence_kinds, ("stable_pattern",))
        validated_by = previous.validated_by if previous else None
        required_contexts = 3 if target_ir_kind == IRKind.EXTENSION.value else 2
        if (
            definition_state is DefinitionState.PROVISIONAL
            and targets_agree
            and target_event is not None
            and target_ir_kind is not None
            and len(context_fingerprints) >= required_contexts
            and len(document_identities) >= 2
            and len(evidence_kinds) >= 3
        ):
            definition_state = DefinitionState.VALIDATED
            validated_by = "autonomous"
            confidence = max(confidence, 0.82)
            evidence = _merge_tuple(
                evidence,
                (
                    "autonomous-validation:independent-contexts="
                    f"{len(context_fingerprints)};evidence-kinds={len(evidence_kinds)}",
                ),
            )
        record = WildcardMemoryRecord(
            signature=signature,
            surface=unit.surface,
            relation=unit.ast_relation or "",
            category=meaning.category,
            hypotheses=hypotheses,
            evidence=evidence,
            confidence=confidence,
            seen_count=seen_count,
            language=self.language,
            semantic_spec_sha256=self.semantic_spec_sha256,
            definition_state=definition_state,
            target_event=target_event,
            target_ir_kind=target_ir_kind,
            evidence_kinds=evidence_kinds,
            context_fingerprints=context_fingerprints,
            document_identities=document_identities,
            validated_by=validated_by,
            extension_name=extension_name,
        )
        self._append(record)
        return record

    def validate(
        self,
        unit: ContextualUnit,
        *,
        target_event: str,
        target_ir_kind: str,
        evidence: tuple[str, ...],
        extension_name: str | None = None,
    ) -> WildcardMemoryRecord:
        previous = self.lookup(unit)
        if previous is None:
            raise ValueError("Cannot validate a definition that wildcard memory has not observed")
        event = SemanticEvent(target_event).value
        ir_kind = IRKind(target_ir_kind)
        allowed_pairs = {(item.event, item.ir_kind) for item in COMMON_CODING_CONCEPTS}
        is_named_extension = (
            event == SemanticEvent.UNKNOWN_SEMANTIC.value
            and ir_kind is IRKind.EXTENSION
            and bool(extension_name and extension_name.strip())
        )
        if (event, ir_kind) not in allowed_pairs and not is_named_extension:
            raise ValueError("Validated event and IR kind must name one Common Core concept")
        if not evidence or any(not item.strip() for item in evidence):
            raise ValueError("Validation requires non-empty evidence")
        record = WildcardMemoryRecord(
            signature=previous.signature,
            surface=previous.surface,
            relation=previous.relation,
            category=previous.category,
            hypotheses=previous.hypotheses,
            evidence=_merge_tuple(previous.evidence, evidence),
            confidence=max(0.9, previous.confidence),
            seen_count=previous.seen_count,
            language=previous.language,
            semantic_spec_sha256=previous.semantic_spec_sha256,
            definition_state=DefinitionState.VALIDATED,
            target_event=event,
            target_ir_kind=ir_kind.value,
            evidence_kinds=_merge_tuple(previous.evidence_kinds, ("external_validation",)),
            context_fingerprints=previous.context_fingerprints,
            document_identities=previous.document_identities,
            validated_by="evidence",
            extension_name=extension_name.strip() if extension_name else None,
        )
        self._append(record)
        return record

    def _append(self, record: WildcardMemoryRecord) -> None:
        self.records[record.signature] = record
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_record_to_json(record), sort_keys=True) + "\n")

    def _load(self) -> None:
        if self.path is None:
            return
        lines = [
            (line_number, line)
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            )
            if line.strip()
        ]
        for position, (line_number, line) in enumerate(lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                if position == len(lines) - 1:
                    self.load_warnings.append(
                        f"ignored truncated final learned-memory record at line {line_number}: {exc.msg}"
                    )
                    continue
                raise ValueError(
                    f"Malformed learned-memory record at line {line_number}: {exc.msg}"
                ) from exc
            record = WildcardMemoryRecord(
                signature=str(payload["signature"]),
                surface=str(payload.get("surface", "")),
                relation=str(payload.get("relation", "")),
                category=str(payload.get("category", "extension")),
                hypotheses=tuple(payload.get("hypotheses", ())),
                evidence=tuple(payload.get("evidence", ())),
                confidence=float(payload.get("confidence", 0.0)),
                seen_count=int(payload.get("seen_count", 1)),
                language=str(payload.get("language", self.language)),
                semantic_spec_sha256=str(
                    payload.get("semantic_spec_sha256", self.semantic_spec_sha256)
                ),
                definition_state=DefinitionState(
                    payload.get("definition_state", DefinitionState.PROVISIONAL.value)
                ),
                target_event=payload.get("target_event"),
                target_ir_kind=payload.get("target_ir_kind"),
                evidence_kinds=tuple(payload.get("evidence_kinds", ())),
                context_fingerprints=tuple(payload.get("context_fingerprints", ())),
                document_identities=tuple(payload.get("document_identities", ())),
                validated_by=payload.get("validated_by"),
                extension_name=payload.get("extension_name"),
            )
            previous = self.records.get(record.signature)
            if previous is None:
                self.records[record.signature] = record
                continue
            newer = record
            if (
                previous.definition_state is DefinitionState.VALIDATED
                and record.definition_state is DefinitionState.PROVISIONAL
            ):
                newer = previous
            self.records[record.signature] = WildcardMemoryRecord(
                signature=record.signature,
                surface=newer.surface or previous.surface,
                relation=newer.relation or previous.relation,
                category=newer.category or previous.category,
                hypotheses=_merge_tuple(previous.hypotheses, record.hypotheses),
                evidence=_merge_tuple(previous.evidence, record.evidence),
                confidence=max(previous.confidence, record.confidence),
                seen_count=max(previous.seen_count, record.seen_count),
                language=newer.language,
                semantic_spec_sha256=newer.semantic_spec_sha256,
                definition_state=newer.definition_state,
                target_event=newer.target_event or previous.target_event,
                target_ir_kind=newer.target_ir_kind or previous.target_ir_kind,
                evidence_kinds=_merge_tuple(previous.evidence_kinds, record.evidence_kinds),
                context_fingerprints=_merge_tuple(
                    previous.context_fingerprints, record.context_fingerprints
                ),
                document_identities=_merge_tuple(
                    previous.document_identities, record.document_identities
                ),
                validated_by=newer.validated_by or previous.validated_by,
                extension_name=newer.extension_name or previous.extension_name,
            )


class WildcardResolver:
    """Scientific-method-ish resolver for constructs outside the common core."""

    def __init__(self, memory: WildcardMemory | None = None) -> None:
        self.memory = memory or WildcardMemory()

    def resolve(self, unit: ContextualUnit, index: ProjectIndex) -> ProvisionalMeaning:
        surface = unit.surface.lower()
        relation = (unit.ast_relation or "").lower()
        evidence = [
            f"event={unit.event}",
            f"relation={unit.ast_relation}",
            f"scope={unit.scope}",
            f"history={list(unit.recent_history[-3:])}",
        ]
        remembered = self.memory.lookup(unit)
        if remembered is not None:
            signature = self.memory.signature_for(unit)
            return ProvisionalMeaning(
                category=remembered.category,
                hypotheses=remembered.hypotheses,
                evidence=tuple(
                    evidence
                    + [
                        f"wildcard memory hit:{signature}",
                        f"seen_count={remembered.seen_count}",
                    ]
                    + list(remembered.evidence)
                ),
                confidence=(
                    remembered.confidence
                    if remembered.definition_state is DefinitionState.VALIDATED
                    else min(0.71, max(0.5, remembered.confidence + 0.15))
                ),
                target_event=remembered.target_event,
                target_ir_kind=remembered.target_ir_kind,
                evidence_kinds=remembered.evidence_kinds,
                extension_name=remembered.extension_name,
            )

        inferred = _infer_target(surface, relation)
        if inferred is not None:
            category, target_event, target_ir_kind, matched_signal, evidence_kinds = inferred
            return ProvisionalMeaning(
                category=category,
                hypotheses=(
                    f"construct may project into Common Core {target_ir_kind}",
                    "the projection must agree across independent contexts before reuse",
                ),
                evidence=tuple(evidence + [f"grammar signal:{matched_signal}"]),
                confidence=0.64,
                target_event=target_event,
                target_ir_kind=target_ir_kind,
                evidence_kinds=evidence_kinds,
            )
        if unit.event == "UNKNOWN_SEMANTIC":
            extension_name = _extension_name(self.memory.language, relation)
            return ProvisionalMeaning(
                category="extension",
                hypotheses=("construct is outside the shared Common Coding Core",),
                evidence=tuple(evidence),
                confidence=0.35,
                target_event=SemanticEvent.UNKNOWN_SEMANTIC.value,
                target_ir_kind=IRKind.EXTENSION.value,
                evidence_kinds=("grammar_structure",) if relation else (),
                extension_name=extension_name,
            )
        return ProvisionalMeaning(
            category="control_flow",
            hypotheses=("construct may affect execution order or symbol visibility",),
            evidence=tuple(evidence),
            confidence=0.25,
        )

    def remember(
        self,
        unit: ContextualUnit,
        meaning: ProvisionalMeaning,
        *,
        document_identity: str | None = None,
    ) -> WildcardMemoryRecord:
        return self.memory.remember(
            unit,
            meaning,
            document_identity=document_identity,
        )

    def resolve_and_remember(
        self,
        unit: ContextualUnit,
        index: ProjectIndex,
        *,
        document_identity: str | None = None,
    ) -> ProvisionalMeaning:
        meaning = self.resolve(unit, index)
        self.remember(unit, meaning, document_identity=document_identity)
        return meaning


def _merge_tuple(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for item in (*left, *right):
        if item not in merged:
            merged.append(item)
    return tuple(merged)


def _infer_target(
    surface: str,
    relation: str,
) -> tuple[str, str, str, str, tuple[str, ...]] | None:
    surface_tokens = set(re.findall(r"[a-z]+", surface))
    relation_tokens = set(re.findall(r"[a-z]+", relation))
    tokens = surface_tokens | relation_tokens
    candidates = (
        ({"match", "switch", "case", "branch", "conditional"}, "control_flow", SemanticEvent.BRANCH, IRKind.BRANCH),
        ({"loop", "for", "while", "repeat"}, "control_flow", SemanticEvent.LOOP, IRKind.LOOP),
        ({"return"}, "return", SemanticEvent.RETURN, IRKind.RETURN),
        ({"call", "invoke", "invocation"}, "call", SemanticEvent.CALL, IRKind.CALL),
        ({"assign", "assignment"}, "state_change", SemanticEvent.ASSIGN, IRKind.ASSIGNMENT),
        ({"import", "include", "using"}, "import", SemanticEvent.IMPORT, IRKind.IMPORT),
        ({"raise", "throw", "error"}, "error", SemanticEvent.RAISE, IRKind.ERROR),
        ({"member", "attribute", "field"}, "member_access", SemanticEvent.MEMBER_ACCESS, IRKind.MEMBER_ACCESS),
        ({"operator", "binary", "unary", "comparison"}, "operator", SemanticEvent.OPERATOR, IRKind.OPERATOR),
    )
    for signals, category, event, ir_kind in candidates:
        matched = sorted(tokens.intersection(signals))
        if matched:
            evidence_kinds = []
            if relation_tokens.intersection(signals):
                evidence_kinds.append("grammar_structure")
            if surface_tokens.intersection(signals):
                evidence_kinds.append("lexical_surface")
            return category, event.value, ir_kind.value, matched[0], tuple(evidence_kinds)
    return None


def _extension_name(language: str, relation: str) -> str:
    node_type = relation.split(".", 1)[0] or "unknown"
    normalized = re.sub(r"[^a-z0-9_]+", "_", node_type.casefold()).strip("_")
    return f"{language}.{normalized or 'unknown'}"


def _record_to_json(record: WildcardMemoryRecord) -> dict[str, object]:
    payload = asdict(record)
    payload["definition_state"] = record.definition_state.value
    payload["hypotheses"] = list(record.hypotheses)
    payload["evidence"] = list(record.evidence)
    payload["evidence_kinds"] = list(record.evidence_kinds)
    payload["context_fingerprints"] = list(record.context_fingerprints)
    payload["document_identities"] = list(record.document_identities)
    return payload
