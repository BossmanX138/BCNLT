from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Literal

from .cache import ProjectIndex
from .core import ContextualUnit
from .events import SemanticEvent
from .extended_core import (
    ExtendedCoreEntry,
    ExtendedCoreLibrary,
    extended_core_id,
    optional_common_core_projection,
)
from .ir import COMMON_CODING_CONCEPTS, IRKind
from .scientific_learning import (
    VERIFIER_ID,
    VERIFIER_VERSION,
    BoundedProjectionVerifier,
    CodingExperimentFixture,
    HypothesisProjection,
    ScientificExperimentEvidence,
    VerificationStatus,
)


MAX_EVIDENCE_ITEMS = 64
MAX_HYPOTHESIS_ITEMS = 16
MAX_EVIDENCE_KIND_ITEMS = 16
MAX_CONTEXT_FINGERPRINTS = 64
MAX_DOCUMENT_IDENTITIES = 64
MAX_SCIENTIFIC_EXPERIMENTS = 256


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
class CodingOutcomeEvidence:
    context_fingerprint: str
    document_identity: str
    target_event: str
    target_ir_kind: str
    verification_oracle: str
    result_fingerprint: str
    correct: bool
    evidence: tuple[str, ...] = ()
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
    coding_outcomes: tuple[CodingOutcomeEvidence, ...] = ()
    learning_schema_version: int = 1
    hypothesis_revision: int = 1
    hypothesis_fingerprint: str | None = None
    validated_domain: str | None = None
    scientific_experiments: tuple[ScientificExperimentEvidence, ...] = ()


class WildcardMemory:
    """Persistent memory for unknown semantic constructs BCNLT has already seen."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        language: str = "python",
        semantic_spec_sha256: str = "legacy-python",
        storage: Literal["jsonl", "sqlite"] = "jsonl",
        minimum_reproductions: int = 3,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.language = language
        self.semantic_spec_sha256 = semantic_spec_sha256
        self.storage = storage
        self.read_only = read_only
        if (
            not isinstance(minimum_reproductions, int)
            or isinstance(minimum_reproductions, bool)
            or minimum_reproductions < 2
        ):
            raise ValueError("Scientific learning requires at least two reproductions")
        self.minimum_reproductions = minimum_reproductions
        self.records: dict[str, WildcardMemoryRecord] = {}
        self.load_warnings: list[str] = []
        self._truncated_tail_offset: int | None = None
        self._sqlite: sqlite3.Connection | None = None
        self._sqlite_pending = 0
        if storage == "sqlite":
            if self.path is None:
                raise ValueError("SQLite wildcard memory requires a path")
            self._open_sqlite()
        elif storage != "jsonl":
            raise ValueError(f"Unsupported wildcard-memory storage: {storage}")
        elif self.path is not None and self.path.exists():
            self._load()

    def signature_for(self, unit: ContextualUnit) -> str:
        relation = (unit.ast_relation or "").strip().lower()
        event = unit.event.strip().upper()
        surface = unit.surface.strip().lower()
        if event == SemanticEvent.UNKNOWN_SEMANTIC.value:
            surface = str(unit.metadata.get("unknown_node") or relation or surface).casefold()
        return f"{self.language}|{self.semantic_spec_sha256}|{event}|{relation}|{surface}"

    def lookup(self, unit: ContextualUnit) -> WildcardMemoryRecord | None:
        return self._lookup_signature(self.signature_for(unit))

    def validated_extended_core(self) -> ExtendedCoreLibrary:
        if self._sqlite is None:
            records = tuple(self.records.values())
        else:
            rows = self._sqlite.execute(
                "SELECT payload FROM wildcard_definitions ORDER BY signature"
            ).fetchall()
            records = tuple(
                _record_from_json(
                    json.loads(str(row[0])),
                    minimum_reproductions=self.minimum_reproductions,
                )
                for row in rows
            )
        return ExtendedCoreLibrary(
            entry
            for record in records
            if (entry := _extended_core_entry(record)) is not None
        )

    def context_fingerprint_for(
        self,
        unit: ContextualUnit,
        document_identity: str | None = None,
    ) -> str:
        payload = {
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
        if self.read_only:
            raise RuntimeError("Frozen wildcard memory is read-only")
        signature = self.signature_for(unit)
        previous = self._lookup_signature(signature)
        seen_count = 1 if previous is None else previous.seen_count + 1
        evidence = _merge_tuple(
            previous.evidence if previous else (),
            meaning.evidence,
            limit=MAX_EVIDENCE_ITEMS,
        )
        hypotheses = _merge_tuple(
            previous.hypotheses if previous else (),
            meaning.hypotheses,
            limit=MAX_HYPOTHESIS_ITEMS,
        )
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
            previous.evidence_kinds if previous else (),
            meaning.evidence_kinds,
            limit=MAX_EVIDENCE_KIND_ITEMS,
        )
        context_fingerprints = _merge_tuple(
            previous.context_fingerprints if previous else (),
            (self.context_fingerprint_for(unit, document_identity),),
            limit=MAX_CONTEXT_FINGERPRINTS,
        )
        document_identities = _merge_tuple(
            previous.document_identities if previous else (),
            (document_identity or "<unknown>",),
            limit=MAX_DOCUMENT_IDENTITIES,
        )
        if len(context_fingerprints) >= 2 and targets_agree:
            evidence_kinds = _merge_tuple(
                evidence_kinds,
                ("context_agreement",),
                limit=MAX_EVIDENCE_KIND_ITEMS,
            )
        if len(context_fingerprints) >= 3 and targets_agree:
            evidence_kinds = _merge_tuple(
                evidence_kinds,
                ("stable_pattern",),
                limit=MAX_EVIDENCE_KIND_ITEMS,
            )
        validated_by = previous.validated_by if previous else None
        hypothesis_revision = previous.hypothesis_revision if previous else 1
        projection_changed = previous is not None and (
            target_event,
            target_ir_kind,
            extension_name,
        ) != (
            previous.target_event,
            previous.target_ir_kind,
            previous.extension_name,
        )
        if projection_changed:
            hypothesis_revision += 1
        if not targets_agree:
            definition_state = DefinitionState.PROVISIONAL
            validated_by = None
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
            coding_outcomes=previous.coding_outcomes if previous else (),
            hypothesis_revision=hypothesis_revision,
            hypothesis_fingerprint=_hypothesis_fingerprint(
                signature=signature,
                revision=hypothesis_revision,
                target_event=target_event,
                target_ir_kind=target_ir_kind,
                language=self.language,
                semantic_spec_sha256=self.semantic_spec_sha256,
                extension_name=extension_name,
            ),
            validated_domain=(
                None if projection_changed else (previous.validated_domain if previous else None)
            ),
            scientific_experiments=(
                previous.scientific_experiments if previous else ()
            ),
        )
        record = _derive_scientific_state(record, self.minimum_reproductions)
        self._append(record)
        return record

    def run_scientific_experiment(
        self,
        unit: ContextualUnit,
        *,
        source: str,
        fixture: CodingExperimentFixture,
        negative_control: HypothesisProjection,
        verifier: BoundedProjectionVerifier | None = None,
    ) -> WildcardMemoryRecord:
        if self.read_only:
            raise RuntimeError("Frozen wildcard memory is read-only")
        previous = self.lookup(unit)
        if previous is None:
            raise ValueError("Cannot test a hypothesis that wildcard memory has not observed")
        if previous.target_event is None or previous.target_ir_kind is None:
            raise ValueError("Wildcard hypothesis has no testable Common Core projection")
        if fixture.document_identity not in previous.document_identities:
            raise ValueError("Experiment fixture document identity was not independently observed")
        context_fingerprint = self.context_fingerprint_for(unit, fixture.document_identity)
        if context_fingerprint not in previous.context_fingerprints:
            raise ValueError("Experiment fixture does not match an observed wildcard context")
        candidate = HypothesisProjection(
            signature=previous.signature,
            revision=previous.hypothesis_revision,
            target_event=previous.target_event,
            target_ir_kind=previous.target_ir_kind,
            language=previous.language,
            semantic_spec_sha256=previous.semantic_spec_sha256,
            extension_name=previous.extension_name,
        )
        if (
            negative_control.signature != candidate.signature
            or negative_control.revision != candidate.revision
            or negative_control.language != candidate.language
            or negative_control.semantic_spec_sha256 != candidate.semantic_spec_sha256
        ):
            raise ValueError("Negative control must use the current hypothesis revision and lineage")
        result = (verifier or BoundedProjectionVerifier()).verify(
            candidate,
            fixture,
            source=source,
            context_fingerprint=context_fingerprint,
            negative_control=negative_control,
        )
        experiments = _merge_scientific_experiments(
            previous.scientific_experiments,
            (result,),
        )
        record = replace(
            previous,
            definition_state=DefinitionState.PROVISIONAL,
            validated_by=None,
            validated_domain=None,
            confidence=min(0.71, previous.confidence),
            evidence_kinds=_merge_tuple(
                previous.evidence_kinds,
                ("scientific_experiment",),
                limit=MAX_EVIDENCE_KIND_ITEMS,
            ),
            scientific_experiments=experiments,
        )
        record = _derive_scientific_state(record, self.minimum_reproductions)
        self._append(record)
        return record

    def validate(
        self,
        unit: ContextualUnit,
        *,
        target_event: str,
        target_ir_kind: str,
        evidence: tuple[str, ...],
        document_identity: str,
        verification_oracle: str,
        result_fingerprint: str,
        extension_name: str | None = None,
    ) -> WildcardMemoryRecord:
        raise RuntimeError(
            "Direct metadata-only validation is disabled; use the executable bounded "
            "run_scientific_experiment path for promotion, or record_coding_outcome "
            "only to retain provisional caller-reported evidence"
        )

    def record_coding_outcome(
        self,
        unit: ContextualUnit,
        *,
        target_event: str,
        target_ir_kind: str,
        evidence: tuple[str, ...],
        document_identity: str,
        verification_oracle: str,
        result_fingerprint: str,
        correct: bool,
        extension_name: str | None = None,
    ) -> WildcardMemoryRecord:
        if self.read_only:
            raise RuntimeError("Frozen wildcard memory is read-only")
        previous = self.lookup(unit)
        if previous is None:
            raise ValueError("Cannot verify a definition that wildcard memory has not observed")
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
        if not document_identity.strip():
            raise ValueError("Validation requires a document identity")
        if not verification_oracle.strip():
            raise ValueError("Validation requires the independent verification oracle")
        if not result_fingerprint.strip():
            raise ValueError("Validation requires a coding-result fingerprint")
        if type(correct) is not bool:
            raise TypeError("correct must be a real boolean")
        context_fingerprint = self.context_fingerprint_for(unit, document_identity)
        if (
            context_fingerprint not in previous.context_fingerprints
            or document_identity not in previous.document_identities
        ):
            raise ValueError(
                "Coding outcomes must correspond to an independently observed wildcard context"
            )
        normalized_extension = extension_name.strip() if extension_name else None
        outcome = CodingOutcomeEvidence(
            context_fingerprint=context_fingerprint,
            document_identity=document_identity,
            target_event=event,
            target_ir_kind=ir_kind.value,
            verification_oracle=verification_oracle.strip(),
            result_fingerprint=result_fingerprint.strip(),
            correct=correct,
            evidence=evidence,
            extension_name=normalized_extension,
        )
        outcomes = _merge_outcomes(previous.coding_outcomes, (outcome,))
        latest_by_context = {item.context_fingerprint: item for item in outcomes}
        expected_extension = normalized_extension or previous.extension_name
        agreed = [
            item
            for item in latest_by_context.values()
            if item.correct
            and item.target_event == previous.target_event
            and item.target_ir_kind == previous.target_ir_kind
            and (
                item.target_ir_kind != IRKind.EXTENSION.value
                or item.extension_name == expected_extension
            )
        ]
        required_contexts = 3 if previous.target_ir_kind == IRKind.EXTENSION.value else 2
        reproduced = (
            len(agreed) >= required_contexts
            and len({item.document_identity for item in agreed}) >= 2
            and len(agreed) == len(latest_by_context)
        )
        evidence_kinds = _merge_tuple(
            previous.evidence_kinds,
            ("coding_outcome_observation",),
            limit=MAX_EVIDENCE_KIND_ITEMS,
        )
        if reproduced:
            evidence_kinds = _merge_tuple(
                evidence_kinds,
                ("reproduced_claim_pending_verifier",),
                limit=MAX_EVIDENCE_KIND_ITEMS,
            )
        record = WildcardMemoryRecord(
            signature=previous.signature,
            surface=previous.surface,
            relation=previous.relation,
            category=previous.category,
            hypotheses=previous.hypotheses,
            evidence=_merge_tuple(evidence, previous.evidence, limit=MAX_EVIDENCE_ITEMS),
            confidence=min(0.71, previous.confidence),
            seen_count=previous.seen_count,
            language=previous.language,
            semantic_spec_sha256=previous.semantic_spec_sha256,
            definition_state=DefinitionState.PROVISIONAL,
            target_event=previous.target_event,
            target_ir_kind=previous.target_ir_kind,
            evidence_kinds=evidence_kinds,
            context_fingerprints=previous.context_fingerprints,
            document_identities=previous.document_identities,
            validated_by=None,
            extension_name=expected_extension,
            coding_outcomes=outcomes,
            learning_schema_version=previous.learning_schema_version,
            hypothesis_revision=previous.hypothesis_revision,
            hypothesis_fingerprint=previous.hypothesis_fingerprint,
            validated_domain=None,
            scientific_experiments=previous.scientific_experiments,
        )
        self._append(record)
        return record

    def _append(self, record: WildcardMemoryRecord) -> None:
        if self._sqlite is not None:
            payload = json.dumps(
                _record_to_json(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self._sqlite.execute(
                "INSERT INTO wildcard_definitions(signature, payload) VALUES (?, ?) "
                "ON CONFLICT(signature) DO UPDATE SET payload = excluded.payload",
                (record.signature, payload),
            )
            self._sqlite_pending += 1
            if self._sqlite_pending >= 1_000:
                self.flush()
            return
        self.records[record.signature] = record
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._truncated_tail_offset is not None:
                with self.path.open("r+b") as handle:
                    handle.truncate(self._truncated_tail_offset)
                self._truncated_tail_offset = None
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_record_to_json(record), sort_keys=True) + "\n")

    def _lookup_signature(self, signature: str) -> WildcardMemoryRecord | None:
        if self._sqlite is None:
            return self.records.get(signature)
        row = self._sqlite.execute(
            "SELECT payload FROM wildcard_definitions WHERE signature = ?",
            (signature,),
        ).fetchone()
        if row is None:
            return None
        return _record_from_json(
            json.loads(str(row[0])),
            minimum_reproductions=self.minimum_reproductions,
        )

    def _open_sqlite(self) -> None:
        assert self.path is not None
        if self.read_only:
            if not self.path.is_file():
                return
            connection = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro&immutable=1",
                uri=True,
            )
            metadata = {
                "schema_version": "bcnlt-wildcard-sqlite-v1",
                "language": self.language,
                "semantic_spec_sha256": self.semantic_spec_sha256,
            }
            existing = dict(connection.execute("SELECT key, value FROM wildcard_metadata"))
            if existing != metadata:
                connection.close()
                raise ValueError("SQLite wildcard memory metadata does not match this cartridge")
            self._sqlite = connection
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS wildcard_metadata ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS wildcard_definitions ("
            "signature TEXT PRIMARY KEY, payload TEXT NOT NULL) WITHOUT ROWID"
        )
        metadata = {
            "schema_version": "bcnlt-wildcard-sqlite-v1",
            "language": self.language,
            "semantic_spec_sha256": self.semantic_spec_sha256,
        }
        existing = dict(connection.execute("SELECT key, value FROM wildcard_metadata"))
        if existing and existing != metadata:
            connection.close()
            raise ValueError("SQLite wildcard memory metadata does not match this cartridge")
        connection.executemany(
            "INSERT OR REPLACE INTO wildcard_metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        connection.commit()
        connection.execute("BEGIN")
        self._sqlite = connection

    def flush(self) -> None:
        if self._sqlite is None or self.read_only:
            return
        self._sqlite.commit()
        self._sqlite.execute("BEGIN")
        self._sqlite_pending = 0

    def close(self) -> None:
        if self._sqlite is None:
            return
        connection = self._sqlite
        self._sqlite = None
        if self.read_only:
            connection.close()
            return
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        self._sqlite_pending = 0

    def _load(self) -> None:
        if self.path is None:
            return
        raw_lines = self.path.read_bytes().decode("utf-8").splitlines(keepends=True)
        lines: list[tuple[int, str, int]] = []
        byte_offset = 0
        for line_number, raw_line in enumerate(raw_lines, start=1):
            line = raw_line.rstrip("\r\n")
            if line.strip():
                lines.append((line_number, line, byte_offset))
            byte_offset += len(raw_line.encode("utf-8"))
        for position, (line_number, line, line_offset) in enumerate(lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                if position == len(lines) - 1:
                    self._truncated_tail_offset = line_offset
                    self.load_warnings.append(
                        f"ignored truncated final learned-memory record at line {line_number}: {exc.msg}"
                    )
                    continue
                raise ValueError(
                    f"Malformed learned-memory record at line {line_number}: {exc.msg}"
                ) from exc
            record = _record_from_json(
                payload,
                minimum_reproductions=self.minimum_reproductions,
            )
            previous = self.records.get(record.signature)
            if previous is None:
                self.records[record.signature] = record
                continue
            if record.hypothesis_revision < previous.hypothesis_revision:
                raise ValueError(
                    "Learned-memory hypothesis revision rollback at line "
                    f"{line_number}: {record.hypothesis_revision} < "
                    f"{previous.hypothesis_revision}"
                )
            if (
                record.hypothesis_revision == previous.hypothesis_revision
                and record.hypothesis_fingerprint != previous.hypothesis_fingerprint
            ):
                raise ValueError(
                    "Conflicting learned-memory hypothesis at the same revision "
                    f"on line {line_number}"
                )
            newer = record
            merged = WildcardMemoryRecord(
                signature=record.signature,
                surface=newer.surface or previous.surface,
                relation=newer.relation or previous.relation,
                category=newer.category or previous.category,
                hypotheses=_merge_tuple(
                    previous.hypotheses,
                    record.hypotheses,
                    limit=MAX_HYPOTHESIS_ITEMS,
                ),
                evidence=_merge_tuple(
                    previous.evidence,
                    record.evidence,
                    limit=MAX_EVIDENCE_ITEMS,
                ),
                confidence=max(previous.confidence, record.confidence),
                seen_count=max(previous.seen_count, record.seen_count),
                language=newer.language,
                semantic_spec_sha256=newer.semantic_spec_sha256,
                definition_state=newer.definition_state,
                target_event=newer.target_event or previous.target_event,
                target_ir_kind=newer.target_ir_kind or previous.target_ir_kind,
                evidence_kinds=_merge_tuple(
                    previous.evidence_kinds,
                    record.evidence_kinds,
                    limit=MAX_EVIDENCE_KIND_ITEMS,
                ),
                context_fingerprints=_merge_tuple(
                    previous.context_fingerprints,
                    record.context_fingerprints,
                    limit=MAX_CONTEXT_FINGERPRINTS,
                ),
                document_identities=_merge_tuple(
                    previous.document_identities,
                    record.document_identities,
                    limit=MAX_DOCUMENT_IDENTITIES,
                ),
                validated_by=newer.validated_by,
                extension_name=newer.extension_name or previous.extension_name,
                coding_outcomes=_merge_outcomes(
                    previous.coding_outcomes, record.coding_outcomes
                ),
                learning_schema_version=newer.learning_schema_version,
                hypothesis_revision=newer.hypothesis_revision,
                hypothesis_fingerprint=newer.hypothesis_fingerprint,
                validated_domain=newer.validated_domain,
                scientific_experiments=_merge_scientific_experiments(
                    previous.scientific_experiments,
                    record.scientific_experiments,
                ),
            )
            self.records[record.signature] = _derive_scientific_state(
                merged, self.minimum_reproductions
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


def _extended_core_entry(record: WildcardMemoryRecord) -> ExtendedCoreEntry | None:
    if (
        record.definition_state is not DefinitionState.VALIDATED
        or record.hypothesis_fingerprint is None
        or record.target_event is None
        or record.target_ir_kind is None
        or record.validated_domain is None
    ):
        return None
    return ExtendedCoreEntry(
        extended_core_id=extended_core_id(
            language=record.language,
            signature=record.signature,
            revision=record.hypothesis_revision,
        ),
        signature=record.signature,
        revision=record.hypothesis_revision,
        language=record.language,
        semantic_spec_sha256=record.semantic_spec_sha256,
        surface=record.surface,
        relation=record.relation,
        category=record.category,
        hypotheses=record.hypotheses,
        evidence=record.evidence,
        confidence=record.confidence,
        validated_domain=record.validated_domain,
        common_core_projection=optional_common_core_projection(
            event=record.target_event,
            ir_kind=record.target_ir_kind,
        ),
        evidence_sha256=tuple(
            item.evidence_sha256
            for item in record.scientific_experiments
            if item.hypothesis_fingerprint == record.hypothesis_fingerprint
            and item.status is VerificationStatus.PASSED
        ),
        extension_name=record.extension_name,
    )


def _merge_tuple(
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in (*left, *right):
        if item not in seen:
            merged.append(item)
            seen.add(item)
            if limit is not None and len(merged) >= limit:
                break
    return tuple(merged)


def _merge_outcomes(
    left: tuple[CodingOutcomeEvidence, ...],
    right: tuple[CodingOutcomeEvidence, ...],
) -> tuple[CodingOutcomeEvidence, ...]:
    merged: list[CodingOutcomeEvidence] = []
    for item in (*left, *right):
        if item not in merged:
            merged.append(item)
    return tuple(merged)


def _merge_scientific_experiments(
    left: tuple[ScientificExperimentEvidence, ...],
    right: tuple[ScientificExperimentEvidence, ...],
) -> tuple[ScientificExperimentEvidence, ...]:
    merged: list[ScientificExperimentEvidence] = []
    seen: set[str] = set()
    for item in (*left, *right):
        if item.evidence_sha256 in seen:
            continue
        seen.add(item.evidence_sha256)
        merged.append(item)
    return tuple(merged[-MAX_SCIENTIFIC_EXPERIMENTS:])


def _hypothesis_fingerprint(
    *,
    signature: str,
    revision: int,
    target_event: str | None,
    target_ir_kind: str | None,
    language: str,
    semantic_spec_sha256: str,
    extension_name: str | None,
) -> str | None:
    if target_event is None or target_ir_kind is None:
        return None
    return HypothesisProjection(
        signature=signature,
        revision=revision,
        target_event=target_event,
        target_ir_kind=target_ir_kind,
        language=language,
        semantic_spec_sha256=semantic_spec_sha256,
        extension_name=extension_name,
    ).fingerprint


def _derive_scientific_state(
    record: WildcardMemoryRecord,
    minimum_reproductions: int,
) -> WildcardMemoryRecord:
    evidence_kinds = tuple(
        item
        for item in record.evidence_kinds
        if item not in {"scientifically_reproduced", "scientific_contradiction"}
    )
    current = tuple(
        item
        for item in record.scientific_experiments
        if item.hypothesis_fingerprint == record.hypothesis_fingerprint
        and item.hypothesis_revision == record.hypothesis_revision
        and item.language == record.language
        and item.semantic_spec_sha256 == record.semantic_spec_sha256
        and item.verifier_id == VERIFIER_ID
        and item.verifier_version == VERIFIER_VERSION
    )
    if not current:
        return replace(
            record,
            definition_state=DefinitionState.PROVISIONAL,
            validated_by=None,
            validated_domain=None,
            confidence=min(0.71, record.confidence),
            evidence_kinds=evidence_kinds,
        )
    if any(item.status is not VerificationStatus.PASSED for item in current):
        return replace(
            record,
            definition_state=DefinitionState.PROVISIONAL,
            validated_by=None,
            validated_domain=None,
            confidence=min(0.71, record.confidence),
            evidence_kinds=_merge_tuple(
                evidence_kinds,
                ("scientific_contradiction",),
                limit=MAX_EVIDENCE_KIND_ITEMS,
            ),
        )
    by_domain: dict[str, list[ScientificExperimentEvidence]] = {}
    for item in current:
        by_domain.setdefault(item.domain, []).append(item)
    for domain, items in by_domain.items():
        unique = {item.independence_key: item for item in items}.values()
        if (
            len(unique) >= minimum_reproductions
            and len({item.context_fingerprint for item in unique}) >= minimum_reproductions
            and len({item.document_identity for item in unique}) >= minimum_reproductions
            and len({item.stimulus_sha256 for item in unique}) >= minimum_reproductions
            and len({item.source_sha256 for item in unique}) >= 2
            and all(item.intended_passed and item.negative_control_rejected for item in unique)
        ):
            return replace(
                record,
                definition_state=DefinitionState.VALIDATED,
                validated_by=f"{VERIFIER_ID}:{VERIFIER_VERSION}",
                validated_domain=domain,
                confidence=max(0.9, record.confidence),
                evidence_kinds=_merge_tuple(
                    evidence_kinds,
                    ("scientifically_reproduced",),
                    limit=MAX_EVIDENCE_KIND_ITEMS,
                ),
            )
    return replace(
        record,
        definition_state=DefinitionState.PROVISIONAL,
        validated_by=None,
        validated_domain=None,
        confidence=min(0.71, record.confidence),
        evidence_kinds=evidence_kinds,
    )


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
    payload["coding_outcomes"] = [
        {**asdict(item), "evidence": list(item.evidence)}
        for item in record.coding_outcomes
    ]
    payload["scientific_experiments"] = [
        item.to_dict() for item in record.scientific_experiments
    ]
    return payload


def _record_from_json(
    payload: dict[str, object],
    *,
    minimum_reproductions: int = 3,
) -> WildcardMemoryRecord:
    record = WildcardMemoryRecord(
        signature=str(payload["signature"]),
        surface=str(payload.get("surface", "")),
        relation=str(payload.get("relation", "")),
        category=str(payload.get("category", "extension")),
        hypotheses=tuple(payload.get("hypotheses", ())),
        evidence=tuple(payload.get("evidence", ())),
        confidence=float(payload.get("confidence", 0.0)),
        seen_count=int(payload.get("seen_count", 1)),
        language=str(payload.get("language", "python")),
        semantic_spec_sha256=str(payload.get("semantic_spec_sha256", "legacy-python")),
        definition_state=DefinitionState(
            payload.get("definition_state", DefinitionState.PROVISIONAL.value)
        ),
        target_event=(
            str(payload["target_event"])
            if payload.get("target_event") is not None
            else None
        ),
        target_ir_kind=(
            str(payload["target_ir_kind"])
            if payload.get("target_ir_kind") is not None
            else None
        ),
        evidence_kinds=tuple(payload.get("evidence_kinds", ())),
        context_fingerprints=tuple(payload.get("context_fingerprints", ())),
        document_identities=tuple(payload.get("document_identities", ())),
        validated_by=(
            str(payload["validated_by"])
            if payload.get("validated_by") is not None
            else None
        ),
        extension_name=(
            str(payload["extension_name"])
            if payload.get("extension_name") is not None
            else None
        ),
        coding_outcomes=_coding_outcomes_from_json(payload),
        learning_schema_version=int(payload.get("learning_schema_version", 1)),
        hypothesis_revision=int(payload.get("hypothesis_revision", 1)),
        hypothesis_fingerprint=(
            str(payload["hypothesis_fingerprint"])
            if payload.get("hypothesis_fingerprint") is not None
            else None
        ),
        validated_domain=(
            str(payload["validated_domain"])
            if payload.get("validated_domain") is not None
            else None
        ),
        scientific_experiments=_scientific_experiments_from_json(payload),
    )
    # A stored fingerprint is evidence metadata, not authority. Recompute it
    # from the stored projection so edited event/IR fields cannot inherit
    # validation earned by a different hypothesis.
    record = replace(
        record,
        hypothesis_fingerprint=_hypothesis_fingerprint(
            signature=record.signature,
            revision=record.hypothesis_revision,
            target_event=record.target_event,
            target_ir_kind=record.target_ir_kind,
            language=record.language,
            semantic_spec_sha256=record.semantic_spec_sha256,
            extension_name=record.extension_name,
        ),
    )
    return _derive_scientific_state(record, minimum_reproductions)


def _coding_outcomes_from_json(
    payload: dict[str, object],
) -> tuple[CodingOutcomeEvidence, ...]:
    outcomes: list[CodingOutcomeEvidence] = []
    for item in payload.get("coding_outcomes", ()):
        correct = item["correct"]
        if type(correct) is not bool:
            raise ValueError("Persisted coding-outcome correct value must be boolean")
        outcomes.append(
            CodingOutcomeEvidence(
                context_fingerprint=str(item["context_fingerprint"]),
                document_identity=str(item["document_identity"]),
                target_event=str(item["target_event"]),
                target_ir_kind=str(item["target_ir_kind"]),
                verification_oracle=str(item["verification_oracle"]),
                result_fingerprint=str(item["result_fingerprint"]),
                correct=correct,
                evidence=tuple(item.get("evidence", ())),
                extension_name=item.get("extension_name"),
            )
        )
    return tuple(outcomes)


def _scientific_experiments_from_json(
    payload: dict[str, object],
) -> tuple[ScientificExperimentEvidence, ...]:
    experiments: list[ScientificExperimentEvidence] = []
    raw = payload.get("scientific_experiments", ())
    if not isinstance(raw, (list, tuple)):
        raise ValueError("Persisted scientific experiments must be a list")
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Persisted scientific experiment must be an object")
        experiments.append(ScientificExperimentEvidence.from_dict(item))
    return tuple(experiments)
