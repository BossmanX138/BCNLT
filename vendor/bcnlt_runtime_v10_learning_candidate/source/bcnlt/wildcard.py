from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

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
    SANDBOX_EVIDENCE_SCHEMA_VERSION,
    SANDBOX_VERIFIER_ID,
    SANDBOX_VERIFIER_VERSION,
    CodingExperimentFixture,
    HypothesisProjection,
    ScientificExperimentEvidence,
    VerificationStatus,
)

MAX_MEMORY_FILE_BYTES = 64 * 1024 * 1024
MAX_MEMORY_RECORD_BYTES = 128 * 1024
MAX_MEMORY_RECORDS = 100_000
MAX_HYPOTHESES_PER_RECORD = 16
MAX_EVIDENCE_ITEMS_PER_RECORD = 32
MAX_EVIDENCE_KINDS_PER_RECORD = 32
MAX_CONTEXT_FINGERPRINTS_PER_RECORD = 256
MAX_DOCUMENT_IDENTITIES_PER_RECORD = 16
MAX_CODING_OUTCOMES_PER_RECORD = 32
MAX_SCIENTIFIC_EXPERIMENTS_PER_RECORD = 32

_LEDGER_PROCESS_LOCK = threading.RLock()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
    """Append-only memory for unknown semantic constructs BCNLT has already seen."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        language: str = "python",
        semantic_spec_sha256: str = "legacy-python",
        minimum_reproductions: int = 3,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.language = language
        self.semantic_spec_sha256 = semantic_spec_sha256
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
        if self.path is not None and self.path.exists():
            if self.read_only:
                self._load()
            else:
                with self._write_transaction():
                    self._reload_from_disk()

    def signature_for(self, unit: ContextualUnit) -> str:
        surface = unit.surface.strip().lower()
        relation = (unit.ast_relation or "").strip().lower()
        event = unit.event.strip().upper()
        return f"{self.language}|{self.semantic_spec_sha256}|{event}|{relation}|{surface}"

    def lookup(self, unit: ContextualUnit) -> WildcardMemoryRecord | None:
        return self.records.get(self.signature_for(unit))

    def validated_extended_core(self) -> ExtendedCoreLibrary:
        entries = []
        for record in self.records.values():
            entry = _extended_core_entry(record)
            if entry is not None:
                entries.append(entry)
        return ExtendedCoreLibrary(entries)

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

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        with _LEDGER_PROCESS_LOCK:
            if self.path is None:
                yield
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_name(f".{self.path.name}.lock")
            with lock_path.open("a+b") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    try:
                        yield
                    finally:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _reload_from_disk(self) -> None:
        if self.path is None:
            return
        self.records.clear()
        self.load_warnings.clear()
        self._truncated_tail_offset = None
        if self.path.exists():
            self._load()

    def remember(
        self,
        unit: ContextualUnit,
        meaning: ProvisionalMeaning,
        *,
        document_identity: str | None = None,
    ) -> WildcardMemoryRecord:
        with self._write_transaction():
            self._reload_from_disk()
            return self._remember_unlocked(
                unit,
                meaning,
                document_identity=document_identity,
            )

    def _remember_unlocked(
        self,
        unit: ContextualUnit,
        meaning: ProvisionalMeaning,
        *,
        document_identity: str | None = None,
    ) -> WildcardMemoryRecord:
        if self.read_only:
            raise RuntimeError("Frozen wildcard memory is read-only")
        signature = self.signature_for(unit)
        previous = self.records.get(signature)
        seen_count = 1 if previous is None else previous.seen_count + 1
        evidence = _bounded_merge_tuple(
            previous.evidence if previous else (),
            meaning.evidence,
            MAX_EVIDENCE_ITEMS_PER_RECORD,
        )
        hypotheses = _bounded_merge_tuple(
            previous.hypotheses if previous else (),
            meaning.hypotheses,
            MAX_HYPOTHESES_PER_RECORD,
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
        evidence_kinds = _bounded_merge_tuple(
            previous.evidence_kinds if previous else (),
            meaning.evidence_kinds,
            MAX_EVIDENCE_KINDS_PER_RECORD,
        )
        context_fingerprints = _bounded_merge_tuple(
            previous.context_fingerprints if previous else (),
            (self.context_fingerprint_for(unit, document_identity),),
            MAX_CONTEXT_FINGERPRINTS_PER_RECORD,
        )
        document_identities = _bounded_merge_tuple(
            previous.document_identities if previous else (),
            (document_identity or "<unknown>",),
            MAX_DOCUMENT_IDENTITIES_PER_RECORD,
        )
        if len(context_fingerprints) >= 2 and targets_agree:
            evidence_kinds = _bounded_merge_tuple(
                evidence_kinds,
                ("context_agreement",),
                MAX_EVIDENCE_KINDS_PER_RECORD,
            )
        if len(context_fingerprints) >= 3 and targets_agree:
            evidence_kinds = _bounded_merge_tuple(
                evidence_kinds,
                ("stable_pattern",),
                MAX_EVIDENCE_KINDS_PER_RECORD,
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
        verifier: object | None = None,
    ) -> WildcardMemoryRecord:
        with self._write_transaction():
            self._reload_from_disk()
            return self._run_scientific_experiment_unlocked(
                unit,
                source=source,
                fixture=fixture,
                negative_control=negative_control,
                verifier=verifier,
            )

    def _run_scientific_experiment_unlocked(
        self,
        unit: ContextualUnit,
        *,
        source: str,
        fixture: CodingExperimentFixture,
        negative_control: HypothesisProjection,
        verifier: object | None = None,
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
        if verifier is None or not callable(getattr(verifier, "verify", None)):
            raise RuntimeError(
                "Scientific promotion requires an explicit sandbox-backed verifier"
            )
        result = verifier.verify(
            candidate,
            fixture,
            source=source,
            context_fingerprint=context_fingerprint,
            negative_control=negative_control,
        )
        if (
            not isinstance(result, ScientificExperimentEvidence)
            or result.schema_version != SANDBOX_EVIDENCE_SCHEMA_VERSION
            or result.verifier_id != SANDBOX_VERIFIER_ID
            or result.verifier_version != SANDBOX_VERIFIER_VERSION
            or result.sandbox_execution is None
            or ScientificExperimentEvidence.from_dict(result.to_dict()) != result
        ):
            raise RuntimeError(
                "Scientific promotion requires intact sandbox-backed evidence"
            )
        experiments = _merge_scientific_experiments(
            previous.scientific_experiments,
            (result,),
        )
        record = WildcardMemoryRecord(
            **{
                **previous.__dict__,
                "definition_state": DefinitionState.PROVISIONAL,
                "validated_by": None,
                "validated_domain": None,
                "confidence": min(0.71, previous.confidence),
                "evidence_kinds": _bounded_merge_tuple(
                    previous.evidence_kinds,
                    ("scientific_experiment",),
                    MAX_EVIDENCE_KINDS_PER_RECORD,
                ),
                "scientific_experiments": experiments,
            }
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
        with self._write_transaction():
            self._reload_from_disk()
            return self._record_coding_outcome_unlocked(
                unit,
                target_event=target_event,
                target_ir_kind=target_ir_kind,
                evidence=evidence,
                document_identity=document_identity,
                verification_oracle=verification_oracle,
                result_fingerprint=result_fingerprint,
                correct=correct,
                extension_name=extension_name,
            )

    def _record_coding_outcome_unlocked(
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
            correct=bool(correct),
            evidence=_bounded_merge_tuple((), evidence, MAX_EVIDENCE_ITEMS_PER_RECORD),
            extension_name=normalized_extension,
        )
        outcomes = _merge_outcomes(previous.coding_outcomes, (outcome,))
        latest_by_context = {
            item.context_fingerprint: item for item in outcomes
        }
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
        required_contexts = (
            3 if previous.target_ir_kind == IRKind.EXTENSION.value else 2
        )
        # The inherited two-context / three-extension counts are retained only
        # as reproducibility bookkeeping. Caller-supplied pass metadata is not
        # an independent correctness oracle, so it can never promote a guess.
        reproduced = (
            len(agreed) >= required_contexts
            and len({item.document_identity for item in agreed}) >= 2
            and len(agreed) == len(latest_by_context)
        )
        evidence_kinds = _bounded_merge_tuple(
            previous.evidence_kinds,
            ("coding_outcome_observation",),
            MAX_EVIDENCE_KINDS_PER_RECORD,
        )
        if reproduced:
            evidence_kinds = _bounded_merge_tuple(
                evidence_kinds,
                ("reproduced_claim_pending_verifier",),
                MAX_EVIDENCE_KINDS_PER_RECORD,
            )
        record = WildcardMemoryRecord(
            signature=previous.signature,
            surface=previous.surface,
            relation=previous.relation,
            category=previous.category,
            hypotheses=previous.hypotheses,
            evidence=_bounded_merge_tuple(
                previous.evidence,
                evidence,
                MAX_EVIDENCE_ITEMS_PER_RECORD,
            ),
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
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._truncated_tail_offset is not None:
                with self.path.open("r+b") as handle:
                    handle.truncate(self._truncated_tail_offset)
                self._truncated_tail_offset = None
            encoded = (
                json.dumps(
                    _record_to_json(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            if len(encoded) > MAX_MEMORY_RECORD_BYTES:
                raise ValueError("Learned-memory record exceeds the bounded line limit")
            existing_size = self.path.stat().st_size if self.path.exists() else 0
            if existing_size + len(encoded) > MAX_MEMORY_FILE_BYTES:
                raise ValueError("Learned-memory ledger exceeds the bounded file limit")
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        self.records[record.signature] = record

    def _load(self) -> None:
        if self.path is None:
            return
        if self.path.stat().st_size > MAX_MEMORY_FILE_BYTES:
            raise ValueError("Learned-memory ledger exceeds the bounded file limit")
        try:
            raw_lines = self.path.read_bytes().decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError as error:
            raise ValueError("Learned-memory ledger is not valid UTF-8") from error
        if len(raw_lines) > MAX_MEMORY_RECORDS:
            raise ValueError("Learned-memory ledger exceeds the record-count limit")
        lines: list[tuple[int, str, int]] = []
        byte_offset = 0
        for line_number, raw_line in enumerate(raw_lines, start=1):
            line_bytes = len(raw_line.encode("utf-8"))
            if line_bytes > MAX_MEMORY_RECORD_BYTES:
                raise ValueError(
                    f"Learned-memory record at line {line_number} exceeds the line limit"
                )
            line = raw_line.rstrip("\r\n")
            if line.strip():
                lines.append((line_number, line, byte_offset))
            byte_offset += line_bytes
        for position, (line_number, line, line_offset) in enumerate(lines):
            try:
                payload = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except (json.JSONDecodeError, ValueError) as exc:
                final_line_is_truncated = (
                    position == len(lines) - 1
                    and bool(raw_lines)
                    and not raw_lines[-1].endswith(("\n", "\r"))
                )
                if final_line_is_truncated:
                    self._truncated_tail_offset = line_offset
                    self.load_warnings.append(
                        f"ignored truncated final learned-memory record at line {line_number}"
                    )
                    continue
                raise ValueError(
                    f"Malformed learned-memory record at line {line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Learned-memory record at line {line_number} must be an object"
                )
            record = WildcardMemoryRecord(
                signature=_required_string(payload, "signature"),
                surface=_string_value(payload, "surface", ""),
                relation=_string_value(payload, "relation", ""),
                category=_string_value(payload, "category", "extension"),
                hypotheses=_string_tuple(payload, "hypotheses"),
                evidence=_string_tuple(payload, "evidence"),
                confidence=_bounded_float(payload, "confidence", 0.0),
                seen_count=_positive_int(payload, "seen_count", 1),
                language=_string_value(payload, "language", self.language),
                semantic_spec_sha256=_string_value(
                    payload, "semantic_spec_sha256", self.semantic_spec_sha256
                ),
                definition_state=DefinitionState(
                    payload.get("definition_state", DefinitionState.PROVISIONAL.value)
                ),
                target_event=_optional_string(payload, "target_event"),
                target_ir_kind=_optional_string(payload, "target_ir_kind"),
                evidence_kinds=_string_tuple(payload, "evidence_kinds"),
                context_fingerprints=_string_tuple(payload, "context_fingerprints"),
                document_identities=_string_tuple(payload, "document_identities"),
                validated_by=_optional_string(payload, "validated_by"),
                extension_name=_optional_string(payload, "extension_name"),
                coding_outcomes=_coding_outcomes_from_json(payload),
                learning_schema_version=_positive_int(
                    payload, "learning_schema_version", 1
                ),
                hypothesis_revision=_positive_int(
                    payload, "hypothesis_revision", 1
                ),
                hypothesis_fingerprint=_optional_string(
                    payload, "hypothesis_fingerprint"
                ),
                validated_domain=_optional_string(payload, "validated_domain"),
                scientific_experiments=_scientific_experiments_from_json(payload),
            )
            if record.language != self.language:
                raise ValueError(
                    f"Learned-memory language mismatch at line {line_number}"
                )
            if record.semantic_spec_sha256 != self.semantic_spec_sha256:
                raise ValueError(
                    f"Learned-memory semantic-spec mismatch at line {line_number}"
                )
            expected_signature_prefix = (
                f"{self.language}|{self.semantic_spec_sha256}|"
            )
            if not record.signature.startswith(expected_signature_prefix):
                raise ValueError(
                    f"Learned-memory signature scope mismatch at line {line_number}"
                )
            # A stored fingerprint is evidence metadata, not authority. Recompute it
            # from the stored projection so edited event/IR fields cannot inherit
            # validation earned by a different hypothesis.
            record = WildcardMemoryRecord(
                **{
                    **record.__dict__,
                    "hypothesis_fingerprint": _hypothesis_fingerprint(
                        signature=record.signature,
                        revision=record.hypothesis_revision,
                        target_event=record.target_event,
                        target_ir_kind=record.target_ir_kind,
                        language=record.language,
                        semantic_spec_sha256=record.semantic_spec_sha256,
                        extension_name=record.extension_name,
                    ),
                }
            )
            record = _derive_scientific_state(record, self.minimum_reproductions)
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
                hypotheses=_bounded_merge_tuple(
                    previous.hypotheses,
                    record.hypotheses,
                    MAX_HYPOTHESES_PER_RECORD,
                ),
                evidence=_bounded_merge_tuple(
                    previous.evidence,
                    record.evidence,
                    MAX_EVIDENCE_ITEMS_PER_RECORD,
                ),
                confidence=max(previous.confidence, record.confidence),
                seen_count=max(previous.seen_count, record.seen_count),
                language=newer.language,
                semantic_spec_sha256=newer.semantic_spec_sha256,
                definition_state=newer.definition_state,
                target_event=newer.target_event or previous.target_event,
                target_ir_kind=newer.target_ir_kind or previous.target_ir_kind,
                evidence_kinds=_bounded_merge_tuple(
                    previous.evidence_kinds,
                    record.evidence_kinds,
                    MAX_EVIDENCE_KINDS_PER_RECORD,
                ),
                context_fingerprints=_bounded_merge_tuple(
                    previous.context_fingerprints,
                    record.context_fingerprints,
                    MAX_CONTEXT_FINGERPRINTS_PER_RECORD,
                ),
                document_identities=_bounded_merge_tuple(
                    previous.document_identities,
                    record.document_identities,
                    MAX_DOCUMENT_IDENTITIES_PER_RECORD,
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
        if unit.event == SemanticEvent.REFERENCE.value:
            return ProvisionalMeaning(
                category="symbol_reference",
                hypotheses=("symbol may be defined outside the current source unit",),
                evidence=tuple(evidence),
                confidence=0.45,
                target_event=SemanticEvent.REFERENCE.value,
                target_ir_kind=IRKind.VARIABLE.value,
            )
        if unit.event == SemanticEvent.CALL.value:
            return ProvisionalMeaning(
                category="call",
                hypotheses=("call target may be defined outside the current source unit",),
                evidence=tuple(evidence),
                confidence=0.45,
                target_event=SemanticEvent.CALL.value,
                target_ir_kind=IRKind.CALL.value,
            )
        return ProvisionalMeaning(
            category="unresolved",
            hypotheses=("construct meaning is unresolved in the current source unit",),
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


def _bounded_merge_tuple(
    left: tuple[str, ...],
    right: tuple[str, ...],
    limit: int,
) -> tuple[str, ...]:
    """Preserve first-seen evidence without unbounded quadratic tuple growth."""

    if type(limit) is not int or limit < 1:
        raise ValueError("bounded merge limit must be a positive integer")
    merged = list(left[:limit])
    seen = set(merged)
    for item in right:
        if item in seen:
            continue
        if len(merged) >= limit:
            break
        merged.append(item)
        seen.add(item)
    return tuple(merged)


def _merge_outcomes(
    left: tuple[CodingOutcomeEvidence, ...],
    right: tuple[CodingOutcomeEvidence, ...],
) -> tuple[CodingOutcomeEvidence, ...]:
    merged: list[CodingOutcomeEvidence] = []
    for item in (*left, *right):
        if item not in merged:
            merged.append(item)
        if len(merged) >= MAX_CODING_OUTCOMES_PER_RECORD:
            break
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
        if len(merged) >= MAX_SCIENTIFIC_EXPERIMENTS_PER_RECORD:
            break
    return tuple(merged)


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
    current = tuple(
        item
        for item in record.scientific_experiments
        if item.hypothesis_fingerprint == record.hypothesis_fingerprint
        and item.hypothesis_revision == record.hypothesis_revision
        and item.language == record.language
        and item.semantic_spec_sha256 == record.semantic_spec_sha256
        and item.schema_version == SANDBOX_EVIDENCE_SCHEMA_VERSION
        and item.verifier_id == SANDBOX_VERIFIER_ID
        and item.verifier_version == SANDBOX_VERIFIER_VERSION
        and item.sandbox_execution is not None
    )
    evidence_kinds = tuple(
        item
        for item in record.evidence_kinds
        if item not in {"scientifically_reproduced", "scientific_contradiction"}
    )
    if not current:
        return WildcardMemoryRecord(
            **{
                **record.__dict__,
                "definition_state": DefinitionState.PROVISIONAL,
                "validated_by": None,
                "validated_domain": None,
                "confidence": min(0.71, record.confidence),
                "evidence_kinds": evidence_kinds,
            }
        )
    if any(item.status is not VerificationStatus.PASSED for item in current):
        return WildcardMemoryRecord(
            **{
                **record.__dict__,
                "definition_state": DefinitionState.PROVISIONAL,
                "validated_by": None,
                "validated_domain": None,
                "confidence": min(0.71, record.confidence),
                "evidence_kinds": _bounded_merge_tuple(
                    evidence_kinds,
                    ("scientific_contradiction",),
                    MAX_EVIDENCE_KINDS_PER_RECORD,
                ),
            }
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
            return WildcardMemoryRecord(
                **{
                    **record.__dict__,
                    "definition_state": DefinitionState.VALIDATED,
                    "validated_by": (
                        f"{SANDBOX_VERIFIER_ID}:{SANDBOX_VERIFIER_VERSION}"
                    ),
                    "validated_domain": domain,
                    "confidence": max(0.9, record.confidence),
                    "evidence_kinds": _bounded_merge_tuple(
                        evidence_kinds,
                        ("scientifically_reproduced",),
                        MAX_EVIDENCE_KINDS_PER_RECORD,
                    ),
                }
            )
    return WildcardMemoryRecord(
        **{
            **record.__dict__,
            "definition_state": DefinitionState.PROVISIONAL,
            "validated_by": None,
            "validated_domain": None,
            "confidence": min(0.71, record.confidence),
            "evidence_kinds": evidence_kinds,
        }
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
        {
            **asdict(item),
            "evidence": list(item.evidence),
        }
        for item in record.coding_outcomes
    ]
    payload["scientific_experiments"] = [
        item.to_dict() for item in record.scientific_experiments
    ]
    return payload


def _coding_outcomes_from_json(
    payload: dict[str, object],
) -> tuple[CodingOutcomeEvidence, ...]:
    outcomes: list[CodingOutcomeEvidence] = []
    raw = payload.get("coding_outcomes", ())
    if not isinstance(raw, (list, tuple)):
        raise ValueError("Persisted coding outcomes must be a list")
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Persisted coding outcome must be an object")
        correct = item["correct"]
        if type(correct) is not bool:
            raise ValueError("Persisted coding-outcome correct value must be boolean")
        outcomes.append(
            CodingOutcomeEvidence(
                context_fingerprint=_required_string(item, "context_fingerprint"),
                document_identity=_required_string(item, "document_identity"),
                target_event=_required_string(item, "target_event"),
                target_ir_kind=_required_string(item, "target_ir_kind"),
                verification_oracle=_required_string(item, "verification_oracle"),
                result_fingerprint=_required_string(item, "result_fingerprint"),
                correct=correct,
                evidence=_string_tuple(item, "evidence"),
                extension_name=_optional_string(item, "extension_name"),
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


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Persisted learned-memory {key} must be a non-empty string")
    return value


def _string_value(payload: dict[str, object], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"Persisted learned-memory {key} must be a string")
    return value


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Persisted learned-memory {key} must be a string or null")
    return value


def _string_tuple(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key, ())
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"Persisted learned-memory {key} must be a string list")
    return tuple(value)


def _positive_int(payload: dict[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Persisted learned-memory {key} must be a positive integer")
    return value


def _bounded_float(payload: dict[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"Persisted learned-memory {key} must be finite in [0, 1]")
    return float(value)
