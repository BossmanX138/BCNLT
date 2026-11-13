from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from .extended_core import (
    CommonCoreProjection,
    ExtendedCoreEntry,
    ExtendedCoreLibrary,
    extended_core_id,
)
from .extended_core_vocab import (
    ExtendedCoreVocabularyManifest,
    build_extended_core_vocab,
    load_extended_core_vocab,
    write_extended_core_vocab,
)
from .ir import COMMON_CODING_CONCEPTS
from .scientific_learning import (
    SANDBOX_EVIDENCE_SCHEMA_VERSION,
    SANDBOX_VERIFIER_ID,
    SANDBOX_VERIFIER_VERSION,
    ScientificExperimentEvidence,
    VerificationStatus,
)
from .tokenizer_manifest import TokenizerManifest
from .wildcard import DefinitionState, WildcardMemory, WildcardMemoryRecord

SCHEMA_VERSION = "bcnlt-extended-core-version-v1"
LIBRARY_SCHEMA_VERSION = "bcnlt-extended-core-library-v1"
VALIDATION_SCHEMA_VERSION = "bcnlt-extended-core-validation-v1"
ACTIVATION_SCHEMA_VERSION = "bcnlt-extended-core-activation-v1"
VOCAB_FILENAME = "bcnlt_extended_core_manifest.json"
LIBRARY_FILENAME = "extended_core_library.json"
VERSION_FILENAME = "version.json"

_ACTIVATION_PROCESS_LOCK = threading.RLock()


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _bytes_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _common_core_sha256() -> str:
    concepts = sorted(
        (
            {"event": item.event, "ir_kind": item.ir_kind.value}
            for item in COMMON_CODING_CONCEPTS
        ),
        key=lambda item: (item["event"], item["ir_kind"]),
    )
    return _digest({"schema": "bcnlt-common-coding-core-v1", "concepts": concepts})


def _entry_dict(entry: ExtendedCoreEntry) -> dict[str, object]:
    return {
        "extended_core_id": entry.extended_core_id,
        "signature": entry.signature,
        "revision": entry.revision,
        "language": entry.language,
        "semantic_spec_sha256": entry.semantic_spec_sha256,
        "surface": entry.surface,
        "relation": entry.relation,
        "category": entry.category,
        "hypotheses": list(entry.hypotheses),
        "evidence": list(entry.evidence),
        "confidence": entry.confidence,
        "validated_domain": entry.validated_domain,
        "common_core_projection": (
            entry.common_core_projection.to_dict()
            if entry.common_core_projection is not None
            else None
        ),
        "evidence_sha256": list(entry.evidence_sha256),
        "extension_name": entry.extension_name,
    }


def _entry_from_dict(payload: dict[str, Any]) -> ExtendedCoreEntry:
    projection_payload = payload["common_core_projection"]
    projection = (
        None
        if projection_payload is None
        else CommonCoreProjection(
            event=str(projection_payload["event"]),
            ir_kind=str(projection_payload["ir_kind"]),
        )
    )
    entry = ExtendedCoreEntry(
        extended_core_id=str(payload["extended_core_id"]),
        signature=str(payload["signature"]),
        revision=int(payload["revision"]),
        language=str(payload["language"]),
        semantic_spec_sha256=str(payload["semantic_spec_sha256"]),
        surface=str(payload["surface"]),
        relation=str(payload["relation"]),
        category=str(payload["category"]),
        hypotheses=tuple(str(item) for item in payload["hypotheses"]),
        evidence=tuple(str(item) for item in payload["evidence"]),
        confidence=float(payload["confidence"]),
        validated_domain=str(payload["validated_domain"]),
        common_core_projection=projection,
        evidence_sha256=tuple(str(item) for item in payload["evidence_sha256"]),
        extension_name=(
            None if payload["extension_name"] is None else str(payload["extension_name"])
        ),
    )
    if entry.extended_core_id != extended_core_id(
        language=entry.language,
        signature=entry.signature,
        revision=entry.revision,
    ):
        raise ValueError("Extended Core entry identity does not match its content")
    if not _is_digest(entry.semantic_spec_sha256):
        raise ValueError("Extended Core semantic specification hash is invalid")
    if not entry.evidence_sha256 or any(
        not _is_digest(item) for item in entry.evidence_sha256
    ):
        raise ValueError("Extended Core evidence hashes are invalid")
    return entry


def _library_payload(library: ExtendedCoreLibrary) -> dict[str, object]:
    return {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "common_core_sha256": _common_core_sha256(),
        "entries": [_entry_dict(entry) for entry in library],
    }


def _eligible_experiments(
    record: WildcardMemoryRecord,
    minimum_reproductions: int,
) -> tuple[ScientificExperimentEvidence, ...]:
    if (
        record.definition_state is not DefinitionState.VALIDATED
        or record.validated_by != f"{SANDBOX_VERIFIER_ID}:{SANDBOX_VERIFIER_VERSION}"
        or record.hypothesis_fingerprint is None
        or record.validated_domain is None
    ):
        return ()
    current: list[ScientificExperimentEvidence] = []
    for item in record.scientific_experiments:
        if (
            item.hypothesis_fingerprint != record.hypothesis_fingerprint
            or item.hypothesis_revision != record.hypothesis_revision
            or item.language != record.language
            or item.semantic_spec_sha256 != record.semantic_spec_sha256
            or item.schema_version != SANDBOX_EVIDENCE_SCHEMA_VERSION
            or item.verifier_id != SANDBOX_VERIFIER_ID
            or item.verifier_version != SANDBOX_VERIFIER_VERSION
            or item.status is not VerificationStatus.PASSED
            or item.sandbox_execution is None
        ):
            continue
        try:
            restored = ScientificExperimentEvidence.from_dict(item.to_dict())
        except (KeyError, TypeError, ValueError):
            continue
        execution = restored.sandbox_execution
        if (
            restored != item
            or execution is None
            or execution.status != "passed"
            or execution.phase != "run"
            or execution.exit_code != 0
            or execution.timed_out
            or execution.output_limited
            or execution.language != record.language
            or not restored.intended_passed
            or not restored.negative_control_rejected
            or restored.domain != record.validated_domain
        ):
            continue
        current.append(restored)
    unique = {item.independence_key: item for item in current}.values()
    eligible = tuple(unique)
    if (
        len(eligible) < minimum_reproductions
        or len({item.context_fingerprint for item in eligible}) < minimum_reproductions
        or len({item.document_identity for item in eligible}) < minimum_reproductions
        or len({item.stimulus_sha256 for item in eligible}) < minimum_reproductions
        or len({item.source_sha256 for item in eligible}) < 2
    ):
        return ()
    return eligible


@dataclass(frozen=True, slots=True)
class ExtendedCorePromotionCandidate:
    state: str
    library: ExtendedCoreLibrary
    minimum_reproductions: int
    evidence_sha256: tuple[str, ...]
    common_core_sha256: str
    candidate_sha256: str

    @classmethod
    def from_memory(cls, memory: WildcardMemory) -> "ExtendedCorePromotionCandidate":
        if memory.minimum_reproductions < 2:
            raise ValueError("Extended Core promotion requires repeated evidence")
        library = memory.validated_extended_core()
        by_signature = {entry.signature: entry for entry in library}
        entries: list[ExtendedCoreEntry] = []
        evidence_hashes: list[str] = []
        for record in sorted(memory.records.values(), key=lambda item: item.signature):
            evidence = _eligible_experiments(record, memory.minimum_reproductions)
            entry = by_signature.get(record.signature)
            if entry is None or not evidence:
                continue
            expected_hashes = tuple(item.evidence_sha256 for item in evidence)
            if entry.evidence_sha256 != expected_hashes:
                raise ValueError("Extended Core entry evidence does not match its attestations")
            entries.append(entry)
            evidence_hashes.extend(expected_hashes)
        eligible_library = ExtendedCoreLibrary(entries)
        if not entries:
            raise ValueError(
                "Extended Core candidate requires sandbox-attested independent reproductions"
            )
        common_core_sha256 = _common_core_sha256()
        body = {
            "state": "candidate",
            "minimum_reproductions": memory.minimum_reproductions,
            "common_core_sha256": common_core_sha256,
            "library": _library_payload(eligible_library),
            "evidence_sha256": sorted(evidence_hashes),
        }
        return cls(
            state="candidate",
            library=eligible_library,
            minimum_reproductions=memory.minimum_reproductions,
            evidence_sha256=tuple(sorted(evidence_hashes)),
            common_core_sha256=common_core_sha256,
            candidate_sha256=_digest(body),
        )

    def validate(self) -> None:
        if self.state != "candidate" or self.minimum_reproductions < 2:
            raise ValueError("Extended Core candidate state is invalid")
        if self.common_core_sha256 != _common_core_sha256():
            raise ValueError("Extended Core candidate Common Core identity changed")
        if not self.evidence_sha256 or any(
            not _is_digest(item) for item in self.evidence_sha256
        ):
            raise ValueError("Extended Core candidate evidence hashes are invalid")
        body = {
            "state": self.state,
            "minimum_reproductions": self.minimum_reproductions,
            "common_core_sha256": self.common_core_sha256,
            "library": _library_payload(self.library),
            "evidence_sha256": list(self.evidence_sha256),
        }
        if self.candidate_sha256 != _digest(body):
            raise ValueError("Extended Core candidate integrity mismatch")


@dataclass(frozen=True, slots=True)
class ExtendedCoreVersionManifest:
    schema_version: str
    state: str
    version_number: int
    parent_version_sha256: str | None
    v1_manifest_sha256: str
    common_core_sha256: str
    candidate_sha256: str
    library_sha256: str
    vocabulary_manifest_sha256: str
    evidence_sha256: tuple[str, ...]
    version_sha256: str

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "version_number": self.version_number,
            "parent_version_sha256": self.parent_version_sha256,
            "v1_manifest_sha256": self.v1_manifest_sha256,
            "common_core_sha256": self.common_core_sha256,
            "candidate_sha256": self.candidate_sha256,
            "library_sha256": self.library_sha256,
            "vocabulary_manifest_sha256": self.vocabulary_manifest_sha256,
            "evidence_sha256": list(self.evidence_sha256),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "version_sha256": self.version_sha256}


@dataclass(frozen=True, slots=True)
class ValidatedExtendedCoreVersion:
    schema_version: str
    state: str
    version_sha256: str
    validation_report_sha256: str
    version_files_sha256: str
    receipt_sha256: str

    def _body(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "version_sha256": self.version_sha256,
            "validation_report_sha256": self.validation_report_sha256,
            "version_files_sha256": self.version_files_sha256,
        }

    def to_dict(self) -> dict[str, str]:
        return {**self._body(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class ActiveExtendedCorePointer:
    schema_version: str
    state: str
    action: str
    sequence: int
    version_sha256: str
    previous_version_sha256: str | None
    pointer_sha256: str

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "action": self.action,
            "sequence": self.sequence,
            "version_sha256": self.version_sha256,
            "previous_version_sha256": self.previous_version_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "pointer_sha256": self.pointer_sha256}


class ExtendedCoreVersionStore:
    """Immutable Extended Core versions with validated atomic activation and rollback."""

    def __init__(self, root: str | Path, v1_manifest: TokenizerManifest) -> None:
        self.root = Path(root)
        self.v1_manifest = v1_manifest
        self.versions = self.root / "versions"
        self.validations = self.root / "validations"
        self.events = self.root / "activation-events"
        self.staging = self.root / ".staging"
        for path in (self.versions, self.validations, self.events, self.staging):
            path.mkdir(parents=True, exist_ok=True)

    def version_path(self, version_sha256: str) -> Path:
        if not _is_digest(version_sha256):
            raise ValueError("Extended Core version hash is invalid")
        return self.versions / version_sha256

    def stage(
        self,
        candidate: ExtendedCorePromotionCandidate,
    ) -> ExtendedCoreVersionManifest:
        candidate.validate()
        current = self.active(required=False)
        parent = None if current is None else self.load_version(current.version_sha256)
        previous_vocab = (
            None
            if parent is None
            else self.load_vocab_manifest(parent.version_sha256)
        )
        vocabulary = build_extended_core_vocab(
            self.v1_manifest,
            candidate.library,
            previous=previous_vocab,
        )
        library_payload = _library_payload(candidate.library)
        library_sha256 = _digest(library_payload)
        provisional = ExtendedCoreVersionManifest(
            schema_version=SCHEMA_VERSION,
            state="staged",
            version_number=1 if parent is None else parent.version_number + 1,
            parent_version_sha256=(None if parent is None else parent.version_sha256),
            v1_manifest_sha256=self.v1_manifest.manifest_sha256,
            common_core_sha256=candidate.common_core_sha256,
            candidate_sha256=candidate.candidate_sha256,
            library_sha256=library_sha256,
            vocabulary_manifest_sha256=vocabulary.manifest_sha256,
            evidence_sha256=candidate.evidence_sha256,
            version_sha256="",
        )
        version = replace(
            provisional,
            version_sha256=_digest(provisional._body()),
        )
        destination = self.version_path(version.version_sha256)
        if destination.exists():
            loaded = self.load_version(version.version_sha256)
            if loaded != version:
                raise ValueError("Existing Extended Core version content conflicts")
            return loaded
        temporary = Path(tempfile.mkdtemp(prefix="version-", dir=self.staging))
        try:
            (temporary / LIBRARY_FILENAME).write_bytes(_canonical(library_payload))
            write_extended_core_vocab(
                temporary / VOCAB_FILENAME,
                vocabulary,
                self.v1_manifest,
            )
            (temporary / VERSION_FILENAME).write_bytes(_canonical(version.to_dict()))
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return self.load_version(version.version_sha256)

    def validate(
        self,
        version_sha256: str,
        *,
        validation_report_sha256: str,
    ) -> ValidatedExtendedCoreVersion:
        if not _is_digest(validation_report_sha256):
            raise ValueError("Extended Core validation report hash is invalid")
        self.load_version(version_sha256)
        files_sha256 = _digest(self.version_file_hashes(version_sha256))
        provisional = ValidatedExtendedCoreVersion(
            schema_version=VALIDATION_SCHEMA_VERSION,
            state="validated",
            version_sha256=version_sha256,
            validation_report_sha256=validation_report_sha256,
            version_files_sha256=files_sha256,
            receipt_sha256="",
        )
        receipt = replace(
            provisional,
            receipt_sha256=_digest(provisional._body()),
        )
        path = self.validations / f"{version_sha256}.json"
        self._write_immutable(path, _canonical(receipt.to_dict()))
        return self._load_validation(version_sha256)

    def activate(self, version_sha256: str) -> ActiveExtendedCorePointer:
        with self._activation_transaction():
            return self._activate_unlocked(version_sha256)

    def _activate_unlocked(self, version_sha256: str) -> ActiveExtendedCorePointer:
        version = self.load_version(version_sha256)
        self._load_validation(version_sha256)
        current = self.active(required=False)
        current_sha = None if current is None else current.version_sha256
        if version.parent_version_sha256 != current_sha:
            raise ValueError("Extended Core activation parent is not the active version")
        return self._set_active(
            version_sha256,
            previous=current_sha,
            action="activate",
        )

    def rollback(self) -> ActiveExtendedCorePointer:
        with self._activation_transaction():
            return self._rollback_unlocked()

    def _rollback_unlocked(self) -> ActiveExtendedCorePointer:
        current = self.active()
        version = self.load_version(current.version_sha256)
        parent_sha = version.parent_version_sha256
        if parent_sha is None:
            raise ValueError("The first Extended Core version has no rollback parent")
        self.load_version(parent_sha)
        self._load_validation(parent_sha)
        return self._set_active(
            parent_sha,
            previous=current.version_sha256,
            action="rollback",
        )

    @contextmanager
    def _activation_transaction(self) -> Iterator[None]:
        with _ACTIVATION_PROCESS_LOCK:
            lock_path = self.root / ".activation.lock"
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

    def active(self, *, required: bool = True) -> ActiveExtendedCorePointer | None:
        path = self.root / "active.json"
        if not path.exists():
            if required:
                raise ValueError("No Extended Core version is active")
            return None
        payload = self._read_canonical_object(path)
        pointer = ActiveExtendedCorePointer(
            schema_version=str(payload["schema_version"]),
            state=str(payload["state"]),
            action=str(payload["action"]),
            sequence=int(payload["sequence"]),
            version_sha256=str(payload["version_sha256"]),
            previous_version_sha256=(
                None
                if payload["previous_version_sha256"] is None
                else str(payload["previous_version_sha256"])
            ),
            pointer_sha256=str(payload["pointer_sha256"]),
        )
        self._validate_pointer(pointer)
        self.load_version(pointer.version_sha256)
        self._load_validation(pointer.version_sha256)
        return pointer

    def load_version(self, version_sha256: str) -> ExtendedCoreVersionManifest:
        directory = self.version_path(version_sha256)
        payload = self._read_canonical_object(directory / VERSION_FILENAME)
        manifest = ExtendedCoreVersionManifest(
            schema_version=str(payload["schema_version"]),
            state=str(payload["state"]),
            version_number=int(payload["version_number"]),
            parent_version_sha256=(
                None
                if payload["parent_version_sha256"] is None
                else str(payload["parent_version_sha256"])
            ),
            v1_manifest_sha256=str(payload["v1_manifest_sha256"]),
            common_core_sha256=str(payload["common_core_sha256"]),
            candidate_sha256=str(payload["candidate_sha256"]),
            library_sha256=str(payload["library_sha256"]),
            vocabulary_manifest_sha256=str(payload["vocabulary_manifest_sha256"]),
            evidence_sha256=tuple(str(item) for item in payload["evidence_sha256"]),
            version_sha256=str(payload["version_sha256"]),
        )
        if (
            manifest.schema_version != SCHEMA_VERSION
            or manifest.state != "staged"
            or manifest.version_number < 1
            or manifest.v1_manifest_sha256 != self.v1_manifest.manifest_sha256
            or manifest.common_core_sha256 != _common_core_sha256()
            or manifest.version_sha256 != version_sha256
            or manifest.version_sha256 != _digest(manifest._body())
            or any(not _is_digest(item) for item in manifest.evidence_sha256)
        ):
            raise ValueError("Extended Core version manifest integrity mismatch")
        library_bytes = (directory / LIBRARY_FILENAME).read_bytes()
        if _bytes_digest(library_bytes) != manifest.library_sha256:
            raise ValueError("Extended Core library content hash mismatch")
        self.load_library(version_sha256)
        vocabulary = self.load_vocab_manifest(version_sha256)
        if vocabulary.manifest_sha256 != manifest.vocabulary_manifest_sha256:
            raise ValueError("Extended Core vocabulary manifest hash mismatch")
        if manifest.parent_version_sha256 is not None:
            parent = self.load_version(manifest.parent_version_sha256)
            if manifest.version_number != parent.version_number + 1:
                raise ValueError("Extended Core version number is not contiguous")
        elif manifest.version_number != 1:
            raise ValueError("Root Extended Core version number must be one")
        return manifest

    def load_library(self, version_sha256: str) -> ExtendedCoreLibrary:
        path = self.version_path(version_sha256) / LIBRARY_FILENAME
        payload = self._read_canonical_object(path)
        if (
            payload.get("schema_version") != LIBRARY_SCHEMA_VERSION
            or payload.get("common_core_sha256") != _common_core_sha256()
            or not isinstance(payload.get("entries"), list)
        ):
            raise ValueError("Extended Core library identity mismatch")
        return ExtendedCoreLibrary(
            _entry_from_dict(item) for item in payload["entries"]
        )

    def load_vocab_manifest(
        self,
        version_sha256: str,
    ) -> ExtendedCoreVocabularyManifest:
        return load_extended_core_vocab(
            self.version_path(version_sha256) / VOCAB_FILENAME,
            self.v1_manifest,
        )

    def version_file_hashes(self, version_sha256: str) -> dict[str, str]:
        directory = self.version_path(version_sha256)
        expected = (LIBRARY_FILENAME, VERSION_FILENAME, VOCAB_FILENAME)
        observed = sorted(path.name for path in directory.iterdir() if path.is_file())
        if observed != sorted(expected):
            raise ValueError("Extended Core version file inventory mismatch")
        return {
            name: _bytes_digest((directory / name).read_bytes()) for name in expected
        }

    def verify_all(self) -> None:
        for path in sorted(self.versions.iterdir()):
            if path.is_dir():
                self.load_version(path.name)
        for path in sorted(self.validations.glob("*.json")):
            self._load_validation(path.stem)
        previous_sequence = 0
        for path in sorted(self.events.glob("*.json")):
            payload = self._read_canonical_object(path)
            pointer = self._pointer_from_payload(payload)
            self._validate_pointer(pointer)
            if pointer.sequence != previous_sequence + 1:
                raise ValueError("Extended Core activation event sequence has a gap")
            if path.name != f"{pointer.sequence:020d}.json":
                raise ValueError("Extended Core activation event filename mismatch")
            self.load_version(pointer.version_sha256)
            self._load_validation(pointer.version_sha256)
            previous_sequence = pointer.sequence
        active = self.active(required=False)
        if active is not None and active.sequence != previous_sequence:
            raise ValueError("Extended Core active pointer is not the latest event")

    def _load_validation(
        self,
        version_sha256: str,
    ) -> ValidatedExtendedCoreVersion:
        path = self.validations / f"{version_sha256}.json"
        payload = self._read_canonical_object(path)
        receipt = ValidatedExtendedCoreVersion(
            schema_version=str(payload["schema_version"]),
            state=str(payload["state"]),
            version_sha256=str(payload["version_sha256"]),
            validation_report_sha256=str(payload["validation_report_sha256"]),
            version_files_sha256=str(payload["version_files_sha256"]),
            receipt_sha256=str(payload["receipt_sha256"]),
        )
        if (
            receipt.schema_version != VALIDATION_SCHEMA_VERSION
            or receipt.state != "validated"
            or receipt.version_sha256 != version_sha256
            or not _is_digest(receipt.validation_report_sha256)
            or receipt.receipt_sha256 != _digest(receipt._body())
            or receipt.version_files_sha256
            != _digest(self.version_file_hashes(version_sha256))
        ):
            raise ValueError("Extended Core validation receipt integrity mismatch")
        return receipt

    def _set_active(
        self,
        version_sha256: str,
        *,
        previous: str | None,
        action: str,
    ) -> ActiveExtendedCorePointer:
        sequence = len(tuple(self.events.glob("*.json"))) + 1
        provisional = ActiveExtendedCorePointer(
            schema_version=ACTIVATION_SCHEMA_VERSION,
            state="active",
            action=action,
            sequence=sequence,
            version_sha256=version_sha256,
            previous_version_sha256=previous,
            pointer_sha256="",
        )
        pointer = replace(
            provisional,
            pointer_sha256=_digest(provisional._body()),
        )
        event_path = self.events / f"{sequence:020d}.json"
        self._write_immutable(event_path, _canonical(pointer.to_dict()))
        self._write_atomic(self.root / "active.json", _canonical(pointer.to_dict()))
        return self.active()

    @staticmethod
    def _pointer_from_payload(payload: dict[str, Any]) -> ActiveExtendedCorePointer:
        return ActiveExtendedCorePointer(
            schema_version=str(payload["schema_version"]),
            state=str(payload["state"]),
            action=str(payload["action"]),
            sequence=int(payload["sequence"]),
            version_sha256=str(payload["version_sha256"]),
            previous_version_sha256=(
                None
                if payload["previous_version_sha256"] is None
                else str(payload["previous_version_sha256"])
            ),
            pointer_sha256=str(payload["pointer_sha256"]),
        )

    @staticmethod
    def _validate_pointer(pointer: ActiveExtendedCorePointer) -> None:
        if (
            pointer.schema_version != ACTIVATION_SCHEMA_VERSION
            or pointer.state != "active"
            or pointer.action not in {"activate", "rollback"}
            or pointer.sequence < 1
            or not _is_digest(pointer.version_sha256)
            or (
                pointer.previous_version_sha256 is not None
                and not _is_digest(pointer.previous_version_sha256)
            )
            or pointer.pointer_sha256 != _digest(pointer._body())
        ):
            raise ValueError("Extended Core active pointer integrity mismatch")

    @staticmethod
    def _read_canonical_object(path: Path) -> dict[str, Any]:
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load Extended Core artifact {path.name}: {exc}") from exc
        if not isinstance(payload, dict) or raw != _canonical(payload):
            raise ValueError(f"Extended Core artifact {path.name} is not canonical")
        return payload

    @staticmethod
    def _write_immutable(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"Immutable Extended Core artifact conflicts: {path.name}")

    @staticmethod
    def _write_atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
