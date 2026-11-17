"""Deterministically bind an already-reviewed instruction corpus to BCNLT."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .bcnlt_source_pipeline import (
    PARSER_RUNTIME_IDENTITY_SHA256,
    PARSER_RUNTIME_NOT_APPLICABLE,
)
from .resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    _canonical_json,
    _encode_u16,
    _sha256_file,
    audit_task_database,
)
from .tokenizer import NativeTokenizerBinding

REVIEWED_INSTRUCTION_CURRICULUM_SCHEMA = (
    "bosscrafts-reviewed-instruction-curriculum-v1"
)
INSTRUCTION_RESPONSE_FAMILY = "instruction_to_response"
INSTRUCTION_LANGUAGE = "english"
INSTRUCTION_SEMANTIC_SPEC_SHA256 = hashlib.sha256(
    b"bosscrafts-reviewed-english-instruction-v1"
).hexdigest()
_SPLITS = ("train", "validation", "test")
_MAX_SOURCE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_id: str
    origin_split: str
    split: str
    instruction: str
    output: str
    source_name: str
    source_row_sha256: str
    provenance_sha256: str
    semantic_review_sha256: str
    normalized_prompt_sha256: str
    normalized_target_sha256: str


@dataclass(frozen=True, slots=True)
class _PackedCandidate:
    candidate: _Candidate
    example_id: str
    input_ids: tuple[int, ...]
    loss_start_index: int


@dataclass(frozen=True, slots=True)
class ReviewedInstructionCurriculumResult:
    database_path: Path
    database_identity_sha256: str
    content_sha256: str
    policy_sha256: str
    example_count: int
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    rejection_counts: dict[str, int]
    total_sequence_tokens: int
    maximum_sequence_tokens: int


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalized_prompt(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n")
    return " ".join(normalized.casefold().split())


def _normalized_target(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    return "\n".join(line.rstrip() for line in normalized.strip().split("\n"))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_content_digest(value: object) -> str:
    """Reproduce the Wizard-Phi manifest's JSON-content digest exactly."""
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return _hash_text(serialized)


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise ValueError(f"reviewed instruction source does not exist: {path.name}")
    size = path.stat().st_size
    if not 2 <= size <= _MAX_SOURCE_BYTES:
        raise ValueError(f"reviewed instruction source size is invalid: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"reviewed instruction source JSON is invalid: {path.name}"
        ) from error


def _load_manifest(root: Path) -> tuple[dict[str, object], Path, Path, str]:
    manifest_path = root / "manifest.json"
    raw_manifest = _load_json(manifest_path)
    if type(raw_manifest) is not dict:
        raise ValueError("reviewed instruction manifest is invalid")
    manifest = cast(dict[str, object], raw_manifest)
    train_path = root / "train.json"
    eval_path = root / "eval.json"
    expected = {
        "train": (train_path, manifest.get("train"), manifest.get("train_sha256")),
        "eval": (eval_path, manifest.get("eval"), manifest.get("eval_sha256")),
    }
    for name, (path, count, sha256) in expected.items():
        if type(count) is not int or count < 1 or not _is_sha256(sha256):
            raise ValueError(f"reviewed instruction {name} manifest entry is invalid")
        if not path.is_file() or not 2 <= path.stat().st_size <= _MAX_SOURCE_BYTES:
            raise ValueError(f"reviewed instruction {name} source is invalid")
    accepted = manifest.get("accepted")
    rejected = manifest.get("rejected")
    quality_review = manifest.get("quality_review")
    if (
        type(accepted) is not int
        or accepted != int(cast(int, manifest["train"])) + int(cast(int, manifest["eval"]))
        or rejected != 0
        or type(quality_review) is not str
        or not quality_review.strip()
    ):
        raise ValueError("reviewed instruction manifest admission evidence is invalid")
    return manifest, train_path, eval_path, _sha256_file(manifest_path)


def _derived_eval_split(source_id: str) -> str:
    return "validation" if int(source_id[-16:], 16) % 2 == 0 else "test"


def _candidate_from_row(row: object, *, origin_split: str) -> _Candidate:
    if type(row) is not dict:
        raise ValueError("reviewed instruction row is not an object")
    source = cast(dict[str, object], row)
    source_id = source.get("id")
    instruction = source.get("instruction")
    output = source.get("output")
    source_name = source.get("source")
    seed_id = source.get("seed_id")
    constraints = source.get("constraints")
    mutations = source.get("mutations")
    provenance = source.get("provenance")
    semantic_review = source.get("semantic_review")
    if (
        type(source_id) is not str
        or not _is_sha256(source_id)
        or type(instruction) is not str
        or not instruction.strip()
        or type(output) is not str
        or not output.strip()
        or type(source_name) is not str
        or not source_name.strip()
        or type(seed_id) is not str
        or not seed_id.strip()
        or type(constraints) is not list
        or any(type(item) is not str or not item.strip() for item in constraints)
        or type(mutations) is not list
        or bool(source.get("generation_truncated"))
        or source.get("review_status") != "approved"
        or type(provenance) is not dict
        or not provenance
        or type(semantic_review) is not dict
        or not semantic_review
    ):
        raise ValueError("reviewed instruction row admission evidence is invalid")
    for score_name in ("reward_score", "reward_floor"):
        score = semantic_review.get(score_name)
        if score is not None and (
            type(score) not in {int, float} or not math.isfinite(float(score))
        ):
            raise ValueError("reviewed instruction semantic score is invalid")
    rendered_prompt = instruction.strip()
    if constraints:
        rendered_prompt += "\n\nRequirements:\n" + "\n".join(
            f"- {item}" for item in cast(list[str], constraints)
        )
    canonical_row = _canonical_json(source)
    prompt_normalized = _normalized_prompt(rendered_prompt)
    target_normalized = _normalized_target(output)
    if not prompt_normalized or not target_normalized:
        raise ValueError("reviewed instruction normalized content is empty")
    return _Candidate(
        source_id=source_id,
        origin_split=origin_split,
        split=(
            "train"
            if origin_split == "train"
            else _derived_eval_split(source_id)
        ),
        instruction=rendered_prompt,
        output=output,
        source_name=source_name,
        source_row_sha256=_hash_text(canonical_row),
        provenance_sha256=_hash_text(_canonical_json(provenance)),
        semantic_review_sha256=_hash_text(_canonical_json(semantic_review)),
        normalized_prompt_sha256=_hash_text(prompt_normalized),
        normalized_target_sha256=_hash_text(target_normalized),
    )


def _screen_prompt_groups(
    candidates: list[_Candidate], rejection_counts: Counter[str]
) -> tuple[list[_Candidate], list[tuple[str, str, str]]]:
    groups: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.normalized_prompt_sha256].append(candidate)
    selected: list[_Candidate] = []
    rejected: list[tuple[str, str, str]] = []
    for group in groups.values():
        targets = {item.normalized_target_sha256 for item in group}
        splits = {item.split for item in group}
        if len(targets) > 1:
            reason = "conflicting_normalized_prompt"
            rejected.extend((item.source_id, item.origin_split, reason) for item in group)
            rejection_counts[reason] += len(group)
            continue
        if len(splits) > 1:
            reason = "cross_split_normalized_prompt"
            rejected.extend((item.source_id, item.origin_split, reason) for item in group)
            rejection_counts[reason] += len(group)
            continue
        chosen = min(group, key=lambda item: item.source_id)
        selected.append(chosen)
        duplicates = [item for item in group if item.source_id != chosen.source_id]
        rejected.extend(
            (item.source_id, item.origin_split, "duplicate_normalized_pair")
            for item in duplicates
        )
        if duplicates:
            rejection_counts["duplicate_normalized_pair"] += len(duplicates)
    selected.sort(key=lambda item: item.source_id)
    rejected.sort()
    return selected, rejected


def _pack_candidate(
    binding: NativeTokenizerBinding,
    candidate: _Candidate,
    *,
    max_sequence: int,
) -> tuple[_PackedCandidate | None, str | None]:
    ids = binding.structural_token_ids
    try:
        prompt_payload = binding.encode_payload(candidate.instruction)
        target_payload = binding.encode_payload(candidate.output)
    except ValueError:
        return None, "reserved_protocol_token"
    prompt_ids = [
        ids["<|bcnlt:task:mixed_instruction|>"],
        ids["<|bcnlt:stream:begin|>"],
        ids["<|bcnlt:mode:text|>"],
        *prompt_payload,
        ids["<|bcnlt:stream:end|>"],
    ]
    target_ids = [
        ids["<|bcnlt:stream:begin|>"],
        ids["<|bcnlt:mode:text|>"],
        *target_payload,
        ids["<|bcnlt:stream:end|>"],
        ids["<|bcnlt:task:end|>"],
    ]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_sequence:
        return None, "sequence_too_long"
    identity = {
        "document_id": candidate.source_id,
        "family": INSTRUCTION_RESPONSE_FAMILY,
        "record_index": 0,
        "input_ids_sha256": _hash_text(_canonical_json(input_ids)),
    }
    return (
        _PackedCandidate(
            candidate=candidate,
            example_id=_hash_text(_canonical_json(identity)),
            input_ids=tuple(input_ids),
            loss_start_index=len(prompt_ids),
        ),
        None,
    )


def build_reviewed_instruction_curriculum(
    binding: NativeTokenizerBinding,
    *,
    source_root: str | Path,
    output_database: str | Path,
    max_sequence: int = 2048,
) -> ReviewedInstructionCurriculumResult:
    """Materialize reviewed rows without making any new semantic judgment."""
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if type(max_sequence) is not int or not 256 <= max_sequence <= 65_536:
        raise ValueError("max_sequence must be between 256 and 65536")
    root = Path(source_root).resolve()
    output = Path(output_database).resolve()
    if output.exists():
        raise ValueError("reviewed instruction curriculum output already exists")
    manifest, train_path, eval_path, manifest_sha256 = _load_manifest(root)
    source_hashes = {
        "train": cast(str, manifest["train_sha256"]),
        "eval": cast(str, manifest["eval_sha256"]),
    }
    source_file_hashes = {
        "train": _sha256_file(train_path),
        "eval": _sha256_file(eval_path),
    }

    candidates: list[_Candidate] = []
    source_ids: set[str] = set()
    for origin_split, path in (("train", train_path), ("eval", eval_path)):
        raw_rows = _load_json(path)
        if type(raw_rows) is not list:
            raise ValueError(f"reviewed instruction {origin_split} source is not an array")
        if _source_content_digest(raw_rows) != source_hashes[origin_split]:
            raise ValueError(
                f"reviewed instruction {origin_split} source hash mismatch"
            )
        expected_count = int(cast(int, manifest[origin_split]))
        if len(raw_rows) != expected_count:
            raise ValueError(f"reviewed instruction {origin_split} count mismatch")
        for raw_row in raw_rows:
            candidate = _candidate_from_row(raw_row, origin_split=origin_split)
            if candidate.source_id in source_ids:
                raise ValueError("reviewed instruction source ID is duplicated")
            source_ids.add(candidate.source_id)
            candidates.append(candidate)

    rejection_counts: Counter[str] = Counter()
    screened, rejected = _screen_prompt_groups(candidates, rejection_counts)
    packed: list[_PackedCandidate] = []
    for candidate in screened:
        item, reason = _pack_candidate(binding, candidate, max_sequence=max_sequence)
        if item is None:
            if reason is None:
                raise AssertionError("rejected packed candidate lacks a reason")
            rejected.append((candidate.source_id, candidate.origin_split, reason))
            rejection_counts[reason] += 1
        else:
            packed.append(item)
    packed.sort(key=lambda item: item.example_id)
    rejected.sort()
    split_counts = Counter(item.candidate.split for item in packed)
    if set(split_counts) != set(_SPLITS) or any(split_counts[split] < 1 for split in _SPLITS):
        raise ValueError("reviewed instruction curriculum lacks a required split")
    if len({item.example_id for item in packed}) != len(packed):
        raise ValueError("reviewed instruction example identity is duplicated")

    policy = {
        "schema_version": REVIEWED_INSTRUCTION_CURRICULUM_SCHEMA,
        "family": INSTRUCTION_RESPONSE_FAMILY,
        "language": INSTRUCTION_LANGUAGE,
        "max_sequence": max_sequence,
        "prompt_mode": "text",
        "target_mode": "text",
        "source_manifest_sha256": manifest_sha256,
        "source_content_hashes": source_hashes,
        "source_file_hashes": source_file_hashes,
        "eval_split_policy": "sha256-source-id-low64-parity-v1",
        "deduplication_policy": "nfkc-prompt-nfc-target-global-fail-closed-v1",
        "semantic_selection": "none; preserve prior approved semantic review evidence",
        "source_id_policy": "opaque historical sha256; final rows anchored by manifest digest",
    }
    policy_sha256 = _hash_text(_canonical_json(policy))
    metadata = {
        "schema_version": TASK_DATABASE_SCHEMA,
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "vocabulary_size": str(binding.vocab_size),
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "maximum_sequence_tokens": str(max_sequence),
        "packing_complete": "true",
        "curriculum_schema": REVIEWED_INSTRUCTION_CURRICULUM_SCHEMA,
        "curriculum_policy_sha256": policy_sha256,
        "source_manifest_sha256": manifest_sha256,
        "source_train_sha256": source_hashes["train"],
        "source_eval_sha256": source_hashes["eval"],
        "source_train_file_sha256": source_file_hashes["train"],
        "source_eval_file_sha256": source_file_hashes["eval"],
        "instruction_family": INSTRUCTION_RESPONSE_FAMILY,
        "language": INSTRUCTION_LANGUAGE,
        "split_counts": _canonical_json(dict(sorted(split_counts.items()))),
        "rejection_counts": _canonical_json(dict(sorted(rejection_counts.items()))),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE tasks(
                       example_id TEXT PRIMARY KEY,document_id TEXT NOT NULL,
                       language TEXT NOT NULL,family TEXT NOT NULL,
                       record_index INTEGER NOT NULL,split TEXT NOT NULL,
                       input_ids_zlib BLOB NOT NULL,token_count INTEGER NOT NULL,
                       loss_start_index INTEGER NOT NULL,
                       semantic_spec_sha256 TEXT NOT NULL,
                       parser_runtime_identity_sha256 TEXT NOT NULL,
                       source_sha256 TEXT NOT NULL)"""
            )
            connection.execute(
                "CREATE INDEX tasks_cell ON tasks(split,language,family)"
            )
            connection.execute(
                """CREATE TABLE instruction_provenance(
                       example_id TEXT PRIMARY KEY,source_id TEXT NOT NULL,
                       origin_split TEXT NOT NULL,source_name TEXT NOT NULL,
                       source_row_sha256 TEXT NOT NULL,
                       provenance_sha256 TEXT NOT NULL,
                       semantic_review_sha256 TEXT NOT NULL,
                       normalized_prompt_sha256 TEXT NOT NULL,
                       normalized_target_sha256 TEXT NOT NULL)"""
            )
            connection.execute(
                """CREATE TABLE rejections(
                       source_id TEXT PRIMARY KEY,origin_split TEXT NOT NULL,
                       reason_code TEXT NOT NULL)"""
            )
            connection.executemany(
                "INSERT INTO metadata VALUES(?,?)", sorted(metadata.items())
            )
            for item in packed:
                candidate = item.candidate
                connection.execute(
                    "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        item.example_id,
                        candidate.source_id,
                        INSTRUCTION_LANGUAGE,
                        INSTRUCTION_RESPONSE_FAMILY,
                        0,
                        candidate.split,
                        zlib.compress(_encode_u16(list(item.input_ids)), 6),
                        len(item.input_ids),
                        item.loss_start_index,
                        INSTRUCTION_SEMANTIC_SPEC_SHA256,
                        PARSER_RUNTIME_NOT_APPLICABLE,
                        candidate.source_row_sha256,
                    ),
                )
                connection.execute(
                    "INSERT INTO instruction_provenance VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        item.example_id,
                        candidate.source_id,
                        candidate.origin_split,
                        candidate.source_name,
                        candidate.source_row_sha256,
                        candidate.provenance_sha256,
                        candidate.semantic_review_sha256,
                        candidate.normalized_prompt_sha256,
                        candidate.normalized_target_sha256,
                    ),
                )
            connection.executemany("INSERT INTO rejections VALUES(?,?,?)", rejected)
            connection.commit()
            connection.execute("VACUUM")
        finally:
            connection.close()
        audit = audit_task_database(binding, temporary)
        if audit.example_count != len(packed) or audit.split_counts != dict(split_counts):
            raise ValueError("reviewed instruction curriculum failed its count audit")
        if (
            _sha256_file(train_path) != source_file_hashes["train"]
            or _sha256_file(eval_path) != source_file_hashes["eval"]
            or _sha256_file(root / "manifest.json") != manifest_sha256
        ):
            raise ValueError("reviewed instruction sources changed during build")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if output.exists():
            raise ValueError("reviewed instruction curriculum output already exists")
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ValueError(
                "reviewed instruction curriculum output already exists"
            ) from error
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.unlink()
    source_counts = Counter(item.candidate.source_name for item in packed)
    return ReviewedInstructionCurriculumResult(
        database_path=output,
        database_identity_sha256=audit.database_identity_sha256,
        content_sha256=audit.content_sha256,
        policy_sha256=policy_sha256,
        example_count=audit.example_count,
        split_counts=audit.split_counts,
        source_counts=dict(sorted(source_counts.items())),
        rejection_counts=dict(sorted(rejection_counts.items())),
        total_sequence_tokens=audit.total_sequence_tokens,
        maximum_sequence_tokens=audit.maximum_sequence_tokens,
    )
