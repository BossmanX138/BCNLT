"""Build a tiny current-lineage SQLite database for full-model hardware gates."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import zlib
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from .bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from .current_task_packing import (
    EXAMPLE_SCHEMA,
    FAMILIES,
    PACKED_MANIFEST_SCHEMA,
    CurrentCorpusTaskPacker,
)
from .resumable_task_packing import TASK_DATABASE_SCHEMA, audit_task_database
from .tokenizer import NativeTokenizerBinding


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _encode_u16(values: list[int]) -> bytes:
    encoded = array("H", values)
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


class _SelectedGateExample(TypedDict):
    example_id: str
    document_id: str
    language: str
    family: str
    record_index: int
    split: str
    input_ids: list[int]
    loss_start_index: int
    target_token_count: int
    semantic_spec_sha256: str
    parser_runtime_identity_sha256: str
    source_sha256: str


_EXAMPLE_KEYS = frozenset(_SelectedGateExample.__required_keys__) | {
    "binding_sha256",
    "schema_version",
    "tokenizer_sha256",
}


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_json_object(text: str, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if type(payload) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return payload


@dataclass(frozen=True)
class TrainingGateDatabaseStatus:
    database_path: Path
    imported_count: int
    stale_spec_skipped_count: int
    database_identity_sha256: str
    gate_only: bool


def build_training_gate_database(
    binding: NativeTokenizerBinding,
    source_manifest_path: str | Path,
    database_path: str | Path,
    *,
    max_examples: int = 6,
) -> TrainingGateDatabaseStatus:
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if type(max_examples) is not int or not 1 <= max_examples <= 64:
        raise ValueError("max_examples must be between 1 and 64")
    manifest_path = Path(source_manifest_path).resolve()
    destination = Path(database_path).resolve()
    if destination.exists():
        raise ValueError("training gate database already exists")
    try:
        manifest_bytes = manifest_path.read_bytes()
        if len(manifest_bytes) > 1024 * 1024:
            raise ValueError("source task manifest exceeds one MiB")
        manifest_text = manifest_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("source task manifest cannot be read") from error
    manifest = _parse_json_object(manifest_text, label="source task manifest")
    claimed_manifest_sha256 = manifest.pop("manifest_sha256", None)
    actual_manifest_sha256 = hashlib.sha256(
        _canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    if claimed_manifest_sha256 != actual_manifest_sha256:
        raise ValueError("source task manifest SHA-256 mismatch")
    if (
        manifest.get("schema_version") != PACKED_MANIFEST_SCHEMA
        or manifest.get("binding_sha256") != binding.binding_sha256
        or manifest.get("tokenizer_sha256") != binding.tokenizer_sha256
        or manifest.get("parser_runtime_identity_sha256")
        != PARSER_RUNTIME_IDENTITY_SHA256
    ):
        raise ValueError("source task manifest lineage mismatch")
    examples_file = manifest.get("examples_file")
    if (
        type(examples_file) is not str
        or not examples_file
        or Path(examples_file).name != examples_file
    ):
        raise ValueError("source task examples path is invalid")
    examples_path = (manifest_path.parent / examples_file).resolve()
    if not examples_path.is_file():
        raise ValueError("source task examples file is missing")
    if _sha256_file(examples_path) != manifest.get("examples_sha256"):
        raise ValueError("source task examples SHA-256 mismatch")
    maximum_sequence_tokens = manifest.get("maximum_sequence_tokens")
    if type(maximum_sequence_tokens) is not int or maximum_sequence_tokens < 2:
        raise ValueError("source task maximum sequence is invalid")

    selected: list[_SelectedGateExample] = []
    selected_cells: set[tuple[str, str]] = set()
    stale_spec_skipped_count = 0
    with examples_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = _parse_json_object(line, label="source task example")
            if set(row) != _EXAMPLE_KEYS or row.get("schema_version") != EXAMPLE_SCHEMA:
                raise ValueError("source task example schema mismatch")
            document_id = row.get("document_id")
            source_sha256 = row.get("source_sha256")
            split = row.get("split")
            family = row.get("family")
            if not _is_sha256(document_id) or not _is_sha256(source_sha256):
                raise ValueError("source task example digest is invalid")
            if (
                split not in {"train", "validation", "test"}
                or split != CurrentCorpusTaskPacker._split(cast(str, document_id))
            ):
                raise ValueError("source task example split is invalid")
            if family not in FAMILIES:
                raise ValueError("source task family is invalid")
            if split != "train":
                continue
            language = row.get("language")
            if (
                type(language) is not str
                or CURRENT_RUNTIME_SPEC_SHA256.get(language)
                != row.get("semantic_spec_sha256")
            ):
                stale_spec_skipped_count += 1
                continue
            cell = (language, family)
            if cell in selected_cells:
                continue
            input_ids = row.get("input_ids")
            loss_start = row.get("loss_start_index")
            target_token_count = row.get("target_token_count")
            record_index = row.get("record_index")
            expected_task_start = binding.structural_token_ids[
                f"<|bcnlt:task:{family}|>"
            ]
            expected_task_end = binding.structural_token_ids["<|bcnlt:task:end|>"]
            if (
                row.get("binding_sha256") != binding.binding_sha256
                or row.get("tokenizer_sha256") != binding.tokenizer_sha256
                or row.get("parser_runtime_identity_sha256")
                != PARSER_RUNTIME_IDENTITY_SHA256
                or type(input_ids) is not list
                or not 2 <= len(input_ids) <= maximum_sequence_tokens
                or any(type(token_id) is not int or not 0 <= token_id < binding.vocab_size for token_id in input_ids)
                or input_ids[0] != expected_task_start
                or input_ids[-1] != expected_task_end
                or type(loss_start) is not int
                or not 0 < loss_start < len(input_ids)
                or type(target_token_count) is not int
                or target_token_count != len(input_ids) - loss_start
                or type(record_index) is not int
                or record_index < 0
            ):
                raise ValueError("source task example is invalid")
            input_ids_sha256 = hashlib.sha256(
                _canonical_json(input_ids).encode("utf-8")
            ).hexdigest()
            identity = {
                "document_id": row["document_id"],
                "family": family,
                "record_index": record_index,
                "input_ids_sha256": input_ids_sha256,
            }
            expected_example_id = hashlib.sha256(
                _canonical_json(identity).encode("utf-8")
            ).hexdigest()
            if row.get("example_id") != expected_example_id:
                raise ValueError("source task example identity mismatch")
            selected.append(cast(_SelectedGateExample, row))
            selected_cells.add(cell)
            if len(selected) == max_examples:
                break
    if not selected:
        raise ValueError("no current-lineage train examples were found")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE tasks (
                       example_id TEXT PRIMARY KEY,
                       document_id TEXT NOT NULL,
                       language TEXT NOT NULL,
                       family TEXT NOT NULL,
                       record_index INTEGER NOT NULL,
                       split TEXT NOT NULL,
                       input_ids_zlib BLOB NOT NULL,
                       token_count INTEGER NOT NULL,
                       loss_start_index INTEGER NOT NULL,
                       semantic_spec_sha256 TEXT NOT NULL,
                       parser_runtime_identity_sha256 TEXT NOT NULL,
                       source_sha256 TEXT NOT NULL
                   )"""
            )
            metadata = {
                "binding_sha256": binding.binding_sha256,
                "gate_only": "true",
                "maximum_sequence_tokens": str(maximum_sequence_tokens),
                "packing_complete": "true",
                "schema_version": TASK_DATABASE_SCHEMA,
                "semantic_spec_sha256": _canonical_json(CURRENT_RUNTIME_SPEC_SHA256),
                "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                "source_pilot_manifest_sha256": str(claimed_manifest_sha256),
                "tokenizer_sha256": binding.tokenizer_sha256,
                "vocabulary_size": str(binding.vocab_size),
            }
            connection.executemany(
                "INSERT INTO metadata(key,value) VALUES(?,?)", sorted(metadata.items())
            )
            for selected_row in selected:
                input_ids = selected_row["input_ids"]
                connection.execute(
                    "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        selected_row["example_id"],
                        selected_row["document_id"],
                        selected_row["language"],
                        selected_row["family"],
                        selected_row["record_index"],
                        selected_row["split"],
                        zlib.compress(_encode_u16(input_ids), level=9),
                        len(input_ids),
                        selected_row["loss_start_index"],
                        selected_row["semantic_spec_sha256"],
                        selected_row["parser_runtime_identity_sha256"],
                        selected_row["source_sha256"],
                    ),
                )
            connection.commit()
        finally:
            connection.close()
        audit = audit_task_database(binding, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if destination.exists():
            raise ValueError("training gate database already exists")
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError("training gate database already exists") from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return TrainingGateDatabaseStatus(
        database_path=destination,
        imported_count=len(selected),
        stale_spec_skipped_count=stale_spec_skipped_count,
        database_identity_sha256=audit.database_identity_sha256,
        gate_only=True,
    )
