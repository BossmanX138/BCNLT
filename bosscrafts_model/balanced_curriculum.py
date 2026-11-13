"""Deterministic balanced task curriculum built directly from admitted sources."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
from typing import cast
import zlib

from .bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from .bcnlt_adapter import MAX_CANONICAL_OUTPUT_BYTES
from .compression import decompress_bounded
from .compact_semantic_wire import COMPACT_SEMANTIC_WIRE_SCHEMA
from .compact_task_packing import CompactCurrentCorpusTaskPacker
from .current_task_packing import CurrentCorpusTaskPacker
from .resumable_current_corpus import RESUMABLE_SCHEMA
from .resumable_task_packing import (
    FULL_SEMANTIC_WIRE_SCHEMA,
    TASK_DATABASE_SCHEMA,
    _canonical_json,
    _decode_u16,
    _encode_u16,
    _sha256_file,
    audit_task_database,
)
from .tokenizer import NativeTokenizerBinding


CURRICULUM_SCHEMA = "bosscrafts-balanced-curriculum-v2"
TASK_FAMILIES = (
    "code_to_semantic",
    "semantic_continuation",
    "semantic_to_code",
)
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class BalancedCurriculumStatus:
    database_path: Path
    example_count: int
    selected_document_count: int
    processed_document_count: int
    total_sequence_tokens: int
    cell_counts: dict[str, int]
    database_identity_sha256: str
    curriculum_policy_sha256: str


def _ro_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _validate_policy(
    languages: tuple[str, ...], quotas: dict[str, int]
) -> tuple[str, ...]:
    if (
        type(languages) is not tuple
        or not languages
        or any(type(language) is not str for language in languages)
        or len(set(languages)) != len(languages)
    ):
        raise ValueError("languages must be a nonempty unique tuple")
    unsupported = sorted(set(languages) - set(CURRENT_RUNTIME_SPEC_SHA256))
    if unsupported:
        raise ValueError(f"unsupported curriculum languages: {unsupported}")
    if (
        type(quotas) is not dict
        or set(quotas) != set(SPLITS)
        or any(type(value) is not int or not 1 <= value <= 100_000 for value in quotas.values())
    ):
        raise ValueError("quotas must define positive train, validation and test limits")
    return tuple(sorted(languages))


def build_balanced_curriculum(
    binding: NativeTokenizerBinding,
    admission_database: str | Path,
    source_catalog: str | Path,
    database_path: str | Path,
    *,
    languages: tuple[str, ...] = tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256)),
    quotas: dict[str, int] | None = None,
    max_sequence: int = 2048,
    semantic_wire: str = "compact_v2",
) -> BalancedCurriculumStatus:
    """Build an atomic, exact-quota task database without expanding the full corpus."""
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if quotas is None:
        quotas = {"train": 2048, "validation": 128, "test": 128}
    languages = _validate_policy(languages, quotas)
    if type(max_sequence) is not int or max_sequence < 64:
        raise ValueError("max_sequence must be an integer of at least 64")
    if semantic_wire not in {"compact_v2", "full_v3"}:
        raise ValueError("semantic_wire must be compact_v2 or full_v3")
    semantic_wire_schema = (
        COMPACT_SEMANTIC_WIRE_SCHEMA
        if semantic_wire == "compact_v2"
        else FULL_SEMANTIC_WIRE_SCHEMA
    )

    admission_path = Path(admission_database).resolve()
    catalog_path = Path(source_catalog).resolve()
    destination = Path(database_path).resolve()
    if not admission_path.is_file():
        raise ValueError("admission database does not exist")
    if not catalog_path.is_file():
        raise ValueError("source catalog does not exist")
    if destination.exists():
        raise ValueError("balanced curriculum database already exists")

    admission = _ro_connect(admission_path)
    try:
        metadata = dict(admission.execute("SELECT key, value FROM metadata"))
        if metadata.get("schema_version") != RESUMABLE_SCHEMA:
            raise ValueError("admission database schema mismatch")
        if metadata.get("catalog_exhausted") != "true":
            raise ValueError("full admission is not complete")
        if (
            metadata.get("binding_sha256") != binding.binding_sha256
            or metadata.get("tokenizer_sha256") != binding.tokenizer_sha256
            or metadata.get("vocabulary_size") != str(binding.vocab_size)
        ):
            raise ValueError("admission tokenizer lineage mismatch")
        expected_specs = _canonical_json(CURRENT_RUNTIME_SPEC_SHA256)
        if metadata.get("semantic_spec_sha256") != expected_specs:
            raise ValueError("admission semantic specification mismatch")
        counts = admission.execute(
            """SELECT SUM(CASE WHEN status='admitted' THEN 1 ELSE 0 END),
                      SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END)
               FROM documents"""
        ).fetchone()
    finally:
        admission.close()
    if counts is None:
        raise RuntimeError("admission database count query returned no row")
    admitted_count = int(counts[0] or 0)
    rejected_count = int(counts[1] or 0)
    if admitted_count == 0:
        raise ValueError("completed admission contains no admitted documents")
    if _sha256_file(catalog_path) != metadata.get("source_catalog_sha256"):
        raise ValueError("source catalog SHA-256 mismatch")

    admission_identity = hashlib.sha256(
        _canonical_json(
            {
                "metadata": metadata,
                "admitted_count": admitted_count,
                "rejected_count": rejected_count,
            }
        ).encode("utf-8")
    ).hexdigest()
    policy = {
        "languages": languages,
        "maximum_sequence_tokens": max_sequence,
        "semantic_wire_schema_version": semantic_wire_schema,
        "quotas_per_language_family": dict(sorted(quotas.items())),
        "schema_version": CURRICULUM_SCHEMA,
        "selection": (
            "admitted documents by language then document_id; examples by example_id; "
            "exact split-language-family quotas"
        ),
        "task_families": TASK_FAMILIES,
    }
    policy_sha256 = hashlib.sha256(_canonical_json(policy).encode("utf-8")).hexdigest()
    expected_cells = {
        (split, language, family): quotas[split]
        for split in SPLITS
        for language in languages
        for family in TASK_FAMILIES
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    output = sqlite3.connect(temporary)
    output.execute("PRAGMA journal_mode=DELETE")
    output.execute("PRAGMA synchronous=FULL")
    admission = _ro_connect(admission_path)
    catalog = _ro_connect(catalog_path)
    packer = (
        CompactCurrentCorpusTaskPacker(binding, max_sequence=max_sequence)
        if semantic_wire == "compact_v2"
        else CurrentCorpusTaskPacker(binding, max_sequence=max_sequence)
    )
    cell_counts: Counter[tuple[str, str, str]] = Counter()
    selected_documents: set[str] = set()
    processed_documents = 0
    total_sequence_tokens = 0
    try:
        output.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        output.execute(
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
        output.execute("CREATE INDEX tasks_cell ON tasks(split, language, family)")
        output_metadata = {
            "binding_sha256": binding.binding_sha256,
            "curriculum_policy_sha256": policy_sha256,
            "curriculum_schema_version": CURRICULUM_SCHEMA,
            "maximum_sequence_tokens": str(max_sequence),
            "packing_complete": "true",
            "review_manifest_sha256": metadata["review_manifest_sha256"],
            "schema_version": TASK_DATABASE_SCHEMA,
            "semantic_wire_schema_version": semantic_wire_schema,
            "semantic_spec_sha256": expected_specs,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "source_admission_identity_sha256": admission_identity,
            "source_catalog_sha256": metadata["source_catalog_sha256"],
            "tokenizer_sha256": binding.tokenizer_sha256,
            "vocabulary_size": str(binding.vocab_size),
        }
        output.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)",
            sorted(output_metadata.items()),
        )

        for language in languages:
            rows = admission.execute(
                """SELECT document_id, content_sha256, relative_path,
                          semantic_spec_sha256, parser_runtime_identity_sha256,
                          canonical_sha256,
                          canonical_jsonl_zlib, semantic_ids_zlib,
                          semantic_record_count, semantic_token_count
                   FROM documents
                   WHERE status='admitted' AND language=?
                   ORDER BY document_id""",
                (language,),
            )
            for admission_row in rows:
                document_id = str(admission_row[0])
                split = CurrentCorpusTaskPacker._split(document_id)
                if all(
                    cell_counts[(split, language, family)] >= quotas[split]
                    for family in TASK_FAMILIES
                ):
                    continue
                source_row = catalog.execute(
                    """SELECT canonical_language, content, relative_path, content_sha256
                       FROM documents WHERE document_id=?""",
                    (document_id,),
                ).fetchone()
                if source_row is None:
                    raise ValueError("admitted document is absent from source catalog")
                source_language, source, relative_path, source_sha256 = map(str, source_row)
                if (
                    source_language != language
                    or relative_path != admission_row[2]
                    or source_sha256 != admission_row[1]
                    or hashlib.sha256(source.encode("utf-8")).hexdigest() != source_sha256
                ):
                    raise ValueError("source catalog document identity mismatch")
                if admission_row[3] != CURRENT_RUNTIME_SPEC_SHA256[language]:
                    raise ValueError("admitted document semantic specification mismatch")
                if admission_row[4] != PARSER_RUNTIME_IDENTITY_SHA256:
                    raise ValueError("admitted document parser runtime identity mismatch")
                canonical_jsonl = decompress_bounded(
                    admission_row[6],
                    maximum_output_bytes=MAX_CANONICAL_OUTPUT_BYTES,
                ).decode("utf-8")
                if hashlib.sha256(canonical_jsonl.encode("utf-8")).hexdigest() != admission_row[5]:
                    raise ValueError("admission canonical JSONL SHA-256 mismatch")
                expected_semantic_bytes = int(admission_row[9]) * 2
                semantic_ids = _decode_u16(
                    decompress_bounded(
                        admission_row[7],
                        maximum_output_bytes=expected_semantic_bytes,
                    )
                )
                if len(semantic_ids) != int(admission_row[9]):
                    raise ValueError("admission semantic token count mismatch")
                row = {
                    "document_id": document_id,
                    "language": language,
                    "relative_path": relative_path,
                    "source": source,
                    "source_sha256": source_sha256,
                    "binding_sha256": binding.binding_sha256,
                    "tokenizer_sha256": binding.tokenizer_sha256,
                    "semantic_spec_sha256": str(admission_row[3]),
                    "parser_runtime_identity_sha256": str(admission_row[4]),
                    "canonical_sha256": str(admission_row[5]),
                    "canonical_jsonl": canonical_jsonl,
                    "semantic_record_count": int(admission_row[8]),
                    "semantic_token_ids": semantic_ids,
                }
                examples, _ = packer.pack_row(row)
                processed_documents += 1
                for item in sorted(examples, key=lambda example: str(example["example_id"])):
                    cell = (str(item["split"]), language, str(item["family"]))
                    target = expected_cells.get(cell)
                    if target is None or cell_counts[cell] >= target:
                        continue
                    raw_input_ids = item.get("input_ids")
                    if (
                        type(raw_input_ids) is not list
                        or any(type(token_id) is not int for token_id in raw_input_ids)
                    ):
                        raise ValueError("packed task input_ids must be an integer list")
                    input_ids = cast(list[int], raw_input_ids)
                    output.execute(
                        "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            item["example_id"],
                            document_id,
                            language,
                            item["family"],
                            item["record_index"],
                            item["split"],
                            zlib.compress(_encode_u16(input_ids), 6),
                            len(input_ids),
                            item["loss_start_index"],
                            item["semantic_spec_sha256"],
                            item["parser_runtime_identity_sha256"],
                            source_sha256,
                        ),
                    )
                    cell_counts[cell] += 1
                    total_sequence_tokens += len(input_ids)
                    selected_documents.add(document_id)
                if all(
                    cell_counts[(candidate_split, language, family)]
                    >= quotas[candidate_split]
                    for candidate_split in SPLITS
                    for family in TASK_FAMILIES
                ):
                    break

        missing = {
            "|".join(cell): target - cell_counts[cell]
            for cell, target in expected_cells.items()
            if cell_counts[cell] != target
        }
        if missing:
            raise ValueError(f"balanced curriculum quotas could not be filled: {missing}")
        output.commit()
    except BaseException:
        output.close()
        admission.close()
        catalog.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        output.close()
        admission.close()
        catalog.close()

    try:
        audit = audit_task_database(binding, temporary)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.replace(destination)
    return BalancedCurriculumStatus(
        database_path=destination,
        example_count=audit.example_count,
        selected_document_count=len(selected_documents),
        processed_document_count=processed_documents,
        total_sequence_tokens=total_sequence_tokens,
        cell_counts={
            "|".join(cell): cell_counts[cell]
            for cell in sorted(expected_cells)
        },
        database_identity_sha256=audit.database_identity_sha256,
        curriculum_policy_sha256=policy_sha256,
    )
