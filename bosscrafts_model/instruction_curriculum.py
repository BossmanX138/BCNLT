"""Held-out-safe Python instruction-to-code curriculum derived from admitted sources."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import textwrap
from typing import cast
import zlib

from .bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
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


INSTRUCTION_CURRICULUM_SCHEMA = "bosscrafts-instruction-curriculum-v1"
INSTRUCTION_FAMILY = "instruction_to_code"
SPLITS = ("train", "validation", "test")
DEFAULT_QUOTAS = {"train": 4096, "validation": 256, "test": 256}
_WORD = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class PythonInstructionCandidate:
    document_id: str
    record_index: int
    relative_path: str
    repository_identity: str
    license_spdx: str
    source_sha256: str
    prompt: str
    target: str
    docstring: str


@dataclass(frozen=True)
class InstructionCurriculumStatus:
    database_path: Path
    database_identity_sha256: str
    example_count: int
    split_counts: dict[str, int]
    candidate_counts: dict[str, int]
    rejection_counts: dict[str, int]
    license_counts: dict[str, int]
    policy_sha256: str


def _normalize_words(text: str) -> tuple[str, ...]:
    return tuple(word.lower() for word in _WORD.findall(text))


class HumanEvalExclusion:
    """Reject exact or near-verbatim overlap with the pinned HumanEval benchmark."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.path = Path(dataset_path).resolve()
        if not self.path.is_file():
            raise ValueError("HumanEval exclusion dataset does not exist")
        self.sha256 = _sha256_file(self.path)
        prompt_words: list[frozenset[str]] = []
        prompt_joined: list[str] = []
        solution_joined: set[str] = set()
        task_ids: list[str] = []
        with gzip.open(self.path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if type(row) is not dict:
                    raise ValueError("HumanEval exclusion row is invalid")
                task_id = row.get("task_id")
                prompt = row.get("prompt")
                solution = row.get("canonical_solution")
                if (
                    type(task_id) is not str
                    or type(prompt) is not str
                    or type(solution) is not str
                ):
                    raise ValueError("HumanEval exclusion fields are invalid")
                words = _normalize_words(prompt)
                prompt_words.append(frozenset(words))
                prompt_joined.append(" ".join(words))
                solution_joined.add(" ".join(_normalize_words(solution)))
                task_ids.append(task_id)
        if task_ids != [f"HumanEval/{index}" for index in range(164)]:
            raise ValueError("HumanEval exclusion task order is invalid")
        self._prompt_words = tuple(prompt_words)
        self._prompt_joined = tuple(prompt_joined)
        self._solution_joined = frozenset(solution_joined)

    def overlaps(self, *, docstring: str, target: str) -> bool:
        candidate_words = frozenset(_normalize_words(docstring))
        candidate_joined = " ".join(_normalize_words(docstring))
        target_joined = " ".join(_normalize_words(target))
        if target_joined and target_joined in self._solution_joined:
            return True
        if len(candidate_joined) >= 40:
            for benchmark in self._prompt_joined:
                if candidate_joined in benchmark or benchmark in candidate_joined:
                    return True
        if len(candidate_words) < 8:
            return False
        for benchmark_words in self._prompt_words:
            union = candidate_words | benchmark_words
            if union and len(candidate_words & benchmark_words) / len(union) >= 0.80:
                return True
        return False


def _module_import_prefix(source: str, module: ast.Module) -> str:
    segments: list[str] = []
    size = 0
    for node in module.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        segment = ast.get_source_segment(source, node)
        if not segment:
            continue
        addition = segment.rstrip() + "\n"
        if size + len(addition) > 1024 or len(segments) >= 12:
            break
        segments.append(addition)
        size += len(addition)
    return "".join(segments)


def extract_python_instruction_candidates(
    *,
    document_id: str,
    relative_path: str,
    repository_identity: str,
    license_spdx: str,
    source_sha256: str,
    source: str,
) -> list[PythonInstructionCandidate]:
    """Extract standalone signature/docstring prompts and original function bodies."""
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError, TypeError, MemoryError):
        return []
    imports = _module_import_prefix(source, module)
    candidates: list[PythonInstructionCandidate] = []
    function_nodes = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    function_nodes.sort(key=lambda node: (node.lineno, node.col_offset))
    for record_index, node in enumerate(function_nodes):
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue
        value = node.body[0].value
        if not isinstance(value, ast.Constant) or type(value.value) is not str:
            continue
        docstring = value.value.strip()
        if not 20 <= len(docstring) <= 1600:
            continue
        segment = ast.get_source_segment(source, node)
        if not segment:
            continue
        # ast.get_source_segment removes indentation from the first line only.
        # Restore that prefix before dedenting so method bodies retain exactly
        # one function-body indentation level in the standalone example.
        function_text = textwrap.dedent(
            (" " * node.col_offset) + segment
        ).replace("\r\n", "\n")
        try:
            parsed = ast.parse(function_text)
        except SyntaxError:
            continue
        if len(parsed.body) != 1 or not isinstance(
            parsed.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        parsed_function = parsed.body[0]
        parsed_doc = parsed_function.body[0]
        if not isinstance(parsed_doc, ast.Expr) or parsed_doc.end_lineno is None:
            continue
        lines = function_text.splitlines(keepends=True)
        prompt_function = "".join(lines[: parsed_doc.end_lineno])
        target = "".join(lines[parsed_doc.end_lineno :])
        if not prompt_function.endswith("\n"):
            prompt_function += "\n"
        if not target.endswith("\n"):
            target += "\n"
        if not target.strip() or target.strip() in {"pass", "..."}:
            continue
        if len(target.splitlines()) > 160:
            continue
        prompt = imports + ("\n" if imports else "") + prompt_function
        try:
            ast.parse(prompt + target)
        except SyntaxError:
            continue
        candidates.append(
            PythonInstructionCandidate(
                document_id=document_id,
                record_index=record_index,
                relative_path=relative_path,
                repository_identity=repository_identity,
                license_spdx=license_spdx,
                source_sha256=source_sha256,
                prompt=prompt,
                target=target,
                docstring=docstring,
            )
        )
    return candidates


def _ro_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _fallback_split(document_id: str) -> str:
    bucket = int(
        hashlib.sha256(f"instruction-split-v1:{document_id}".encode("utf-8")).hexdigest()[:8],
        16,
    ) % 100
    if bucket < 5:
        return "test"
    if bucket < 10:
        return "validation"
    return "train"


def _example_identity(
    candidate: PythonInstructionCandidate, input_ids: list[int]
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "document_id": candidate.document_id,
                "family": INSTRUCTION_FAMILY,
                "record_index": candidate.record_index,
                "input_ids_sha256": hashlib.sha256(
                    _canonical_json(input_ids).encode("utf-8")
                ).hexdigest(),
            }
        ).encode("utf-8")
    ).hexdigest()


def build_instruction_curriculum(
    binding: NativeTokenizerBinding,
    *,
    source_catalog: str | Path,
    admission_database: str | Path,
    split_database: str | Path,
    humaneval_dataset: str | Path,
    output_database: str | Path,
    quotas: dict[str, int] | None = None,
    max_sequence: int = 2048,
) -> InstructionCurriculumStatus:
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    if set(quotas) != set(SPLITS) or any(
        type(value) is not int or value < 1 for value in quotas.values()
    ):
        raise ValueError("quotas must define positive train, validation and test limits")
    if type(max_sequence) is not int or max_sequence < 256:
        raise ValueError("max_sequence must be at least 256")

    catalog_path = Path(source_catalog).resolve()
    admission_path = Path(admission_database).resolve()
    split_path = Path(split_database).resolve()
    output_path = Path(output_database).resolve()
    if output_path.exists():
        raise ValueError("instruction curriculum output already exists")
    for name, path in (
        ("source catalog", catalog_path),
        ("admission database", admission_path),
        ("split database", split_path),
    ):
        if not path.is_file():
            raise ValueError(f"{name} does not exist")

    exclusion = HumanEvalExclusion(humaneval_dataset)
    split_audit = audit_task_database(binding, split_path)
    split_connection = _ro_connect(split_path)
    try:
        split_rows = split_connection.execute(
            "SELECT document_id,MIN(split),COUNT(DISTINCT split) FROM tasks GROUP BY document_id"
        ).fetchall()
    finally:
        split_connection.close()
    if any(int(row[2]) != 1 for row in split_rows):
        raise ValueError("split database violates document isolation")
    preserved_splits = {str(row[0]): str(row[1]) for row in split_rows}

    admission = _ro_connect(admission_path)
    try:
        admission_metadata = dict(admission.execute("SELECT key,value FROM metadata"))
        admitted_ids = {
            str(row[0])
            for row in admission.execute(
                "SELECT document_id FROM documents WHERE status='admitted' AND language='python'"
            )
        }
    finally:
        admission.close()
    if admission_metadata.get("catalog_exhausted") != "true":
        raise ValueError("source admission is incomplete")
    if admission_metadata.get("binding_sha256") != binding.binding_sha256:
        raise ValueError("source admission binding mismatch")
    catalog_sha256 = _sha256_file(catalog_path)
    if admission_metadata.get("source_catalog_sha256") != catalog_sha256:
        raise ValueError("source catalog lineage mismatch")

    task_token = binding.structural_token_ids["<|bcnlt:task:mixed_instruction|>"]
    stream_begin = binding.structural_token_ids["<|bcnlt:stream:begin|>"]
    stream_end = binding.structural_token_ids["<|bcnlt:stream:end|>"]
    raw_mode = binding.structural_token_ids["<|bcnlt:mode:raw_code|>"]
    task_end = binding.structural_token_ids["<|bcnlt:task:end|>"]

    rows_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in SPLITS}
    rejection_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    catalog = _ro_connect(catalog_path)
    try:
        rows = catalog.execute(
            """SELECT document_id,relative_path,repository_identity,license_spdx,
                      content_sha256,content
               FROM documents WHERE canonical_language='python' ORDER BY document_id"""
        )
        for row in rows:
            document_id = str(row[0])
            if document_id not in admitted_ids:
                continue
            source = str(row[5])
            source_sha256 = str(row[4])
            if hashlib.sha256(source.encode("utf-8")).hexdigest() != source_sha256:
                raise ValueError("source catalog content SHA-256 mismatch")
            split = preserved_splits.get(document_id, _fallback_split(document_id))
            for candidate in extract_python_instruction_candidates(
                document_id=document_id,
                relative_path=str(row[1]),
                repository_identity=str(row[2]),
                license_spdx=str(row[3]),
                source_sha256=source_sha256,
                source=source,
            ):
                candidate_counts[split] += 1
                if exclusion.overlaps(docstring=candidate.docstring, target=candidate.target):
                    rejection_counts["humaneval_overlap"] += 1
                    continue
                try:
                    prompt_payload = binding.encode_payload(candidate.prompt)
                    target_payload = binding.encode_payload(candidate.target)
                except ValueError:
                    rejection_counts["reserved_protocol_token"] += 1
                    continue
                prompt_ids = [task_token, stream_begin, raw_mode, *prompt_payload, stream_end]
                target_ids = [stream_begin, raw_mode, *target_payload, stream_end, task_end]
                input_ids = prompt_ids + target_ids
                if len(input_ids) > max_sequence:
                    rejection_counts["sequence_too_long"] += 1
                    continue
                if len(target_ids) < 8:
                    rejection_counts["target_too_short"] += 1
                    continue
                example_id = _example_identity(candidate, input_ids)
                rows_by_split[split].append(
                    {
                        "example_id": example_id,
                        "candidate": candidate,
                        "input_ids": input_ids,
                        "loss_start_index": len(prompt_ids),
                    }
                )
    finally:
        catalog.close()

    for split in SPLITS:
        rows_by_split[split].sort(key=lambda item: str(item["example_id"]))
        if len(rows_by_split[split]) < quotas[split]:
            raise ValueError(
                f"instruction curriculum has only {len(rows_by_split[split])} "
                f"eligible {split} examples; quota is {quotas[split]}"
            )
        rows_by_split[split] = rows_by_split[split][: quotas[split]]

    policy = {
        "schema_version": INSTRUCTION_CURRICULUM_SCHEMA,
        "family": INSTRUCTION_FAMILY,
        "language": "python",
        "quotas": quotas,
        "max_sequence": max_sequence,
        "split_policy": "preserve-clean-v5-document-split-else-sha256-90-5-5",
        "prompt_protocol": "mixed_instruction + raw_code_stream -> raw_code_stream + task_end",
        "humaneval_exclusion_sha256": exclusion.sha256,
    }
    policy_sha256 = hashlib.sha256(
        _canonical_json(policy).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": TASK_DATABASE_SCHEMA,
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "vocabulary_size": str(binding.vocab_size),
        "maximum_sequence_tokens": str(max_sequence),
        "packing_complete": "true",
        "curriculum_schema": INSTRUCTION_CURRICULUM_SCHEMA,
        "curriculum_policy_sha256": policy_sha256,
        "source_catalog_sha256": catalog_sha256,
        "source_admission_sha256": _sha256_file(admission_path),
        "split_database_identity_sha256": split_audit.database_identity_sha256,
        "humaneval_exclusion_sha256": exclusion.sha256,
        "instruction_family": INSTRUCTION_FAMILY,
        "language": "python",
        "quotas": _canonical_json(quotas),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    license_counts: Counter[str] = Counter()
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
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
        connection.execute("CREATE INDEX tasks_cell ON tasks(split,language,family)")
        connection.execute(
            """CREATE TABLE instruction_provenance (
                   example_id TEXT PRIMARY KEY,
                   relative_path TEXT NOT NULL,
                   repository_identity TEXT NOT NULL,
                   license_spdx TEXT NOT NULL,
                   prompt_sha256 TEXT NOT NULL,
                   target_sha256 TEXT NOT NULL
               )"""
        )
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)", sorted(metadata.items())
        )
        for split in SPLITS:
            for item in rows_by_split[split]:
                raw_candidate = item["candidate"]
                raw_input_ids = item["input_ids"]
                if not isinstance(raw_candidate, PythonInstructionCandidate):
                    raise ValueError("instruction task candidate type is invalid")
                if type(raw_input_ids) is not list:
                    raise ValueError("instruction task input IDs are invalid")
                candidate = raw_candidate
                input_ids = cast(list[int], raw_input_ids)
                license_counts[candidate.license_spdx] += 1
                connection.execute(
                    "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        item["example_id"],
                        candidate.document_id,
                        "python",
                        INSTRUCTION_FAMILY,
                        candidate.record_index,
                        split,
                        zlib.compress(_encode_u16(input_ids), 6),
                        len(input_ids),
                        item["loss_start_index"],
                        CURRENT_RUNTIME_SPEC_SHA256["python"],
                        PARSER_RUNTIME_NOT_APPLICABLE,
                        candidate.source_sha256,
                    ),
                )
                connection.execute(
                    "INSERT INTO instruction_provenance VALUES(?,?,?,?,?,?)",
                    (
                        item["example_id"],
                        candidate.relative_path,
                        candidate.repository_identity,
                        candidate.license_spdx,
                        hashlib.sha256(candidate.prompt.encode("utf-8")).hexdigest(),
                        hashlib.sha256(candidate.target.encode("utf-8")).hexdigest(),
                    ),
                )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    temporary.replace(output_path)
    audit = audit_task_database(binding, output_path)
    expected_total = sum(quotas.values())
    if audit.example_count != expected_total or audit.split_counts != quotas:
        raise ValueError("written instruction curriculum failed its count audit")
    return InstructionCurriculumStatus(
        database_path=output_path,
        database_identity_sha256=audit.database_identity_sha256,
        example_count=audit.example_count,
        split_counts=audit.split_counts,
        candidate_counts=dict(sorted(candidate_counts.items())),
        rejection_counts=dict(sorted(rejection_counts.items())),
        license_counts=dict(sorted(license_counts.items())),
        policy_sha256=policy_sha256,
    )
