"""Resumable CodeX-2M final-code curriculum for the frozen native-65K model."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import zlib
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

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
from .tokenizer import NativeTokenizerBinding, _load_owned_tokenizers

CURRICULUM_SCHEMA = "bosscrafts-codex-final-code-curriculum-v1"
DATASET_NAME = "Modotte/CodeX-2M-Thinking"
DATASET_REVISION = "01e0bb5"
DATASET_LICENSE = "Apache-2.0"
INSTRUCTION_FAMILY = "instruction_to_code"
SPLIT_POLICY = (
    "sha256(canonical_input_and_code)_mod_100:train_0_95,validation_96_97,test_98_99"
)
SELECTION_POLICY = "final_supported_fenced_code_after_closing_think"
_FENCE = re.compile(r"```([^\r\n`]*)\r?\n(.*?)```", re.DOTALL)
_WORD = re.compile(r"[A-Za-z0-9_]+")
_HEX64 = re.compile(r"[0-9a-f]{64}")

_LANGUAGE_ALIASES = {
    "bash": "bash",
    "c": "c",
    "c#": "c_sharp",
    "c++": "cpp",
    "cpp": "cpp",
    "cs": "c_sharp",
    "csharp": "c_sharp",
    "go": "go",
    "golang": "go",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "kotlin": "kotlin",
    "php": "php",
    "py": "python",
    "python": "python",
    "python3": "python",
    "rb": "ruby",
    "ruby": "ruby",
    "rs": "rust",
    "rust": "rust",
    "sh": "bash",
    "shell": "bash",
    "swift": "swift",
    "ts": "typescript",
    "typescript": "typescript",
}


@dataclass(frozen=True)
class ExtractedCode:
    language: str
    code: str
    fence_label: str


@dataclass(frozen=True)
class ShardBuildStatus:
    database_path: str
    source_path: str
    source_sha256: str
    source_size_bytes: int
    source_rows: int
    admitted_rows: int
    split_counts: dict[str, int]
    language_counts: dict[str, int]
    rejection_counts: dict[str, int]
    sequence_tokens: int
    target_tokens: int


@dataclass(frozen=True)
class CurriculumBuildStatus:
    database_path: Path
    database_identity_sha256: str
    source_shard_count: int
    source_row_count: int
    example_count: int
    split_counts: dict[str, int]
    language_counts: dict[str, int]
    rejection_counts: dict[str, int]
    duplicate_count: int
    total_sequence_tokens: int
    total_target_tokens: int
    maximum_sequence_tokens: int
    source_manifest_sha256: str
    policy_sha256: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_words(value: str) -> str:
    return " ".join(word.lower() for word in _WORD.findall(value))


def normalize_language(label: str) -> str | None:
    if type(label) is not str:
        raise TypeError("fence language label must be a string")
    normalized = label.strip().lower().replace("language-", "")
    return _LANGUAGE_ALIASES.get(normalized)


def extract_final_code(output: str) -> ExtractedCode | None:
    """Return the last supported fenced program after the reasoning section."""
    if type(output) is not str or not output:
        return None
    tail = output.split("</think>", 1)[-1]
    candidates: list[ExtractedCode] = []
    for label, payload in _FENCE.findall(tail):
        language = normalize_language(label)
        code = payload.strip()
        if language is None and not label.strip():
            try:
                ast.parse(code)
            except (SyntaxError, ValueError, TypeError, MemoryError):
                pass
            else:
                language = "python"
        if language is None or not code:
            continue
        candidates.append(
            ExtractedCode(
                language=language,
                code=code.replace("\r\n", "\n").replace("\r", "\n") + "\n",
                fence_label=label.strip(),
            )
        )
    return candidates[-1] if candidates else None


class HumanEvalExclusion:
    """Fast exact/containment filter for the pinned 164-task HumanEval set."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.path = Path(dataset_path).resolve()
        if not self.path.is_file():
            raise ValueError("HumanEval exclusion dataset does not exist")
        self.sha256 = _sha256_file(self.path)
        prompts: list[str] = []
        solutions: set[str] = set()
        task_ids: list[str] = []
        with gzip.open(self.path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = row.get("task_id")
                prompt = row.get("prompt")
                solution = row.get("canonical_solution")
                if any(type(value) is not str for value in (task_id, prompt, solution)):
                    raise ValueError("HumanEval exclusion row is invalid")
                task_ids.append(str(task_id))
                prompts.append(_normalize_words(str(prompt)))
                solutions.add(_normalize_words(str(solution)))
        if task_ids != [f"HumanEval/{index}" for index in range(164)]:
            raise ValueError("HumanEval exclusion task order is invalid")
        self.prompts = tuple(value for value in prompts if len(value) >= 40)
        self.solutions = frozenset(value for value in solutions if value)

    def overlaps(self, prompt: str, code: str) -> bool:
        normalized_code = _normalize_words(code)
        if normalized_code in self.solutions:
            return True
        normalized_prompt = _normalize_words(prompt)
        if len(normalized_prompt) < 40:
            return False
        return any(
            benchmark in normalized_prompt or normalized_prompt in benchmark
            for benchmark in self.prompts
        )


def _split(content_sha256: str) -> str:
    if not _HEX64.fullmatch(content_sha256):
        raise ValueError("content SHA-256 is invalid")
    bucket = int(content_sha256[:8], 16) % 100
    if bucket < 96:
        return "train"
    if bucket < 98:
        return "validation"
    return "test"


def _example_identity(
    *, document_id: str, record_index: int, input_ids: list[int]
) -> str:
    input_ids_sha256 = _sha256_text(_canonical_json(input_ids))
    return _sha256_text(
        _canonical_json(
            {
                "document_id": document_id,
                "family": INSTRUCTION_FAMILY,
                "record_index": record_index,
                "input_ids_sha256": input_ids_sha256,
            }
        )
    )


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _initialize_shard_database(
    path: Path, metadata: dict[str, str]
) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    # The shard database is a private temporary artifact until it is closed and
    # atomically renamed. NORMAL avoids a physical disk flush for every bounded
    # transaction while still leaving an interrupted build unpublishable.
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
    )
    connection.execute(
        """CREATE TABLE tasks(
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
    connection.execute(
        """CREATE TABLE codex_provenance(
               example_id TEXT PRIMARY KEY,
               content_sha256 TEXT NOT NULL,
               source_parquet TEXT NOT NULL,
               source_parquet_sha256 TEXT NOT NULL,
               source_row_index INTEGER NOT NULL,
               input_sha256 TEXT NOT NULL,
               full_output_sha256 TEXT NOT NULL,
               final_code_sha256 TEXT NOT NULL,
               fence_label TEXT NOT NULL,
               license_spdx TEXT NOT NULL
           )"""
    )
    connection.execute(
        "CREATE TABLE rejection_counts(reason_code TEXT PRIMARY KEY,count INTEGER NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO metadata(key,value) VALUES(?,?)", sorted(metadata.items())
    )
    return connection


def _load_completed_shard(
    path: Path, expected: dict[str, str]
) -> ShardBuildStatus | None:
    if not path.is_file():
        return None
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"existing shard database lineage mismatch: {path}")
        if metadata.get("packing_complete") != "true":
            return None
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise ValueError(f"existing shard database integrity failed: {path}")
        split_counts = dict(
            connection.execute("SELECT split,COUNT(*) FROM tasks GROUP BY split")
        )
        language_counts = dict(
            connection.execute("SELECT language,COUNT(*) FROM tasks GROUP BY language")
        )
        rejection_counts = dict(
            connection.execute("SELECT reason_code,count FROM rejection_counts")
        )
        counts = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(token_count),0),"
            "COALESCE(SUM(token_count-loss_start_index),0) FROM tasks"
        ).fetchone()
        provenance_rows = connection.execute(
            """SELECT t.example_id,t.document_id,t.language,t.family,t.record_index,
                      t.split,t.semantic_spec_sha256,
                      t.parser_runtime_identity_sha256,t.source_sha256,
                      p.content_sha256,p.source_parquet,p.source_parquet_sha256,
                      p.source_row_index,p.input_sha256,p.full_output_sha256,
                      p.final_code_sha256,p.fence_label,p.license_spdx
               FROM tasks AS t JOIN codex_provenance AS p USING(example_id)
               ORDER BY t.example_id"""
        ).fetchall()
        task_count = int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
        provenance_count = int(
            connection.execute("SELECT COUNT(*) FROM codex_provenance").fetchone()[0]
        )
    finally:
        connection.close()
    if counts is None:
        raise ValueError("CodeXGLUE shard database aggregate query returned no row")
    if task_count != provenance_count or len(provenance_rows) != task_count:
        raise ValueError("existing shard provenance coverage mismatch")
    expected_source_name = Path(expected["source_path"]).name
    expected_source_sha256 = expected["source_sha256"]
    source_rows = int(expected["source_rows"])
    seen_content: set[str] = set()
    seen_source_rows: set[int] = set()
    for row in provenance_rows:
        (
            _example_id,
            document_id,
            language,
            family,
            record_index,
            split,
            semantic_spec_sha256,
            parser_runtime_identity_sha256,
            source_sha256,
            content_sha256,
            source_parquet,
            source_parquet_sha256,
            provenance_source_row,
            input_sha256,
            full_output_sha256,
            final_code_sha256,
            _fence_label,
            license_spdx,
        ) = row
        hashes = (
            str(document_id),
            str(semantic_spec_sha256),
            str(source_sha256),
            str(content_sha256),
            str(source_parquet_sha256),
            str(input_sha256),
            str(full_output_sha256),
            str(final_code_sha256),
        )
        record_index = int(record_index)
        provenance_source_row = int(provenance_source_row)
        content_sha256 = str(content_sha256)
        expected_content_sha256 = _sha256_text(
            _canonical_json(
                {
                    "input_sha256": str(input_sha256),
                    "final_code_sha256": str(final_code_sha256),
                }
            )
        )
        expected_document_id = _sha256_text(
            _canonical_json(
                {
                    "dataset": DATASET_NAME,
                    "revision": DATASET_REVISION,
                    "source_parquet_sha256": expected_source_sha256,
                    "source_row_index": record_index,
                    "content_sha256": content_sha256,
                }
            )
        )
        if (
            any(not _HEX64.fullmatch(value) for value in hashes)
            or str(source_parquet) != expected_source_name
            or str(source_parquet_sha256) != expected_source_sha256
            or str(license_spdx) != DATASET_LICENSE
            or str(family) != INSTRUCTION_FAMILY
            or str(split) not in {"train", "validation", "test"}
            or str(language) not in CURRENT_RUNTIME_SPEC_SHA256
            or str(semantic_spec_sha256)
            != CURRENT_RUNTIME_SPEC_SHA256[str(language)]
            or str(parser_runtime_identity_sha256)
            != PARSER_RUNTIME_NOT_APPLICABLE
            or record_index != provenance_source_row
            or not 0 <= record_index < source_rows
            or content_sha256 != expected_content_sha256
            or str(document_id) != expected_document_id
            or content_sha256 in seen_content
            or record_index in seen_source_rows
        ):
            raise ValueError("existing shard provenance is invalid")
        seen_content.add(content_sha256)
        seen_source_rows.add(record_index)
    if any(count < 0 for count in rejection_counts.values()) or (
        task_count + sum(int(count) for count in rejection_counts.values())
        != source_rows
    ):
        raise ValueError("existing shard provenance source-row coverage mismatch")
    return ShardBuildStatus(
        database_path=str(path),
        source_path=metadata["source_path"],
        source_sha256=metadata["source_sha256"],
        source_size_bytes=int(metadata["source_size_bytes"]),
        source_rows=source_rows,
        admitted_rows=int(counts[0]),
        split_counts={str(k): int(v) for k, v in split_counts.items()},
        language_counts={str(k): int(v) for k, v in language_counts.items()},
        rejection_counts={str(k): int(v) for k, v in rejection_counts.items()},
        sequence_tokens=int(counts[1]),
        target_tokens=int(counts[2]),
    )


def _build_shard_worker(
    source_path_text: str,
    destination_text: str,
    tokenizer_path_text: str,
    structural_ids: Mapping[str, int],
    binding_sha256: str,
    tokenizer_sha256: str,
    humaneval_path_text: str,
    max_sequence: int,
    batch_size: int,
) -> ShardBuildStatus:
    from pyarrow import parquet

    tokenizers = _load_owned_tokenizers()

    if tokenizers.__version__ != "0.23.2":
        raise RuntimeError("CodeX curriculum requires tokenizers 0.23.2")
    source_path = Path(source_path_text).resolve()
    destination = Path(destination_text).resolve()
    source_sha256 = _sha256_file(source_path)
    source_size = source_path.stat().st_size
    parquet_file = parquet.ParquetFile(source_path)
    source_rows = parquet_file.metadata.num_rows
    expected = {
        "schema_version": "bosscrafts-codex-shard-database-v1",
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "selection_policy": SELECTION_POLICY,
        "binding_sha256": binding_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "maximum_sequence_tokens": str(max_sequence),
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_size_bytes": str(source_size),
        "source_rows": str(source_rows),
        "humaneval_exclusion_sha256": _sha256_file(Path(humaneval_path_text)),
    }
    existing = _load_completed_shard(destination, expected)
    if existing is not None:
        return existing
    if destination.exists():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    metadata = {**expected, "packing_complete": "false"}
    connection = _initialize_shard_database(temporary, metadata)
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path_text)
    exclusion = HumanEvalExclusion(humaneval_path_text)
    task_token = structural_ids["<|bcnlt:task:mixed_instruction|>"]
    stream_begin = structural_ids["<|bcnlt:stream:begin|>"]
    stream_end = structural_ids["<|bcnlt:stream:end|>"]
    raw_mode = structural_ids["<|bcnlt:mode:raw_code|>"]
    task_end = structural_ids["<|bcnlt:task:end|>"]
    rejection_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    seen_content: set[str] = set()
    admitted = 0
    pending_since_commit = 0
    sequence_tokens = 0
    target_tokens = 0
    row_offset = 0
    try:
        for row_group in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(row_group, columns=["input", "output"])
            prompts = table["input"].to_pylist()
            outputs = table["output"].to_pylist()
            for start in range(0, len(prompts), batch_size):
                raw_batch: list[tuple[int, str, str, ExtractedCode, str]] = []
                for local_index, (prompt, output) in enumerate(
                    zip(
                        prompts[start : start + batch_size],
                        outputs[start : start + batch_size],
                        strict=True,
                    )
                ):
                    source_row_index = row_offset + start + local_index
                    if (
                        type(prompt) is not str
                        or type(output) is not str
                        or not prompt
                        or not output
                    ):
                        rejection_counts["invalid_text_fields"] += 1
                        continue
                    extracted = extract_final_code(output)
                    if extracted is None:
                        rejection_counts["no_supported_final_code_fence"] += 1
                        continue
                    if extracted.language == "python":
                        try:
                            ast.parse(extracted.code)
                        except (SyntaxError, ValueError, TypeError, MemoryError):
                            rejection_counts["python_syntax_invalid"] += 1
                            continue
                    if exclusion.overlaps(prompt, extracted.code):
                        rejection_counts["humaneval_overlap"] += 1
                        continue
                    input_sha256 = _sha256_text(prompt)
                    code_sha256 = _sha256_text(extracted.code)
                    content_sha256 = _sha256_text(
                        _canonical_json(
                            {
                                "input_sha256": input_sha256,
                                "final_code_sha256": code_sha256,
                            }
                        )
                    )
                    if content_sha256 in seen_content:
                        rejection_counts["duplicate_within_shard"] += 1
                        continue
                    seen_content.add(content_sha256)
                    raw_batch.append(
                        (source_row_index, prompt, output, extracted, content_sha256)
                    )
                if not raw_batch:
                    continue
                prompt_encodings = tokenizer.encode_batch(
                    [item[1] for item in raw_batch]
                )
                code_encodings = tokenizer.encode_batch(
                    [item[3].code for item in raw_batch]
                )
                admitted_before_batch = admitted
                for item, prompt_encoding, code_encoding in zip(
                    raw_batch, prompt_encodings, code_encodings, strict=True
                ):
                    source_row_index, prompt, output, extracted, content_sha256 = item
                    if any(
                        token_id <= 76
                        for token_id in (*prompt_encoding.ids, *code_encoding.ids)
                    ):
                        rejection_counts["reserved_protocol_token"] += 1
                        continue
                    prompt_ids = [
                        task_token,
                        stream_begin,
                        raw_mode,
                        *prompt_encoding.ids,
                        stream_end,
                    ]
                    target_ids = [
                        stream_begin,
                        raw_mode,
                        *code_encoding.ids,
                        stream_end,
                        task_end,
                    ]
                    input_ids = prompt_ids + target_ids
                    if len(input_ids) > max_sequence:
                        rejection_counts["sequence_too_long"] += 1
                        continue
                    if len(target_ids) < 8:
                        rejection_counts["target_too_short"] += 1
                        continue
                    document_id = _sha256_text(
                        _canonical_json(
                            {
                                "dataset": DATASET_NAME,
                                "revision": DATASET_REVISION,
                                "source_parquet_sha256": source_sha256,
                                "source_row_index": source_row_index,
                                "content_sha256": content_sha256,
                            }
                        )
                    )
                    example_id = _example_identity(
                        document_id=document_id,
                        record_index=source_row_index,
                        input_ids=input_ids,
                    )
                    split = _split(content_sha256)
                    source_content_sha256 = _sha256_text(prompt + "\0" + extracted.code)
                    connection.execute(
                        "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            example_id,
                            document_id,
                            extracted.language,
                            INSTRUCTION_FAMILY,
                            source_row_index,
                            split,
                            zlib.compress(_encode_u16(input_ids), 6),
                            len(input_ids),
                            len(prompt_ids),
                            CURRENT_RUNTIME_SPEC_SHA256[extracted.language],
                            PARSER_RUNTIME_NOT_APPLICABLE,
                            source_content_sha256,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO codex_provenance VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            example_id,
                            content_sha256,
                            source_path.name,
                            source_sha256,
                            source_row_index,
                            _sha256_text(prompt),
                            _sha256_text(output),
                            _sha256_text(extracted.code),
                            extracted.fence_label,
                            DATASET_LICENSE,
                        ),
                    )
                    admitted += 1
                    sequence_tokens += len(input_ids)
                    target_tokens += len(target_ids)
                    split_counts[split] += 1
                    language_counts[extracted.language] += 1
                pending_since_commit += admitted - admitted_before_batch
                if pending_since_commit >= 512:
                    connection.commit()
                    pending_since_commit = 0
            row_offset += len(prompts)
        connection.executemany(
            "INSERT INTO rejection_counts VALUES(?,?)", sorted(rejection_counts.items())
        )
        connection.execute(
            "UPDATE metadata SET value='true' WHERE key='packing_complete'"
        )
        connection.commit()
        connection.execute("VACUUM")
    except BaseException:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()
    temporary.replace(destination)
    return ShardBuildStatus(
        database_path=str(destination),
        source_path=str(source_path),
        source_sha256=source_sha256,
        source_size_bytes=source_size,
        source_rows=source_rows,
        admitted_rows=admitted,
        split_counts=dict(sorted(split_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        rejection_counts=dict(sorted(rejection_counts.items())),
        sequence_tokens=sequence_tokens,
        target_tokens=target_tokens,
    )


def _merge_shards(
    binding: NativeTokenizerBinding,
    statuses: list[ShardBuildStatus],
    destination: Path,
    humaneval_sha256: str,
    max_sequence: int,
) -> CurriculumBuildStatus:
    if destination.exists():
        raise ValueError("CodeX curriculum output database already exists")
    source_manifest = [
        {
            "name": Path(status.source_path).name,
            "rows": status.source_rows,
            "sha256": status.source_sha256,
            "size_bytes": status.source_size_bytes,
        }
        for status in sorted(statuses, key=lambda item: Path(item.source_path).name)
    ]
    source_manifest_sha256 = _sha256_text(_canonical_json(source_manifest))
    policy = {
        "schema_version": CURRICULUM_SCHEMA,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "selection_policy": SELECTION_POLICY,
        "split_policy": SPLIT_POLICY,
        "maximum_sequence_tokens": max_sequence,
        "humaneval_exclusion_sha256": humaneval_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "prompt_protocol": "mixed_instruction + raw_code_stream -> raw_code_stream + task_end",
        "deduplication": "sha256(canonical input hash and final code hash)",
    }
    policy_sha256 = _sha256_text(_canonical_json(policy))
    metadata = {
        "schema_version": TASK_DATABASE_SCHEMA,
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "vocabulary_size": str(binding.vocab_size),
        "maximum_sequence_tokens": str(max_sequence),
        "packing_complete": "true",
        "curriculum_schema_version": CURRICULUM_SCHEMA,
        "curriculum_policy_sha256": policy_sha256,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "selection_policy": SELECTION_POLICY,
        "split_policy": SPLIT_POLICY,
        "source_manifest_sha256": source_manifest_sha256,
        "humaneval_exclusion_sha256": humaneval_sha256,
        "instruction_family": INSTRUCTION_FAMILY,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    output = sqlite3.connect(temporary)
    output.execute("PRAGMA journal_mode=DELETE")
    output.execute("PRAGMA synchronous=FULL")
    output.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    output.execute(
        """CREATE TABLE tasks(
               example_id TEXT PRIMARY KEY,document_id TEXT NOT NULL,
               language TEXT NOT NULL,family TEXT NOT NULL,
               record_index INTEGER NOT NULL,split TEXT NOT NULL,
               input_ids_zlib BLOB NOT NULL,token_count INTEGER NOT NULL,
               loss_start_index INTEGER NOT NULL,
               semantic_spec_sha256 TEXT NOT NULL,
               parser_runtime_identity_sha256 TEXT NOT NULL,
               source_sha256 TEXT NOT NULL
           )"""
    )
    output.execute("CREATE INDEX tasks_cell ON tasks(split,language,family)")
    output.execute(
        """CREATE TABLE codex_provenance(
               example_id TEXT PRIMARY KEY,content_sha256 TEXT NOT NULL UNIQUE,
               source_parquet TEXT NOT NULL,source_parquet_sha256 TEXT NOT NULL,
               source_row_index INTEGER NOT NULL,input_sha256 TEXT NOT NULL,
               full_output_sha256 TEXT NOT NULL,final_code_sha256 TEXT NOT NULL,
               fence_label TEXT NOT NULL,license_spdx TEXT NOT NULL
           )"""
    )
    output.execute(
        """CREATE TABLE source_shards(
               name TEXT PRIMARY KEY,sha256 TEXT NOT NULL,size_bytes INTEGER NOT NULL,
               source_rows INTEGER NOT NULL,admitted_rows INTEGER NOT NULL
           )"""
    )
    output.execute(
        "CREATE TABLE rejection_counts(reason_code TEXT PRIMARY KEY,count INTEGER NOT NULL)"
    )
    output.executemany("INSERT INTO metadata VALUES(?,?)", sorted(metadata.items()))
    seen: set[str] = set()
    duplicate_count = 0
    rejection_counts: Counter[str] = Counter()
    source_row_count = 0
    try:
        for status in sorted(statuses, key=lambda item: Path(item.source_path).name):
            source_row_count += status.source_rows
            rejection_counts.update(status.rejection_counts)
            source = sqlite3.connect(
                f"{Path(status.database_path).as_uri()}?mode=ro", uri=True, timeout=30
            )
            source.execute("PRAGMA query_only=ON")
            try:
                rows = source.execute(
                    """SELECT t.example_id,t.document_id,t.language,t.family,t.record_index,
                              t.split,t.input_ids_zlib,t.token_count,t.loss_start_index,
                               t.semantic_spec_sha256,t.parser_runtime_identity_sha256,
                               t.source_sha256,
                              p.content_sha256,p.source_parquet,p.source_parquet_sha256,
                              p.source_row_index,p.input_sha256,p.full_output_sha256,
                              p.final_code_sha256,p.fence_label,p.license_spdx
                       FROM tasks t JOIN codex_provenance p USING(example_id)
                       ORDER BY t.example_id"""
                )
                admitted_for_source = 0
                for row in rows:
                    content_sha256 = str(row[12])
                    if content_sha256 in seen:
                        duplicate_count += 1
                        continue
                    seen.add(content_sha256)
                    output.execute(
                        "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", row[:12]
                    )
                    output.execute(
                        "INSERT INTO codex_provenance VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (row[0], *row[12:]),
                    )
                    admitted_for_source += 1
                output.execute(
                    "INSERT INTO source_shards VALUES(?,?,?,?,?)",
                    (
                        Path(status.source_path).name,
                        status.source_sha256,
                        status.source_size_bytes,
                        status.source_rows,
                        admitted_for_source,
                    ),
                )
                output.commit()
            finally:
                source.close()
        if duplicate_count:
            rejection_counts["duplicate_across_shards"] += duplicate_count
        output.executemany(
            "INSERT INTO rejection_counts VALUES(?,?)", sorted(rejection_counts.items())
        )
        output.commit()
        output.execute("VACUUM")
    except BaseException:
        output.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        output.close()
    try:
        audit = audit_task_database(binding, temporary)
        connection = sqlite3.connect(
            f"{temporary.as_uri()}?mode=ro", uri=True, timeout=30
        )
        connection.execute("PRAGMA query_only=ON")
        try:
            target_tokens = int(
                connection.execute(
                    "SELECT COALESCE(SUM(token_count-loss_start_index),0) FROM tasks"
                ).fetchone()[0]
            )
        finally:
            connection.close()
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if destination.exists():
            raise ValueError("CodeX curriculum output database already exists")
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError(
                "CodeX curriculum output database already exists"
            ) from error
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.unlink()
    return CurriculumBuildStatus(
        database_path=destination,
        database_identity_sha256=audit.database_identity_sha256,
        source_shard_count=len(statuses),
        source_row_count=source_row_count,
        example_count=audit.example_count,
        split_counts=audit.split_counts,
        language_counts=audit.language_counts,
        rejection_counts=dict(sorted(rejection_counts.items())),
        duplicate_count=duplicate_count,
        total_sequence_tokens=audit.total_sequence_tokens,
        total_target_tokens=target_tokens,
        maximum_sequence_tokens=audit.maximum_sequence_tokens,
        source_manifest_sha256=source_manifest_sha256,
        policy_sha256=policy_sha256,
    )


def build_codex_code_curriculum(
    binding: NativeTokenizerBinding,
    *,
    dataset_root: str | Path,
    shard_indices: tuple[int, ...],
    output_root: str | Path,
    humaneval_dataset: str | Path,
    max_sequence: int = 2048,
    workers: int = 1,
    batch_size: int = 32,
) -> CurriculumBuildStatus:
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if (
        type(shard_indices) is not tuple
        or not shard_indices
        or any(
            type(index) is not int or not 0 <= index <= 250 for index in shard_indices
        )
        or len(set(shard_indices)) != len(shard_indices)
    ):
        raise ValueError("shard_indices must be a unique nonempty tuple in [0,250]")
    if type(max_sequence) is not int or max_sequence < 256:
        raise ValueError("max_sequence must be at least 256")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be between 1 and 128")
    root = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    humaneval = Path(humaneval_dataset).resolve()
    if not root.is_dir() or not humaneval.is_file():
        raise ValueError("dataset root and HumanEval exclusion file must exist")
    source_paths = [
        root / f"shard_{index:05d}.parquet" for index in sorted(shard_indices)
    ]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise ValueError(f"CodeX source shards are missing: {missing}")
    final_database = output / "tasks.sqlite"
    if final_database.exists():
        raise ValueError("CodeX curriculum output database already exists")
    shard_root = output / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    tokenizer_path = binding.bundle_path / "tokenizer.json"
    arguments = [
        (
            str(source_path),
            str(shard_root / f"{source_path.stem}.sqlite"),
            str(tokenizer_path),
            binding.structural_token_ids,
            binding.binding_sha256,
            binding.tokenizer_sha256,
            str(humaneval),
            max_sequence,
            batch_size,
        )
        for source_path in source_paths
    ]
    statuses: list[ShardBuildStatus] = []
    _write_progress(
        progress_path,
        {
            "schema_version": "bosscrafts-codex-build-progress-v1",
            "dataset_name": DATASET_NAME,
            "source_shards": len(arguments),
            "completed_shards": 0,
            "failed_shards": 0,
            "status": "building_shards",
        },
    )
    if workers == 1:
        for argument in arguments:
            statuses.append(_build_shard_worker(*argument))
            _write_progress(
                progress_path,
                {
                    "schema_version": "bosscrafts-codex-build-progress-v1",
                    "dataset_name": DATASET_NAME,
                    "source_shards": len(arguments),
                    "completed_shards": len(statuses),
                    "failed_shards": 0,
                    "admitted_rows": sum(item.admitted_rows for item in statuses),
                    "status": "building_shards",
                },
            )
    else:
        failures: list[str] = []
        with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as executor:
            futures = {
                executor.submit(_build_shard_worker, *argument): argument[0]
                for argument in arguments
            }
            for future in as_completed(futures):
                try:
                    statuses.append(future.result())
                except BaseException as error:
                    failures.append(
                        f"{futures[future]}: {type(error).__name__}: {error}"
                    )
                _write_progress(
                    progress_path,
                    {
                        "schema_version": "bosscrafts-codex-build-progress-v1",
                        "dataset_name": DATASET_NAME,
                        "source_shards": len(arguments),
                        "completed_shards": len(statuses),
                        "failed_shards": len(failures),
                        "admitted_rows": sum(item.admitted_rows for item in statuses),
                        "failures": failures,
                        "status": "failed" if failures else "building_shards",
                    },
                )
        if failures:
            raise RuntimeError("CodeX shard build failed: " + "; ".join(failures))
    result = _merge_shards(
        binding,
        statuses,
        final_database,
        _sha256_file(humaneval),
        max_sequence,
    )
    report = asdict(result)
    report["database_path"] = str(result.database_path)
    _write_progress(output / "build-report.json", report)
    _write_progress(
        progress_path,
        {
            "schema_version": "bosscrafts-codex-build-progress-v1",
            "dataset_name": DATASET_NAME,
            "source_shards": len(arguments),
            "completed_shards": len(statuses),
            "failed_shards": 0,
            "admitted_rows": result.example_count,
            "database_identity_sha256": result.database_identity_sha256,
            "status": "complete",
        },
    )
    return result
