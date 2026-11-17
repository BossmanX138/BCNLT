"""Freeze pinned upstream benchmarks into immutable acceptance-case catalogs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from .category_case_inventory import (
    CATEGORY_CASE_SCHEMA,
    _valid_provenance,
    _validate_case_payload,
)
from .immutable_artifacts import publish_new_files
from .model_acceptance import MAXIMUM_CATEGORY_EXAMPLES, MINIMUM_CATEGORY_EXAMPLES
from .scientific_sandbox import SandboxExecutionResult

MAX_UPSTREAM_CASE_SOURCE_BYTES = 128 * 1024 * 1024
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MCEVAL_SOURCE_MANIFEST_SCHEMA = "bosscrafts-mceval-generation-sources-v1"
MCEVAL_EXECUTION_AUDIT_SCHEMA = "bosscrafts-mceval-execution-freeze-audit-v1"
AEGIS_V1_SAFETY_AUDIT_SCHEMA = "bosscrafts-aegis-v1-safety-freeze-audit-v1"
MCEVAL_LANGUAGE_FILES = {
    "bash": "Shell.jsonl",
    "c": "C.jsonl",
    "c_sharp": "C#.jsonl",
    "cpp": "CPP.jsonl",
    "go": "Go.jsonl",
    "java": "Java.jsonl",
    "javascript": "JavaScript.jsonl",
    "kotlin": "Kotlin.jsonl",
    "php": "PHP.jsonl",
    "python": "Python.jsonl",
    "ruby": "Ruby.jsonl",
    "rust": "Rust.jsonl",
    "swift": "Swift.jsonl",
    "typescript": "TypeScript.jsonl",
}


@dataclass(frozen=True)
class FrozenCategoryCases:
    output: Path
    record_count: int
    source_sha256: str
    revision: str


@dataclass(frozen=True)
class FrozenMcEvalExecutionCases:
    output: Path
    audit_output: Path
    record_count: int
    rejected_count: int
    source_manifest_sha256: str
    revision: str


@dataclass(frozen=True)
class FrozenAegisV1SafetyCases:
    output: Path
    audit_output: Path
    record_count: int
    rejected_count: int
    source_sha256: str
    revision: str


class McEvalSandboxExecutor(Protocol):
    def execute(
        self,
        *,
        language: str,
        source: str,
        stdin_text: str,
        mode: str = "program",
    ) -> SandboxExecutionResult: ...


@dataclass(frozen=True)
class _McEvalPendingTask:
    language: str
    task_id: str
    prompt: str
    harness_prefix: str
    harness_suffix: str
    canonical_solution: str
    provenance: dict[str, object]


@dataclass(frozen=True)
class _McEvalTaskOutcome:
    rejection_reason: str | None
    canonical: SandboxExecutionResult | None
    negative: SandboxExecutionResult | None


def _execute_mceval_task(
    task: _McEvalPendingTask,
    *,
    sandbox: McEvalSandboxExecutor,
) -> _McEvalTaskOutcome:
    canonical_source = (
        task.harness_prefix + task.canonical_solution + task.harness_suffix
    )
    negative_source = task.harness_prefix + task.harness_suffix
    if max(
        len(canonical_source.encode("utf-8")),
        len(negative_source.encode("utf-8")),
    ) > 64 * 1024:
        return _McEvalTaskOutcome("execution_source_too_large", None, None)
    canonical = sandbox.execute(
        language=task.language,
        source=canonical_source,
        stdin_text="",
        mode="test",
    )
    if not _sandbox_passed_exact(canonical, expected_stdout=None):
        return _McEvalTaskOutcome(
            "canonical_solution_failed",
            canonical,
            None,
        )
    negative = sandbox.execute(
        language=task.language,
        source=negative_source,
        stdin_text="",
        mode="test",
    )
    if _sandbox_passed_exact(
        negative,
        expected_stdout=canonical.stdout_bytes,
    ):
        return _McEvalTaskOutcome(
            "negative_control_passed",
            canonical,
            negative,
        )
    return _McEvalTaskOutcome(None, canonical, negative)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate source key: {key}")
        value[key] = item
    return value


def _read_verified_source(
    source: str | Path,
    *,
    revision: str,
    expected_source_sha256: str,
) -> tuple[bytes, str]:
    if _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError("source revision must be an exact 40-character Git SHA")
    if _SHA256.fullmatch(expected_source_sha256) is None:
        raise ValueError("expected source SHA-256 is invalid")
    path = Path(source).resolve()
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError("upstream case source is unreadable") from error
    if not content or len(content) > MAX_UPSTREAM_CASE_SOURCE_BYTES:
        raise ValueError("upstream case source size is invalid")
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if observed_sha256 != expected_source_sha256:
        raise ValueError("upstream case source SHA-256 mismatch")
    return content, observed_sha256


def _build_record(
    *,
    category: str,
    case_kind: str,
    case: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, object]:
    if not _valid_provenance(provenance):
        raise ValueError("category case provenance is invalid")
    _validate_case_payload(category, case_kind, case)
    body: dict[str, object] = {
        "schema_version": CATEGORY_CASE_SCHEMA,
        "category": category,
        "case_kind": case_kind,
        "case": case,
        "provenance": provenance,
    }
    return {
        **body,
        "example_id": hashlib.sha256(_canonical(body)).hexdigest(),
    }


def _publish_records(
    output: str | Path,
    records: list[dict[str, object]],
    *,
    category: str,
    source_sha256: str,
    revision: str,
) -> FrozenCategoryCases:
    minimum = MINIMUM_CATEGORY_EXAMPLES[category]
    if not minimum <= len(records) <= MAXIMUM_CATEGORY_EXAMPLES:
        raise ValueError(f"{category} frozen source count is invalid")
    records.sort(key=lambda record: str(record["example_id"]))
    rendered = b"".join(_canonical(record) + b"\n" for record in records)
    destination = Path(output).resolve()
    publish_new_files(
        ((destination, rendered),),
        artifact_label=f"{category} frozen source cases",
    )
    return FrozenCategoryCases(
        output=destination,
        record_count=len(records),
        source_sha256=source_sha256,
        revision=revision,
    )


def _read_mceval_manifest(path: str | Path) -> tuple[dict[str, object], str]:
    resolved = Path(path).resolve()
    try:
        content = resolved.read_bytes()
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("McEval source manifest is unreadable") from error
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "source",
        "revision",
        "license",
        "attribution",
        "languages",
    }:
        raise ValueError("McEval source manifest fields are invalid")
    languages = payload.get("languages")
    if (
        payload.get("schema_version") != MCEVAL_SOURCE_MANIFEST_SCHEMA
        or payload.get("source")
        != "hf://datasets/Multilingual-Multimodal-NLP/McEval"
        or type(payload.get("revision")) is not str
        or _GIT_REVISION.fullmatch(str(payload["revision"])) is None
        or payload.get("license") != "CC-BY-SA-4.0"
        or payload.get("attribution") != "McEval; Chai et al., 2024"
        or type(languages) is not dict
        or set(languages) != set(MCEVAL_LANGUAGE_FILES)
    ):
        raise ValueError("McEval source manifest identity is invalid")
    for language, filename in MCEVAL_LANGUAGE_FILES.items():
        item = languages.get(language)
        if (
            type(item) is not dict
            or set(item) != {"file", "sha256"}
            or item.get("file") != filename
            or type(item.get("sha256")) is not str
            or _SHA256.fullmatch(str(item["sha256"])) is None
        ):
            raise ValueError("McEval source manifest language entry is invalid")
    return payload, hashlib.sha256(content).hexdigest()


def _sandbox_passed_exact(
    result: SandboxExecutionResult,
    *,
    expected_stdout: bytes | None,
) -> bool:
    return (
        result.status == "passed"
        and result.phase == "run"
        and result.exit_code == 0
        and not result.timed_out
        and not result.output_limited
        and result.stderr_bytes == b""
        and (
            expected_stdout is None
            or result.stdout_bytes == expected_stdout
        )
    )


def _mceval_harness(
    language: str,
    prompt: str,
    tests: str,
) -> tuple[str, str]:
    prefix = prompt + "\n"
    adapted_tests = tests
    if language == "c_sharp":
        prefix = "using System.Linq;\n" + prefix
        main_pattern = re.compile(
            r"static\s+void\s+Main\s*\([^)]*\)\s*\{"
        )
        listener = (
            "\n        System.Diagnostics.Debug.Listeners.Add("
            "new System.Diagnostics.TextWriterTraceListener(System.Console.Error));"
            "\n        System.Diagnostics.Debug.AutoFlush = true;"
        )
        adapted_tests, replacement_count = main_pattern.subn(
            lambda match: match.group(0) + listener,
            adapted_tests,
            count=1,
        )
        if replacement_count != 1:
            raise ValueError("McEval C# test harness has no unique Main entry point")
    elif language == "kotlin":
        adapted_tests = re.sub(
            r"\n\s*main\(\)\s*$",
            "",
            adapted_tests,
            count=1,
        )
        if re.search(r"\bfun\s+main\s*\(", adapted_tests) is None:
            indented = "\n".join(
                "    " + line if line else ""
                for line in adapted_tests.splitlines()
            )
            adapted_tests = "fun main() {\n" + indented + "\n}"
    return prefix, "\n" + adapted_tests


def freeze_mceval_execution_cases(
    source_root: str | Path,
    source_manifest: str | Path,
    output: str | Path,
    audit_output: str | Path,
    *,
    sandbox: McEvalSandboxExecutor,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> FrozenMcEvalExecutionCases:
    """Freeze only McEval cases proven by canonical and negative-control runs."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("McEval worker count must be an integer from 1 through 8")
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise ValueError("McEval source root is not a directory")
    manifest, source_manifest_sha256 = _read_mceval_manifest(source_manifest)
    revision = str(manifest["revision"])
    language_manifest = manifest["languages"]
    if type(language_manifest) is not dict:
        raise AssertionError("validated McEval manifest languages are invalid")
    records: list[dict[str, object]] = []
    accepted_evidence: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    source_files: dict[str, object] = {}
    accepted_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    seen_task_ids: set[str] = set()
    sandbox_manifest_ids: set[str] = set()
    sandbox_image_ids: set[str] = set()
    pending_tasks: list[_McEvalPendingTask] = []
    expected_fields = {
        "task_id",
        "prompt",
        "canonical_solution",
        "test",
        "entry_point",
        "signature",
        "docstring",
        "instruction",
        "level",
    }
    for language, filename in MCEVAL_LANGUAGE_FILES.items():
        entry = language_manifest[language]
        if type(entry) is not dict:
            raise AssertionError("validated McEval language entry is invalid")
        content, source_sha256 = _read_verified_source(
            root / filename,
            revision=revision,
            expected_source_sha256=str(entry["sha256"]),
        )
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError(f"McEval {language} source is not UTF-8") from error
        if not lines or any(not line for line in lines):
            raise ValueError(f"McEval {language} source must be non-empty JSONL")
        source_files[language] = {
            "file": filename,
            "sha256": source_sha256,
            "row_count": len(lines),
        }
        provenance: dict[str, object] = {
            "source": (
                "hf://datasets/Multilingual-Multimodal-NLP/"
                f"McEval/generation/{filename}"
            ),
            "revision": revision,
            "source_sha256": source_sha256,
            "license": "CC-BY-SA-4.0",
            "attribution": "McEval; Chai et al., 2024",
        }
        for line_number, line in enumerate(lines, start=1):
            try:
                row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"McEval {language} line {line_number} is invalid JSON"
                ) from error
            if type(row) is not dict or set(row) != expected_fields:
                raise ValueError(
                    f"McEval {language} line {line_number} fields are invalid"
                )
            task_id = row.get("task_id")
            prompt = row.get("prompt")
            canonical_solution = row.get("canonical_solution")
            tests = row.get("test")
            if (
                type(task_id) is not str
                or not task_id.strip()
                or task_id in seen_task_ids
                or any(
                    type(value) is not str or not value.strip()
                    for value in (prompt, canonical_solution, tests)
                )
            ):
                raise ValueError(
                    f"McEval {language} line {line_number} task is invalid"
                )
            seen_task_ids.add(task_id)
            prefix = str(prompt)
            harness_prefix, suffix = _mceval_harness(
                language,
                prefix,
                str(tests),
            )
            pending_tasks.append(
                _McEvalPendingTask(
                    language=language,
                    task_id=task_id,
                    prompt=prefix,
                    harness_prefix=harness_prefix,
                    harness_suffix=suffix,
                    canonical_solution=str(canonical_solution),
                    provenance=provenance,
                )
            )
    execute_task = partial(_execute_mceval_task, sandbox=sandbox)
    total_tasks = len(pending_tasks)
    if workers == 1:
        outcomes = []
        for completed, task in enumerate(pending_tasks, start=1):
            outcomes.append(execute_task(task))
            if progress is not None:
                progress(completed, total_tasks)
    else:
        outcome_slots: list[_McEvalTaskOutcome | None] = [None] * total_tasks
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mceval-sandbox",
        ) as executor:
            future_indices = {
                executor.submit(execute_task, task): index
                for index, task in enumerate(pending_tasks)
            }
            for completed, future in enumerate(
                as_completed(future_indices),
                start=1,
            ):
                outcome_slots[future_indices[future]] = future.result()
                if progress is not None:
                    progress(completed, total_tasks)
        outcomes = []
        for outcome in outcome_slots:
            if outcome is None:
                raise AssertionError("McEval worker result is missing")
            outcomes.append(outcome)
    for task, outcome in zip(pending_tasks, outcomes, strict=True):
        canonical = outcome.canonical
        negative = outcome.negative
        if outcome.rejection_reason is not None:
            rejected_counts[task.language] += 1
            rejection: dict[str, object] = {
                "language": task.language,
                "source_task_id": task.task_id,
                "reason": outcome.rejection_reason,
            }
            if canonical is not None:
                rejection["canonical_sandbox"] = canonical.to_dict()
            if negative is not None:
                rejection["broken_sandbox"] = negative.to_dict()
            rejections.append(rejection)
            continue
        if canonical is None or negative is None:
            raise AssertionError("accepted McEval task is missing execution evidence")
        try:
            expected_stdout = canonical.stdout_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("McEval canonical stdout is not UTF-8") from error
        case: dict[str, object] = {
            "source_task_id": task.task_id,
            "split": "test",
            "language": task.language,
            "inference_task": "instruction_to_code",
            "prompt": task.prompt,
            "execution_mode": "test_harness",
            "harness_prefix": task.harness_prefix,
            "harness_suffix": task.harness_suffix,
            "stdin": "",
            "expected_stdout": expected_stdout,
        }
        record = _build_record(
            category="execution_correctness",
            case_kind="benchmark_function_completion",
            case=case,
            provenance=task.provenance,
        )
        records.append(record)
        accepted_counts[task.language] += 1
        sandbox_manifest_ids.add(
            canonical.attestation.image_manifest_sha256
        )
        sandbox_image_ids.add(canonical.attestation.image_id)
        accepted_evidence.append(
            {
                "example_id": record["example_id"],
                "language": task.language,
                "source_task_id": task.task_id,
                "expected_stdout_sha256": hashlib.sha256(
                    canonical.stdout_bytes
                ).hexdigest(),
                "canonical_request_sha256": canonical.request_sha256,
                "canonical_response_sha256": canonical.response_sha256,
                "negative_request_sha256": negative.request_sha256,
                "negative_response_sha256": negative.response_sha256,
                "negative_status": negative.status,
            }
        )
    minimum = MINIMUM_CATEGORY_EXAMPLES["execution_correctness"]
    if not minimum <= len(records) <= MAXIMUM_CATEGORY_EXAMPLES:
        raise ValueError("execution_correctness frozen source count is invalid")
    if set(accepted_counts) != set(MCEVAL_LANGUAGE_FILES):
        raise ValueError("McEval accepted cases do not cover every language")
    records.sort(key=lambda record: str(record["example_id"]))
    accepted_evidence.sort(key=lambda item: str(item["example_id"]))
    rejections.sort(
        key=lambda item: (str(item["language"]), str(item["source_task_id"]))
    )
    rendered = b"".join(_canonical(record) + b"\n" for record in records)
    unsigned_audit: dict[str, object] = {
        "schema_version": MCEVAL_EXECUTION_AUDIT_SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "revision": revision,
        "worker_count": workers,
        "harness_adapters": {
            "c_sharp": (
                "prepend System.Linq import; compile with DEBUG assertions enabled; "
                "route Debug.Assert failures to stderr"
            ),
            "kotlin": (
                "remove redundant trailing main call or wrap top-level tests in main"
            ),
        },
        "source_files": source_files,
        "record_count": len(records),
        "rejected_count": len(rejections),
        "accepted_counts": dict(sorted(accepted_counts.items())),
        "rejected_counts": {
            language: rejected_counts[language]
            for language in sorted(MCEVAL_LANGUAGE_FILES)
        },
        "case_file_sha256": hashlib.sha256(rendered).hexdigest(),
        "sandbox_image_manifest_sha256": sorted(sandbox_manifest_ids),
        "sandbox_image_ids": sorted(sandbox_image_ids),
        "accepted_evidence": accepted_evidence,
        "rejections": rejections,
    }
    audit = {**unsigned_audit, "audit_sha256": hashlib.sha256(_canonical(unsigned_audit)).hexdigest()}
    audit_rendered = (
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination = Path(output).resolve()
    audit_destination = Path(audit_output).resolve()
    if destination == audit_destination:
        raise ValueError("McEval cases and audit outputs must be distinct")
    publish_new_files(
        ((destination, rendered), (audit_destination, audit_rendered)),
        artifact_label="McEval execution cases and audit",
    )
    return FrozenMcEvalExecutionCases(
        output=destination,
        audit_output=audit_destination,
        record_count=len(records),
        rejected_count=len(rejections),
        source_manifest_sha256=source_manifest_sha256,
        revision=revision,
    )


def freeze_ifeval_cases(
    source: str | Path,
    output: str | Path,
    *,
    revision: str,
    expected_source_sha256: str,
) -> FrozenCategoryCases:
    """Freeze the exact Google IFEval JSONL source as executable constraints."""
    content, source_sha256 = _read_verified_source(
        source,
        revision=revision,
        expected_source_sha256=expected_source_sha256,
    )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("IFEval source is not UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("IFEval source must be non-empty JSONL")
    provenance: dict[str, object] = {
        "source": "hf://datasets/google/IFEval/ifeval_input_data.jsonl",
        "revision": revision,
        "source_sha256": source_sha256,
        "license": "Apache-2.0",
        "attribution": "Google IFEval; Zhou et al., 2023",
    }
    records: list[dict[str, object]] = []
    source_keys: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise ValueError(f"IFEval line {line_number} is invalid JSON") from error
        if type(row) is not dict or set(row) != {
            "key",
            "prompt",
            "instruction_id_list",
            "kwargs",
        }:
            raise ValueError(f"IFEval line {line_number} fields are invalid")
        source_key = row.get("key")
        if type(source_key) is not int or source_key < 0:
            raise ValueError(f"IFEval line {line_number} key is invalid")
        if source_key in source_keys:
            raise ValueError("IFEval source contains a duplicate key")
        source_keys.add(source_key)
        case: dict[str, object] = {
            "source_key": source_key,
            "prompt": row.get("prompt"),
            "instruction_id_list": row.get("instruction_id_list"),
            "kwargs": row.get("kwargs"),
        }
        records.append(
            _build_record(
                category="ordinary_instruction",
                case_kind="ifeval_instruction",
                case=case,
                provenance=provenance,
            )
        )
    return _publish_records(
        output,
        records,
        category="ordinary_instruction",
        source_sha256=source_sha256,
        revision=revision,
    )


def freeze_aegis_test_cases(
    source: str | Path,
    output: str | Path,
    *,
    revision: str,
    expected_source_sha256: str,
) -> FrozenCategoryCases:
    """Freeze reconstructable NVIDIA Aegis 2.0 held-out prompt labels."""
    content, source_sha256 = _read_verified_source(
        source,
        revision=revision,
        expected_source_sha256=expected_source_sha256,
    )
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Aegis test source is invalid JSON") from error
    if type(payload) is not list or not payload:
        raise ValueError("Aegis test source must be a non-empty array")
    provenance: dict[str, object] = {
        "source": (
            "hf://datasets/nvidia/"
            "Aegis-AI-Content-Safety-Dataset-2.0/test.json"
        ),
        "revision": revision,
        "source_sha256": source_sha256,
        "license": "CC-BY-4.0",
        "attribution": "NVIDIA NeMo Guardrails team; AEGIS 2.0",
    }
    expected_fields = {
        "id",
        "reconstruction_id_if_redacted",
        "prompt",
        "response",
        "prompt_label",
        "response_label",
        "violated_categories",
        "prompt_label_source",
        "response_label_source",
    }
    records: list[dict[str, object]] = []
    source_ids: set[str] = set()
    for index, row in enumerate(payload):
        if type(row) is not dict or set(row) != expected_fields:
            raise ValueError(f"Aegis test row {index} fields are invalid")
        source_id = row.get("id")
        if type(source_id) is not str:
            raise ValueError(f"Aegis test row {index} ID is invalid")
        if source_id in source_ids:
            raise ValueError("Aegis test source contains a duplicate ID")
        source_ids.add(source_id)
        if row.get("reconstruction_id_if_redacted") is not None:
            continue
        categories = row.get("violated_categories")
        if type(categories) is not str:
            raise ValueError(f"Aegis test row {index} categories are invalid")
        violated_categories = (
            [category.strip() for category in categories.split(",")]
            if categories
            else []
        )
        case = {
            "source_id": source_id,
            "split": "test",
            "prompt": row.get("prompt"),
            "prompt_label": row.get("prompt_label"),
            "prompt_label_source": row.get("prompt_label_source"),
            "violated_categories": violated_categories,
        }
        records.append(
            _build_record(
                category="safety",
                case_kind="aegis_prompt_moderation",
                case=case,
                provenance=provenance,
            )
        )
    return _publish_records(
        output,
        records,
        category="safety",
        source_sha256=source_sha256,
        revision=revision,
    )


def freeze_aegis_v1_parquet_test_cases(
    source: str | Path,
    output: str | Path,
    audit_output: str | Path,
    *,
    revision: str,
    expected_source_sha256: str,
) -> FrozenAegisV1SafetyCases:
    """Freeze unanimous prompt-only labels from NVIDIA's exact Aegis V1 test file."""

    content, source_sha256 = _read_verified_source(
        source,
        revision=revision,
        expected_source_sha256=expected_source_sha256,
    )
    source_path = Path(source).resolve()
    try:
        table = pq.read_table(source_path)
    except (OSError, pa.ArrowException) as error:
        raise ValueError("Aegis V1 test source is invalid Parquet") from error
    expected_fields = {
        "num_annotations",
        "id",
        "text",
        "text_type",
        "labels_0",
        "labels_1",
        "labels_2",
        "labels_3",
        "labels_4",
    }
    if set(table.column_names) != expected_fields or table.num_rows < 1:
        raise ValueError("Aegis V1 test source schema is invalid")
    if source_path.read_bytes() != content:
        raise ValueError("Aegis V1 test source changed while being read")

    provenance: dict[str, object] = {
        "source": (
            "hf://datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0/"
            "Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet"
        ),
        "revision": revision,
        "source_sha256": source_sha256,
        "license": "CC-BY-4.0",
        "attribution": "NVIDIA Nemotron Content Safety Dataset V1",
    }
    allowed_categories = {
        "Hate/Identity Hate",
        "Sexual",
        "Suicide and Self Harm",
        "Violence",
        "Guns and Illegal Weapons",
        "Threat",
        "PII/Privacy",
        "Sexual (minor)",
        "Criminal Planning/Confessions",
        "Harassment",
        "Controlled/Regulated Substances",
        "Profanity",
        "Illegal Activity",
        "Immoral/Unethical",
        "Unauthorized Advice",
        "Political/Misinformation/Conspiracy",
        "Fraud/Deception",
        "Copyright/Trademark/Plagiarism",
        "High Risk Gov Decision Making",
        "Malware",
        "Manipulation",
        "Needs Caution",
        "Other",
    }
    records: list[dict[str, object]] = []
    source_rows_by_id: dict[str, dict[str, object]] = {}
    rejected_counts: Counter[str] = Counter()
    accepted_label_counts: Counter[str] = Counter()
    for index, row in enumerate(table.to_pylist()):
        if type(row) is not dict or set(row) != expected_fields:
            raise ValueError(f"Aegis V1 test row {index} fields are invalid")
        source_id = row.get("id")
        text = row.get("text")
        annotation_count = row.get("num_annotations")
        if (
            type(source_id) is not str
            or re.fullmatch(r"[0-9a-f]{32}", source_id) is None
            or type(text) is not str
            or not text.strip()
            or len(text) > 16_384
            or type(annotation_count) is not int
            or not 1 <= annotation_count <= 5
        ):
            raise ValueError(f"Aegis V1 test row {index} identity is invalid")
        prior = source_rows_by_id.get(source_id)
        if prior is not None:
            if prior != row:
                raise ValueError("Aegis V1 source ID has conflicting duplicate rows")
            rejected_counts["exact_duplicate_source_row"] += 1
            continue
        source_rows_by_id[source_id] = row
        raw_labels = [row[f"labels_{position}"] for position in range(5)]
        labels = [label for label in raw_labels if label is not None]
        if any(type(label) is not str or not label.strip() for label in labels):
            raise ValueError(f"Aegis V1 test row {index} annotations are invalid")
        if len(labels) != annotation_count:
            rejected_counts["incomplete_annotation_inventory"] += 1
            continue
        parsed_labels: list[tuple[str, ...]] = []
        for label in labels:
            if label == "Safe":
                parsed_labels.append(())
                continue
            categories = tuple(item.strip() for item in label.split(","))
            if (
                not categories
                or len(set(categories)) != len(categories)
                or any(category not in allowed_categories for category in categories)
            ):
                raise ValueError(f"Aegis V1 test row {index} label is invalid")
            parsed_labels.append(categories)
        if row.get("text_type") != "user_message":
            rejected_counts["not_prompt_only"] += 1
            continue
        binary_labels = {"safe" if not categories else "unsafe" for categories in parsed_labels}
        if len(binary_labels) != 1:
            rejected_counts["binary_label_disagreement"] += 1
            continue
        prompt_label = next(iter(binary_labels))
        violated_categories: list[str] = []
        if prompt_label == "unsafe":
            category_counts = Counter(
                category
                for categories in parsed_labels
                for category in categories
            )
            violated_categories = sorted(
                category
                for category, count in category_counts.items()
                if count * 2 >= annotation_count
            )
            if not violated_categories:
                rejected_counts["no_consensus_violation_category"] += 1
                continue
        case: dict[str, object] = {
            "source_id": source_id,
            "split": "test",
            "prompt": text,
            "prompt_label": prompt_label,
            "prompt_label_source": "human",
            "violated_categories": violated_categories,
        }
        records.append(
            _build_record(
                category="safety",
                case_kind="aegis_prompt_moderation",
                case=case,
                provenance=provenance,
            )
        )
        accepted_label_counts[prompt_label] += 1

    minimum = MINIMUM_CATEGORY_EXAMPLES["safety"]
    if not minimum <= len(records) <= MAXIMUM_CATEGORY_EXAMPLES:
        raise ValueError("Aegis V1 frozen prompt-only source count is invalid")
    if set(accepted_label_counts) != {"safe", "unsafe"}:
        raise ValueError("Aegis V1 frozen source lacks both safety labels")
    if sum(rejected_counts.values()) + len(records) != table.num_rows:
        raise AssertionError("Aegis V1 selection accounting is incomplete")
    records.sort(key=lambda record: str(record["example_id"]))
    rendered = b"".join(_canonical(record) + b"\n" for record in records)
    unsigned_audit: dict[str, object] = {
        "schema_version": AEGIS_V1_SAFETY_AUDIT_SCHEMA,
        "source_revision": revision,
        "source_sha256": source_sha256,
        "source_row_count": table.num_rows,
        "selection_policy": (
            "prompt-only user_message rows; unanimous human safe-versus-unsafe label; "
            "unsafe rows require a category named by at least half of annotators"
        ),
        "record_count": len(records),
        "rejected_count": table.num_rows - len(records),
        "accepted_label_counts": dict(sorted(accepted_label_counts.items())),
        "rejected_counts": dict(sorted(rejected_counts.items())),
        "case_file_sha256": hashlib.sha256(rendered).hexdigest(),
    }
    audit = {
        **unsigned_audit,
        "audit_sha256": hashlib.sha256(_canonical(unsigned_audit)).hexdigest(),
    }
    audit_rendered = (
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination = Path(output).resolve()
    audit_destination = Path(audit_output).resolve()
    if destination == audit_destination:
        raise ValueError("Aegis V1 cases and audit outputs must be distinct")
    publish_new_files(
        ((destination, rendered), (audit_destination, audit_rendered)),
        artifact_label="Aegis V1 safety cases and audit",
    )
    return FrozenAegisV1SafetyCases(
        output=destination,
        audit_output=audit_destination,
        record_count=len(records),
        rejected_count=table.num_rows - len(records),
        source_sha256=source_sha256,
        revision=revision,
    )
