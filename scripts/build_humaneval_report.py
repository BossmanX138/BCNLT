"""Build a fail-closed, hash-bound report from an isolated HumanEval run."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model import humaneval_sandbox
from bosscrafts_model.generation_artifacts import (
    HUMANEVAL_GENERATION_SCHEMA,
    HumanEvalGenerationRecord,
    load_humaneval_generation_records,
    validate_humaneval_generation_records,
)
from bosscrafts_model.immutable_artifacts import publish_new_files
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config

SCHEMA = "bosscrafts-openai-humaneval-report-v3"
TASK_COUNT = 164
OPENAI_HUMANEVAL_SHA256 = (
    "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
)
MINIMUM_PASS_AT_1 = 0.05
MINIMUM_PASSED_TASKS = math.ceil(TASK_COUNT * MINIMUM_PASS_AT_1)
MINIMUM_SYNTAX_VALID_RATE = 0.75
MINIMUM_PROTOCOL_COMPLETE_RATE = 0.95
_PASS_AT_1 = re.compile(
    r"['\"]?pass@1['\"]?\s*:\s*(?:np\.float64\()?([0-9]+(?:\.[0-9]+)?)"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_sha256",
        "dataset_sha256",
        "database_identity_sha256",
        "device",
        "dtype",
        "limit",
        "max_new_tokens",
        "model_fingerprint",
        "tokenizer_sha256",
        "binding_sha256",
        "generation",
        "prompt_protocol",
        "run_identity_sha256",
    }
)
_SUMMARY_FIELDS = _MANIFEST_FIELDS | {
    "checkpoint_epoch_index",
    "checkpoint_step",
    "completed_tasks",
    "planned_tasks",
    "generation_complete",
    "total_generated_tokens",
    "total_generation_seconds",
    "mean_tokens_per_second",
}


class _HumanEvalTask(TypedDict):
    task_id: str
    prompt: str
    test: str
    entry_point: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON evidence: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"JSON evidence is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {line_number}: {path}") from error
            if type(row) is not dict:
                raise ValueError(f"non-object JSONL row {line_number}: {path}")
            rows.append(row)
    return rows


def _classify_failure(result: str) -> str:
    lowered = result.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if any(
        marker in lowered
        for marker in (
            "invalid syntax",
            "syntaxerror",
            "indentationerror",
            "unexpected indent",
            "unindent",
            "taberror",
        )
    ):
        return "syntax_error"
    if result.strip() in {"failed:", "failed"} or "assert" in lowered:
        return "assertion_failure"
    return "runtime_or_test_failure"


def _read_tasks(path: Path) -> list[_HumanEvalTask]:
    if not path.is_file() or _sha256(path) != OPENAI_HUMANEVAL_SHA256:
        raise ValueError("official HumanEval dataset is missing or has the wrong hash")
    tasks: list[_HumanEvalTask] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id") if type(row) is dict else None
            prompt = row.get("prompt") if type(row) is dict else None
            tests = row.get("test") if type(row) is dict else None
            entry_point = row.get("entry_point") if type(row) is dict else None
            if (
                type(task_id) is not str
                or type(prompt) is not str
                or type(tests) is not str
                or type(entry_point) is not str
            ):
                raise ValueError("official HumanEval dataset row is invalid")
            tasks.append(
                _HumanEvalTask(
                    task_id=task_id,
                    prompt=prompt,
                    test=tests,
                    entry_point=entry_point,
                )
            )
    if [task["task_id"] for task in tasks] != [
        f"HumanEval/{index}" for index in range(TASK_COUNT)
    ]:
        raise ValueError("official HumanEval task order is invalid")
    return tasks


def _validate_sample_results(
    expected_ids: list[str],
    records: list[HumanEvalGenerationRecord],
    samples: list[dict[str, object]],
    results: list[dict[str, object]],
) -> None:
    if (
        len(records) != len(expected_ids)
        or len(samples) != len(expected_ids)
        or len(results) != len(expected_ids)
    ):
        raise ValueError("HumanEval samples or results have the wrong row count")
    for task_id, record, sample, result_row in zip(
        expected_ids, records, samples, results, strict=True
    ):
        if (
            set(sample) != {"task_id", "completion"}
            or sample.get("task_id") != task_id
            or sample.get("completion") != record["clean_completion"]
        ):
            raise ValueError("HumanEval samples do not match generation records")
        if (
            result_row.get("task_id") != task_id
            or result_row.get("completion") != sample["completion"]
            or type(result_row.get("passed")) is not bool
            or type(result_row.get("result")) is not str
        ):
            raise ValueError("HumanEval results do not match submitted samples")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="official HumanEval.jsonl.gz; defaults to the benchmark source directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    generation = arguments.generation_dir.resolve()
    evaluation = arguments.evaluation_dir.resolve()
    dataset_path = (
        arguments.dataset.resolve()
        if arguments.dataset is not None
        else generation.parent / "source" / "HumanEval.jsonl.gz"
    )
    manifest_path = generation / "manifest.json"
    summary_path = generation / "summary.json"
    records_path = generation / "generation-records.jsonl"
    samples_path = generation / "samples.jsonl"
    metrics_path = evaluation / "metrics.txt"
    results_candidates = (
        evaluation / "samples.jsonl_results.jsonl",
        evaluation / "samples.jsonl_results.jsonl.gz",
    )
    if results_candidates[1].exists():
        raise ValueError("gzip HumanEval results are not supported by this report builder")
    results_path = results_candidates[0]
    receipt_path = evaluation / "receipt.json"
    required = (
        manifest_path,
        summary_path,
        records_path,
        samples_path,
        metrics_path,
        results_path,
        receipt_path,
    )
    if any(not path.is_file() for path in required):
        raise ValueError("HumanEval evidence set is incomplete")

    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    receipt = _read_json(receipt_path)
    tasks = _read_tasks(dataset_path)
    results = humaneval_sandbox.load_results(results_path)
    expected_ids = [f"HumanEval/{index}" for index in range(TASK_COUNT)]
    run_identity = manifest.get("run_identity_sha256")
    if (
        set(manifest) != _MANIFEST_FIELDS
        or manifest.get("schema_version") != HUMANEVAL_GENERATION_SCHEMA
        or type(run_identity) is not str
        or _SHA256.fullmatch(run_identity) is None
    ):
        raise ValueError("HumanEval manifest contract is invalid")
    identity_payload = dict(manifest)
    identity_payload.pop("run_identity_sha256")
    recomputed_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if recomputed_identity != run_identity:
        raise ValueError("HumanEval generation identity mismatch")
    maximum_new_tokens_value = manifest.get("max_new_tokens")
    if (
        manifest.get("dataset_sha256") != OPENAI_HUMANEVAL_SHA256
        or manifest.get("limit") != TASK_COUNT
        or manifest.get("generation") != "greedy_argmax"
        or manifest.get("prompt_protocol") != "instruction_to_code"
        or type(maximum_new_tokens_value) is not int
        or not 1 <= maximum_new_tokens_value <= 512
        or manifest.get("device") not in {"cpu", "cuda"}
        or manifest.get("dtype") not in {"float32", "bfloat16"}
        or any(
            _SHA256.fullmatch(str(manifest.get(field))) is None
            for field in (
                "checkpoint_sha256",
                "database_identity_sha256",
                "model_fingerprint",
                "tokenizer_sha256",
                "binding_sha256",
            )
        )
    ):
        raise ValueError("HumanEval manifest is not a release-gate run")
    maximum_new_tokens = maximum_new_tokens_value
    binding = NativeTokenizerBinding.load()
    config = bound_native_65k_config(binding)
    if (
        manifest.get("binding_sha256") != binding.binding_sha256
        or manifest.get("tokenizer_sha256") != binding.tokenizer_sha256
        or manifest.get("model_fingerprint") != config.fingerprint()
    ):
        raise ValueError("HumanEval manifest tokenizer or model lineage mismatch")
    records_by_id = load_humaneval_generation_records(records_path, run_identity)
    if list(records_by_id) != expected_ids:
        raise ValueError("generation records do not cover official HumanEval order")
    records: list[HumanEvalGenerationRecord] = [
        records_by_id[task_id] for task_id in expected_ids
    ]
    validate_humaneval_generation_records(
        binding,
        tasks,
        records_by_id,
        prompt_protocol="instruction_to_code",
        maximum_new_tokens=maximum_new_tokens,
        maximum_sequence=config.max_sequence,
    )
    if set(summary) != _SUMMARY_FIELDS or any(
        summary.get(key) != value for key, value in manifest.items()
    ):
        raise ValueError("HumanEval summary lineage does not match the manifest")
    if (
        summary.get("generation_complete") is not True
        or summary.get("completed_tasks") != TASK_COUNT
        or summary.get("planned_tasks") != TASK_COUNT
    ):
        raise ValueError("HumanEval generation did not complete all tasks")
    total_generated_tokens = sum(record["generated_token_count"] for record in records)
    total_generation_seconds = sum(record["generation_seconds"] for record in records)
    checkpoint_epoch_index = summary.get("checkpoint_epoch_index")
    checkpoint_step = summary.get("checkpoint_step")
    summary_seconds = summary.get("total_generation_seconds")
    summary_rate = summary.get("mean_tokens_per_second")
    if (
        type(checkpoint_epoch_index) is not int
        or checkpoint_epoch_index < 0
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
        or summary.get("total_generated_tokens") != total_generated_tokens
        or isinstance(summary_seconds, bool)
        or not isinstance(summary_seconds, (int, float))
        or not math.isfinite(summary_seconds)
        or not math.isclose(
            summary_seconds,
            total_generation_seconds,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or isinstance(summary_rate, bool)
        or not isinstance(summary_rate, (int, float))
        or not math.isfinite(summary_rate)
        or not math.isclose(
            summary_rate,
            total_generated_tokens / total_generation_seconds,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("HumanEval summary metrics are inconsistent")
    samples = _read_jsonl(samples_path)
    _validate_sample_results(expected_ids, records, samples, results)
    for index, (task, sample, result) in enumerate(
        zip(tasks, samples, results, strict=True)
    ):
        humaneval_sandbox.validate_result(result, task, sample, task_index=index)
    evaluator_path = PROJECT_ROOT / "scripts" / "evaluate_humaneval_sandboxed.py"
    humaneval_sandbox.validate_receipt(
        receipt,
        results,
        dataset_sha256=_sha256(dataset_path),
        samples_sha256=_sha256(samples_path),
        results_bytes=results_path.read_bytes(),
        metrics_bytes=metrics_path.read_bytes(),
        evaluator_source_sha256=_sha256(evaluator_path),
    )

    metrics_text = metrics_path.read_text(encoding="utf-8")
    matches = _PASS_AT_1.findall(metrics_text)
    if len(matches) != 1:
        raise ValueError("HumanEval pass@1 metric is missing or ambiguous")
    pass_at_1 = float(matches[0])
    passed = sum(row.get("passed") is True for row in results)
    if not math.isclose(pass_at_1, passed / TASK_COUNT, abs_tol=1e-12):
        raise ValueError("HumanEval metric disagrees with per-task results")

    failures: Counter[str] = Counter()
    for row in results:
        if row.get("passed") is True:
            continue
        result_text = row.get("result")
        task_id_value = row.get("task_id")
        completion_value = row.get("completion")
        if (
            type(result_text) is not str
            or type(task_id_value) is not str
            or type(completion_value) is not str
        ):
            raise ValueError("HumanEval result row has invalid failure text")
        try:
            prompt = tasks[int(task_id_value.removeprefix("HumanEval/"))]["prompt"]
            compile(prompt + completion_value, f"<{task_id_value}>", "exec")
        except SyntaxError:
            failures["syntax_error"] += 1
        else:
            failures[_classify_failure(result_text)] += 1
    prompt_protocol = manifest.get("prompt_protocol")
    raw_protocols = {"raw", "raw_openai_humaneval_prompt"}
    protocol_complete = sum(
        (
            row.get("response_protocol_complete") is True
            if prompt_protocol not in raw_protocols
            else row.get("response_protocol_complete", True) is True
        )
        for row in records
    )
    protocol_complete_rate = protocol_complete / TASK_COUNT
    syntax_valid_rate = 1.0 - failures["syntax_error"] / TASK_COUNT
    thresholds = {
        "minimum_pass_at_1": MINIMUM_PASS_AT_1,
        "minimum_passed_tasks": MINIMUM_PASSED_TASKS,
        "minimum_syntax_valid_rate": MINIMUM_SYNTAX_VALID_RATE,
        "minimum_protocol_complete_rate": MINIMUM_PROTOCOL_COMPLETE_RATE,
        "required_prompt_protocol": "instruction_to_code",
    }
    gate_passed = (
        prompt_protocol == "instruction_to_code"
        and passed >= MINIMUM_PASSED_TASKS
        and syntax_valid_rate >= MINIMUM_SYNTAX_VALID_RATE
        and protocol_complete_rate >= MINIMUM_PROTOCOL_COMPLETE_RATE
    )
    report: dict[str, object] = {
        "schema_version": SCHEMA,
        "benchmark_status": "completed",
        "task_count": TASK_COUNT,
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "checkpoint_step": summary.get("checkpoint_step"),
        "checkpoint_epoch_index": summary.get("checkpoint_epoch_index"),
        "task_database_identity_sha256": manifest.get(
            "database_identity_sha256"
        ),
        "model_fingerprint": manifest.get("model_fingerprint"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "binding_sha256": manifest.get("binding_sha256"),
        "prompt_protocol": prompt_protocol,
        "max_new_tokens": manifest.get("max_new_tokens"),
        "run_identity_sha256": run_identity,
        "pass_at_1": pass_at_1,
        "passed": passed,
        "failed": TASK_COUNT - passed,
        "syntax_valid_rate": syntax_valid_rate,
        "protocol_complete_count": protocol_complete,
        "protocol_complete_rate": protocol_complete_rate,
        "sandbox_image_manifest_sha256": receipt.get(
            "sandbox_image_manifest_sha256"
        ),
        "sandbox_image_ids": receipt.get("sandbox_image_ids"),
        "sandbox_evaluator_source_sha256": receipt.get(
            "evaluator_source_sha256"
        ),
        "sandbox_result_attestation_sha256": receipt.get(
            "result_attestation_sha256"
        ),
        "failure_breakdown": {
            name: failures[name]
            for name in (
                "syntax_error",
                "assertion_failure",
                "runtime_or_test_failure",
                "timeout",
            )
        },
        "release_coding_thresholds": thresholds,
        "release_coding_gate_passed": gate_passed,
        "evidence_sha256": {
            "dataset": _sha256(dataset_path),
            "manifest": _sha256(manifest_path),
            "summary": _sha256(summary_path),
            "generation_records": _sha256(records_path),
            "samples": _sha256(samples_path),
            "metrics": _sha256(metrics_path),
            "results": _sha256(results_path),
            "sandbox_receipt": _sha256(receipt_path),
        },
    }
    report["report_manifest_sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    rendered = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_new_files(
        ((arguments.output.resolve(), rendered),),
        artifact_label="HumanEval report",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
