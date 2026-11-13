"""Build a fail-closed, hash-bound report from an isolated HumanEval run."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "bosscrafts-openai-humaneval-report-v2"
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


def _read_prompts(path: Path) -> dict[str, str]:
    if not path.is_file() or _sha256(path) != OPENAI_HUMANEVAL_SHA256:
        raise ValueError("official HumanEval dataset is missing or has the wrong hash")
    prompts: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id") if type(row) is dict else None
            prompt = row.get("prompt") if type(row) is dict else None
            if type(task_id) is not str or type(prompt) is not str or task_id in prompts:
                raise ValueError("official HumanEval dataset row is invalid")
            prompts[task_id] = prompt
    if list(prompts) != [f"HumanEval/{index}" for index in range(TASK_COUNT)]:
        raise ValueError("official HumanEval task order is invalid")
    return prompts


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    required = (
        manifest_path,
        summary_path,
        records_path,
        samples_path,
        metrics_path,
        results_path,
    )
    if any(not path.is_file() for path in required):
        raise ValueError("HumanEval evidence set is incomplete")

    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    records = _read_jsonl(records_path)
    results = _read_jsonl(results_path)
    prompts = _read_prompts(dataset_path)
    expected_ids = [f"HumanEval/{index}" for index in range(TASK_COUNT)]
    if [row.get("task_id") for row in records] != expected_ids:
        raise ValueError("generation records do not cover official HumanEval order")
    if [row.get("task_id") for row in results] != expected_ids:
        raise ValueError("evaluation results do not cover official HumanEval order")
    run_identity = manifest.get("run_identity_sha256")
    if (
        type(run_identity) is not str
        or len(run_identity) != 64
        or summary.get("run_identity_sha256") != run_identity
        or any(row.get("run_identity_sha256") != run_identity for row in records)
    ):
        raise ValueError("HumanEval generation identity mismatch")
    if (
        summary.get("generation_complete") is not True
        or summary.get("completed_tasks") != TASK_COUNT
        or summary.get("planned_tasks") != TASK_COUNT
    ):
        raise ValueError("HumanEval generation did not complete all tasks")

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
        result = row.get("result")
        task_id = row.get("task_id")
        completion = row.get("completion")
        if type(result) is not str or type(task_id) is not str or type(completion) is not str:
            raise ValueError("HumanEval result row has invalid failure text")
        try:
            compile(prompts[task_id] + completion, f"<{task_id}>", "exec")
        except SyntaxError:
            failures["syntax_error"] += 1
        else:
            failures[_classify_failure(result)] += 1
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
        },
    }
    report["report_manifest_sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    _atomic_json(arguments.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
