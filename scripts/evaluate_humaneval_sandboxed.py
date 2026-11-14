"""Execute official HumanEval tasks one-by-one in the attested BossCrafts sandbox."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.humaneval_sandbox import (
    MAX_RESULT_LINE_BYTES,
    HumanEvalSample,
    HumanEvalTask,
    build_receipt,
    canonical_json,
    evaluate_sample,
    render_results,
    validate_result,
)
from bosscrafts_model.immutable_artifacts import publish_new_files
from bosscrafts_model.scientific_sandbox import DockerScientificSandbox

OPENAI_HUMANEVAL_SHA256 = (
    "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
)
TASK_COUNT = 164
MAX_SAMPLES_BYTES = 16 * 1024 * 1024
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "artifacts"
    / "benchmarks"
    / "openai-humaneval-v1"
    / "source"
    / "HumanEval.jsonl.gz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tasks(path: Path) -> list[HumanEvalTask]:
    if not path.is_file() or _sha256(path) != OPENAI_HUMANEVAL_SHA256:
        raise ValueError("official HumanEval dataset is missing or has the wrong hash")
    tasks: list[HumanEvalTask] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_unique_object)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"HumanEval task {line_number} is invalid JSON") from error
            required = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}
            if (
                type(row) is not dict
                or set(row) != required
                or any(type(row[field]) is not str for field in required)
            ):
                raise ValueError(f"HumanEval task {line_number} fields are invalid")
            tasks.append(
                HumanEvalTask(
                    task_id=row["task_id"],
                    prompt=row["prompt"],
                    test=row["test"],
                    entry_point=row["entry_point"],
                )
            )
    if [task["task_id"] for task in tasks] != [
        f"HumanEval/{index}" for index in range(TASK_COUNT)
    ]:
        raise ValueError("official HumanEval task order is invalid")
    return tasks


def _read_samples(path: Path) -> list[HumanEvalSample]:
    if not path.is_file() or path.stat().st_size > MAX_SAMPLES_BYTES:
        raise ValueError("HumanEval samples are missing or exceed the size limit")
    samples: list[HumanEvalSample] = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline(MAX_RESULT_LINE_BYTES + 1)
            if not raw:
                break
            if len(raw) > MAX_RESULT_LINE_BYTES or not raw.endswith(b"\n"):
                raise ValueError("HumanEval sample has invalid bounded framing")
            try:
                row = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise ValueError("HumanEval sample is invalid JSON") from error
            if (
                type(row) is not dict
                or set(row) != {"task_id", "completion"}
                or type(row.get("task_id")) is not str
                or type(row.get("completion")) is not str
            ):
                raise ValueError("HumanEval sample fields are invalid")
            samples.append(
                HumanEvalSample(
                    task_id=row["task_id"],
                    completion=row["completion"],
                )
            )
    if [sample["task_id"] for sample in samples] != [
        f"HumanEval/{index}" for index in range(TASK_COUNT)
    ]:
        raise ValueError("HumanEval samples do not cover the official task order")
    return samples


def _load_progress_record(
    path: Path,
    task: HumanEvalTask,
    sample: HumanEvalSample,
    task_index: int,
) -> dict[str, object]:
    raw = path.read_bytes()
    if len(raw) > MAX_RESULT_LINE_BYTES or not raw.endswith(b"\n"):
        raise ValueError("HumanEval progress record has invalid bounded framing")
    try:
        record = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("HumanEval progress record is invalid JSON") from error
    if type(record) is not dict:
        raise ValueError("HumanEval progress record must be an object")
    validate_result(record, task, sample, task_index=task_index)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    dataset = arguments.dataset.resolve()
    samples_path = arguments.samples.resolve()
    output = arguments.output.resolve()
    tasks = _read_tasks(dataset)
    samples = _read_samples(samples_path)
    work = output / "task-records"
    sandbox = DockerScientificSandbox.load_default()
    records: list[dict[str, object]] = []
    for index, (task, sample) in enumerate(zip(tasks, samples, strict=True)):
        record_path = work / f"{index:03d}.json"
        if record_path.is_file():
            record = _load_progress_record(record_path, task, sample, index)
        else:
            record = evaluate_sample(sandbox, task, sample, task_index=index)
            validate_result(record, task, sample, task_index=index)
            publish_new_files(
                ((record_path, canonical_json(record)),),
                artifact_label="HumanEval task record",
            )
        records.append(record)
        print(
            json.dumps(
                {
                    "completed_tasks": len(records),
                    "planned_tasks": TASK_COUNT,
                    "task_id": task["task_id"],
                    "passed": record["passed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    results_bytes = render_results(records)
    passed_count = sum(record["passed"] is True for record in records)
    pass_at_1 = passed_count / TASK_COUNT
    metrics_bytes = (json.dumps({"pass@1": pass_at_1}, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    evaluator_source_sha256 = _sha256(Path(__file__).resolve())
    receipt = build_receipt(
        dataset_sha256=_sha256(dataset),
        samples_sha256=_sha256(samples_path),
        results_bytes=results_bytes,
        metrics_bytes=metrics_bytes,
        evaluator_source_sha256=evaluator_source_sha256,
        records=records,
    )
    publish_new_files(
        (
            (output / "samples.jsonl_results.jsonl", results_bytes),
            (output / "metrics.txt", metrics_bytes),
            (output / "receipt.json", canonical_json(receipt)),
        ),
        artifact_label="HumanEval sandbox evaluation",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate HumanEval key: {key}")
        result[key] = value
    return result


if __name__ == "__main__":
    raise SystemExit(main())
