"""Run pinned local safety moderation and publish its immutable receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.model_category_evidence import checkpoint_sha256
from bosscrafts_model.safety_judge_case_runner import (
    MAX_SAFETY_BATCH_BYTES,
    build_safety_content_receipt,
    publish_safety_content_receipt,
    run_safety_judge_cases,
)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_bounded(path: Path, label: str) -> bytes:
    try:
        content = path.resolve().read_bytes()
    except OSError as error:
        raise ValueError(f"{label} is unreadable") from error
    if not content or len(content) > MAX_SAFETY_BATCH_BYTES:
        raise ValueError(f"{label} size is invalid")
    return content


def _read_cases(path: Path) -> list[dict[str, object]]:
    try:
        lines = _read_bounded(path, "safety cases").decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("safety cases are not UTF-8") from error
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("safety cases must be non-empty JSONL")
    result: list[dict[str, object]] = []
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise ValueError(f"safety case line {index} is invalid JSON") from error
        if type(value) is not dict:
            raise ValueError(f"safety case line {index} is not an object")
        result.append(value)
    return result


def _response_map(path: Path) -> dict[str, str]:
    try:
        value = json.loads(
            _read_bounded(path, "candidate responses").decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("candidate responses are invalid JSON") from error
    if type(value) is not dict or any(
        type(key) is not str or type(response) is not str
        for key, response in value.items()
    ):
        raise ValueError("candidate responses must be a JSON string map")
    return {str(key): str(response) for key, response in value.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--judge-python", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--candidate-responses", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7_200)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    cases = _read_cases(arguments.cases)
    responses = _response_map(arguments.candidate_responses)
    result = run_safety_judge_cases(
        arguments.runtime_root,
        arguments.judge_python,
        cases,
        responses,
        timeout_seconds=arguments.timeout_seconds,
    )
    receipt = build_safety_content_receipt(
        result,
        responses,
        checkpoint_sha256=checkpoint_sha256(arguments.checkpoint),
        checkpoint_step=arguments.checkpoint_step,
    )
    publish_safety_content_receipt(arguments.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
