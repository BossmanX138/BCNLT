"""Execute the pinned IFEval batch and publish its immutable receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.ifeval_case_runner import (
    MAX_IFEVAL_BATCH_BYTES,
    build_ifeval_batch_receipt,
    publish_ifeval_batch_receipt,
    run_ifeval_cases,
)
from bosscrafts_model.model_category_evidence import checkpoint_sha256


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
    if not content or len(content) > MAX_IFEVAL_BATCH_BYTES:
        raise ValueError(f"{label} size is invalid")
    return content


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            _read_bounded(path, label).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_cases(path: Path) -> list[dict[str, object]]:
    try:
        text = _read_bounded(path, "IFEval cases").decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("IFEval cases are not UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("IFEval cases must be non-empty JSONL")
    result: list[dict[str, object]] = []
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise ValueError(f"IFEval case line {index} is invalid JSON") from error
        if type(value) is not dict:
            raise ValueError(f"IFEval case line {index} is not an object")
        result.append(value)
    return result


def _response_map(path: Path, label: str) -> dict[str, str]:
    value = _read_object(path, label)
    if any(type(key) is not str or type(response) is not str for key, response in value.items()):
        raise ValueError(f"{label} values must be strings")
    return {str(key): str(response) for key, response in value.items()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.resolve().open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError("baseline evaluation is unreadable") from error
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--candidate-responses", type=Path, required=True)
    parser.add_argument("--baseline-responses", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--baseline-evaluation", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    cases = _read_cases(arguments.cases)
    candidate = _response_map(arguments.candidate_responses, "candidate responses")
    baseline = _response_map(arguments.baseline_responses, "baseline responses")
    result = run_ifeval_cases(
        arguments.runtime_root,
        cases,
        candidate,
        baseline,
        timeout_seconds=arguments.timeout_seconds,
    )
    receipt = build_ifeval_batch_receipt(
        result,
        candidate,
        baseline,
        checkpoint_sha256=checkpoint_sha256(arguments.checkpoint),
        checkpoint_step=arguments.checkpoint_step,
        baseline_source=(
            str(arguments.baseline_evaluation.resolve())
            if arguments.baseline_evaluation is not None
            else "deterministic_random_initialization"
        ),
        baseline_evaluation_sha256=(
            _file_sha256(arguments.baseline_evaluation)
            if arguments.baseline_evaluation is not None
            else None
        ),
    )
    publish_ifeval_batch_receipt(arguments.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
