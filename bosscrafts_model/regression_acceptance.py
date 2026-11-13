"""Execute the fixed full regression and publish checkpoint-bound evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .immutable_artifacts import publish_new_files

REGRESSION_WORKER_SCHEMA = "bosscrafts-regression-worker-v1"
REGRESSION_RECEIPT_SCHEMA = "bosscrafts-regression-receipt-v1"
REGRESSION_SUITE_ID = "bosscrafts-pre-model-full-regression-v1"
MAX_REGRESSION_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_REGRESSION_RECORDS = 100_000
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKER_SCRIPT = _PROJECT_ROOT / "scripts" / "run_regression_worker.py"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class RegressionResult:
    git_commit: str
    python_version: str
    pytest_version: str
    returncode: int
    records: tuple[dict[str, object], ...]
    stdout_sha256: str
    stderr_sha256: str


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate regression key: {key}")
        result[key] = value
    return result


def _git_output(project_root: Path, *arguments: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise ValueError("regression Git executable is unavailable")
    try:
        completed = subprocess.run(  # noqa: S603
            [str(Path(git).resolve()), "-C", str(project_root), *arguments],
            capture_output=True,
            timeout=30,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("regression Git identity is unavailable") from error
    if completed.returncode != 0 or len(completed.stdout) > 1024 * 1024:
        raise ValueError("regression Git identity is invalid")
    try:
        return completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("regression Git output is not UTF-8") from error


def _clean_git_commit(project_root: Path) -> str:
    commit = _git_output(project_root, "rev-parse", "HEAD")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("regression Git commit is invalid")
    if _git_output(project_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("regression suite requires a clean Git worktree")
    return commit


def _worker_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
    }
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in allowed
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return environment


def _invoke_worker(
    python_executable: Path,
    project_root: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    try:
        completed = subprocess.run(  # noqa: S603
            [
                str(python_executable),
                "-I",
                str(_WORKER_SCRIPT),
                "--project-root",
                str(project_root),
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            shell=False,
            cwd=project_root,
            env=_worker_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("regression worker failed to complete") from error
    if not completed.stdout or len(completed.stdout) > MAX_REGRESSION_OUTPUT_BYTES:
        raise ValueError("regression worker output size is invalid")
    try:
        value = json.loads(
            completed.stdout.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        detail = completed.stderr.decode("utf-8", errors="replace")[:4096]
        raise ValueError(f"regression worker output is invalid: {detail}") from error
    if type(value) is not dict:
        raise ValueError("regression worker output must be an object")
    return value


_WORKER_FIELDS = {
    "schema_version",
    "git_commit",
    "python_version",
    "pytest_version",
    "returncode",
    "records",
    "records_sha256",
    "stdout_sha256",
    "stderr_sha256",
}


def _validated_records(value: object) -> list[dict[str, object]]:
    if (
        type(value) is not list
        or not value
        or len(value) > MAX_REGRESSION_RECORDS
    ):
        raise ValueError("regression result inventory is invalid")
    fields = {"record_id", "kind", "nodeid", "phase", "outcome", "context"}
    records: list[dict[str, object]] = []
    ids: set[str] = set()
    for item in value:
        if type(item) is not dict or set(item) != fields:
            raise ValueError("regression result record fields are invalid")
        body = {key: member for key, member in item.items() if key != "record_id"}
        record_id = item.get("record_id")
        if (
            not _is_sha256(record_id)
            or record_id != _sha256(body)
            or record_id in ids
            or item.get("kind") not in {"test", "subtest", "collection"}
            or type(item.get("nodeid")) is not str
            or not str(item["nodeid"])
            or item.get("phase") not in {"collect", "setup", "call", "teardown"}
            or item.get("outcome") not in {"passed", "failed", "skipped"}
            or (item.get("context") is not None and type(item.get("context")) is not str)
        ):
            raise ValueError("regression result record is invalid")
        ids.add(str(record_id))
        records.append(item)
    if [str(item["record_id"]) for item in records] != sorted(ids):
        raise ValueError("regression result records are not canonically ordered")
    return records


def _validated_worker_result(
    value: object, *, expected_git_commit: str
) -> RegressionResult:
    if type(value) is not dict or set(value) != _WORKER_FIELDS:
        raise ValueError("regression worker fields are invalid")
    records = _validated_records(value.get("records"))
    returncode = value.get("returncode")
    if (
        value.get("schema_version") != REGRESSION_WORKER_SCHEMA
        or value.get("git_commit") != expected_git_commit
        or type(value.get("python_version")) is not str
        or not str(value["python_version"])
        or type(value.get("pytest_version")) is not str
        or not str(value["pytest_version"])
        or type(returncode) is not int
        or not 0 <= returncode <= 255
        or value.get("records_sha256") != _sha256(records)
        or not _is_sha256(value.get("stdout_sha256"))
        or not _is_sha256(value.get("stderr_sha256"))
    ):
        raise ValueError("regression worker result is invalid")
    return RegressionResult(
        git_commit=expected_git_commit,
        python_version=str(value["python_version"]),
        pytest_version=str(value["pytest_version"]),
        returncode=returncode,
        records=tuple(records),
        stdout_sha256=str(value["stdout_sha256"]),
        stderr_sha256=str(value["stderr_sha256"]),
    )


def run_regression_suite(
    python_executable: str | Path,
    *,
    project_root: str | Path = _PROJECT_ROOT,
    timeout_seconds: int = 1_800,
) -> RegressionResult:
    root = Path(project_root).resolve()
    executable = Path(python_executable).resolve()
    if (
        root != _PROJECT_ROOT
        or not executable.is_file()
        or type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= 7_200
    ):
        raise ValueError("regression runner arguments are invalid")
    commit = _clean_git_commit(root)
    return _validated_worker_result(
        _invoke_worker(executable, root, timeout_seconds),
        expected_git_commit=commit,
    )


def _summary(records: Sequence[Mapping[str, object]], returncode: int) -> dict[str, object]:
    test_count = sum(
        record.get("kind") == "test"
        and record.get("phase") == "call"
        and record.get("outcome") == "passed"
        for record in records
    )
    subtest_count = sum(
        record.get("kind") == "subtest" and record.get("outcome") == "passed"
        for record in records
    )
    failure_count = sum(record.get("outcome") != "passed" for record in records)
    return {
        "full_regression_passed": returncode == 0 and failure_count == 0,
        "test_count": test_count,
        "subtest_count": subtest_count,
        "failure_count": failure_count,
    }


def build_regression_receipt(
    result: RegressionResult,
    *,
    checkpoint_sha256: str,
    checkpoint_step: int,
    example_id: str,
) -> dict[str, object]:
    if (
        not _is_sha256(checkpoint_sha256)
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
        or not _is_sha256(example_id)
        or _GIT_COMMIT.fullmatch(result.git_commit) is None
    ):
        raise ValueError("regression receipt lineage is invalid")
    records = _validated_records(list(result.records))
    summary = _summary(records, result.returncode)
    category_record = {"example_id": example_id, **summary}
    body: dict[str, object] = {
        "schema_version": REGRESSION_RECEIPT_SCHEMA,
        "suite_id": REGRESSION_SUITE_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "git_commit": result.git_commit,
        "python_version": result.python_version,
        "pytest_version": result.pytest_version,
        "returncode": result.returncode,
        "records": records,
        "records_sha256": _sha256(records),
        "stdout_sha256": result.stdout_sha256,
        "stderr_sha256": result.stderr_sha256,
        "category_record": category_record,
    }
    return {**body, "receipt_sha256": _sha256(body)}


_RECEIPT_FIELDS = {
    "schema_version",
    "suite_id",
    "checkpoint_sha256",
    "checkpoint_step",
    "git_commit",
    "python_version",
    "pytest_version",
    "returncode",
    "records",
    "records_sha256",
    "stdout_sha256",
    "stderr_sha256",
    "category_record",
    "receipt_sha256",
}


def validate_regression_receipt(
    payload: object,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
    expected_example_ids: Sequence[str] | None = None,
    expected_records: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != _RECEIPT_FIELDS:
        raise ValueError("regression receipt fields are invalid")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != _sha256(body):
        raise ValueError("regression receipt integrity mismatch")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    checkpoint_step = payload.get("checkpoint_step")
    returncode = payload.get("returncode")
    if (
        payload.get("schema_version") != REGRESSION_RECEIPT_SCHEMA
        or payload.get("suite_id") != REGRESSION_SUITE_ID
        or not _is_sha256(checkpoint_sha256)
        or type(checkpoint_step) is not int
        or checkpoint_step < 1
        or type(payload.get("git_commit")) is not str
        or _GIT_COMMIT.fullmatch(str(payload["git_commit"])) is None
        or type(payload.get("python_version")) is not str
        or type(payload.get("pytest_version")) is not str
        or type(returncode) is not int
        or not 0 <= returncode <= 255
        or not _is_sha256(payload.get("stdout_sha256"))
        or not _is_sha256(payload.get("stderr_sha256"))
    ):
        raise ValueError("regression receipt lineage is invalid")
    records = _validated_records(payload.get("records"))
    if payload.get("records_sha256") != _sha256(records):
        raise ValueError("regression receipt records integrity mismatch")
    category_record = payload.get("category_record")
    if type(category_record) is not dict or category_record != {
        "example_id": category_record.get("example_id"),
        **_summary(records, returncode),
    } or not _is_sha256(category_record.get("example_id")):
        raise ValueError("regression receipt category record is inconsistent")
    if expected_checkpoint_sha256 is not None and checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("regression receipt checkpoint mismatch")
    if expected_checkpoint_step is not None and checkpoint_step != expected_checkpoint_step:
        raise ValueError("regression receipt checkpoint step mismatch")
    if expected_example_ids is not None and tuple(expected_example_ids) != (
        str(category_record["example_id"]),
    ):
        raise ValueError("regression receipt expected inventory mismatch")
    if expected_records is not None and list(expected_records) != [category_record]:
        raise ValueError("regression receipt expected records mismatch")
    return payload


def regression_receipt_metrics(payload: object) -> dict[str, object]:
    receipt = validate_regression_receipt(payload)
    return {
        "regression_receipt_sha256": receipt["receipt_sha256"],
        "regression_git_commit": receipt["git_commit"],
    }


def publish_regression_receipt(output: str | Path, payload: dict[str, object]) -> Path:
    validated = validate_regression_receipt(payload)
    rendered = (
        json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(rendered) > MAX_REGRESSION_OUTPUT_BYTES:
        raise ValueError("regression receipt exceeds its size bound")
    destination = Path(output).resolve()
    publish_new_files(
        ((destination, rendered),), artifact_label="regression acceptance receipt"
    )
    return destination


class _RegressionPlugin:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def _append(
        self,
        *,
        kind: str,
        nodeid: str,
        phase: str,
        outcome: str,
        context: str | None,
    ) -> None:
        body: dict[str, object] = {
            "kind": kind,
            "nodeid": nodeid,
            "phase": phase,
            "outcome": outcome,
            "context": context,
        }
        self.records.append({**body, "record_id": _sha256(body)})

    def pytest_runtest_logreport(self, report: Any) -> None:
        context_value = getattr(report, "context", None)
        is_subtest = context_value is not None or type(report).__name__ == "SubTestReport"
        if is_subtest:
            self._append(
                kind="subtest",
                nodeid=str(report.nodeid),
                phase=str(report.when),
                outcome=str(report.outcome),
                context=str(context_value) if context_value is not None else None,
            )
        elif report.when == "call" or report.failed:
            self._append(
                kind="test",
                nodeid=str(report.nodeid),
                phase=str(report.when),
                outcome=str(report.outcome),
                context=None,
            )

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self._append(
                kind="collection",
                nodeid=str(report.nodeid),
                phase="collect",
                outcome="failed",
                context=None,
            )


def worker_main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import contextlib
    import io

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        project_root = arguments.project_root.resolve()
        if project_root != _PROJECT_ROOT:
            raise ValueError("regression worker project root is invalid")
        git_commit = _clean_git_commit(project_root)
        pytest = __import__("pytest")
        plugin = _RegressionPlugin()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = int(
                pytest.main(
                    [
                        "-q",
                        "--tb=short",
                        "--disable-warnings",
                        str(project_root / "tests"),
                    ],
                    plugins=[plugin],
                )
            )
        stdout_bytes = stdout.getvalue().encode("utf-8")
        stderr_bytes = stderr.getvalue().encode("utf-8")
        if (
            len(stdout_bytes) > MAX_REGRESSION_OUTPUT_BYTES
            or len(stderr_bytes) > MAX_REGRESSION_OUTPUT_BYTES
        ):
            raise ValueError("regression output exceeds its size bound")
        records = sorted(plugin.records, key=lambda record: str(record["record_id"]))
        payload: dict[str, object] = {
            "schema_version": REGRESSION_WORKER_SCHEMA,
            "git_commit": git_commit,
            "python_version": sys.version,
            "pytest_version": str(pytest.__version__),
            "returncode": returncode,
            "records": records,
            "records_sha256": _sha256(records),
            "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        }
        rendered = _canonical(payload)
        if len(rendered) > MAX_REGRESSION_OUTPUT_BYTES:
            raise ValueError("regression worker result exceeds its size bound")
        sys.stdout.buffer.write(rendered)
        return 0
    except (OSError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
