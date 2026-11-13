from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bosscrafts_model.regression_acceptance import (
    RegressionResult,
    _RegressionPlugin,
    build_regression_receipt,
    publish_regression_receipt,
    validate_regression_receipt,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _record(
    kind: str, nodeid: str, *, context: str | None = None
) -> dict[str, object]:
    body: dict[str, object] = {
        "kind": kind,
        "nodeid": nodeid,
        "phase": "call",
        "outcome": "passed",
        "context": context,
    }
    return {**body, "record_id": hashlib.sha256(_canonical(body)).hexdigest()}


def _result() -> RegressionResult:
    records = tuple(
        sorted(
            (
                _record("test", "tests/test_a.py::test_a"),
                _record("test", "tests/test_b.py::test_b"),
                _record(
                    "subtest",
                    "tests/test_b.py::test_b",
                    context="value=1",
                ),
            ),
            key=lambda record: str(record["record_id"]),
        )
    )
    return RegressionResult(
        git_commit="a" * 40,
        python_version="3.13.9 fixture",
        pytest_version="9.1.1",
        returncode=0,
        records=records,
        stdout_sha256="b" * 64,
        stderr_sha256="c" * 64,
    )


def test_builds_recomputable_checkpoint_bound_regression_receipt(
    tmp_path: Path,
) -> None:
    receipt = build_regression_receipt(
        _result(),
        checkpoint_sha256="d" * 64,
        checkpoint_step=17,
        example_id="e" * 64,
    )

    assert receipt["category_record"] == {
        "example_id": "e" * 64,
        "full_regression_passed": True,
        "test_count": 2,
        "subtest_count": 1,
        "failure_count": 0,
    }
    assert validate_regression_receipt(
        receipt,
        expected_checkpoint_sha256="d" * 64,
        expected_checkpoint_step=17,
        expected_example_ids=("e" * 64,),
        expected_records=[receipt["category_record"]],
    ) == receipt
    output = tmp_path / "regression.json"
    assert publish_regression_receipt(output, receipt) == output.resolve()
    with pytest.raises(ValueError, match="already exists"):
        publish_regression_receipt(output, receipt)

    forged = json.loads(json.dumps(receipt))
    forged["category_record"]["test_count"] = 999
    body = {
        key: value for key, value in forged.items() if key != "receipt_sha256"
    }
    forged["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(ValueError, match="category record is inconsistent"):
        validate_regression_receipt(forged)


def test_plugin_records_tests_subtests_and_collection_failures() -> None:
    plugin = _RegressionPlugin()
    plugin.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="tests/test_a.py::test_a",
            when="call",
            outcome="passed",
            failed=False,
        )
    )
    plugin.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="tests/test_a.py::test_a",
            when="call",
            outcome="passed",
            failed=False,
            context="value=1",
        )
    )
    plugin.pytest_collectreport(
        SimpleNamespace(
            nodeid="tests/test_b.py",
            failed=True,
        )
    )

    assert [record["kind"] for record in plugin.records] == [
        "test",
        "subtest",
        "collection",
    ]


def test_regression_acceptance_script_exposes_fixed_inputs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "run_regression_acceptance.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--python-executable" in completed.stdout
    assert "--checkpoint" in completed.stdout
    assert "--checkpoint-step" in completed.stdout
    assert "--example-id" in completed.stdout
    assert "--output" in completed.stdout
