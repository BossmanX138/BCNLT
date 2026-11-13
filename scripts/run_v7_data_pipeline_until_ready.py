"""Wait for frozen V7 admission, then build and audit the compact curriculum."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.resumable_task_packing import (  # noqa: E402
    TaskDatabaseAuditResult,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402


ADMISSION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-full-resumable-v1"
    / "admission.sqlite"
)
ADMISSION_PID = ADMISSION.with_name("admission.pid")
OVERSAMPLED = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-balanced-oversampled-compact-v2-v1"
    / "tasks.sqlite"
)
DETERMINISTIC = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-deterministic-clean-v1"
    / "tasks.sqlite"
)
DETERMINISM_AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluations"
    / "task-prompt-determinism-v7.json"
)
DEFAULT_STATUS = (
    PROJECT_ROOT / "artifacts" / "pipelines" / "bcnlt-v7-data-v1" / "status.json"
)
EXPECTED_OVERSAMPLED_EXAMPLES = 193_536
EXPECTED_DETERMINISTIC_EXAMPLES = 96_768
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _admission_status() -> dict[str, object]:
    if not ADMISSION.is_file():
        return {
            "database": str(ADMISSION),
            "database_exists": False,
            "catalog_exhausted": False,
            "processed": 0,
            "admitted": 0,
            "rejected": 0,
        }
    connection = sqlite3.connect(
        f"{ADMISSION.as_uri()}?mode=ro", uri=True, timeout=30
    )
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        counts = connection.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN status='admitted' THEN 1 ELSE 0 END),
                      SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END)
               FROM documents"""
        ).fetchone()
    finally:
        connection.close()
    if counts is None:
        raise RuntimeError("admission count query returned no row")
    return {
        "database": str(ADMISSION),
        "database_exists": True,
        "schema_version": metadata.get("schema_version"),
        "host_code_identity_sha256": metadata.get("host_code_identity_sha256"),
        "catalog_exhausted": metadata.get("catalog_exhausted") == "true",
        "processed": int(counts[0] or 0),
        "admitted": int(counts[1] or 0),
        "rejected": int(counts[2] or 0),
    }


def _process_is_active(process_id: int) -> bool:
    if os.name != "nt":
        raise RuntimeError("the V7 pipeline controller currently requires Windows")
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, process_id
    )
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise OSError("GetExitCodeProcess failed")
        return int(exit_code.value) == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _admission_process_id() -> int:
    try:
        value = int(ADMISSION_PID.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise RuntimeError("V7 admission PID file is missing or invalid") from error
    if value < 1:
        raise RuntimeError("V7 admission PID is invalid")
    return value


def _run_script(name: str) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / name)]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")


def _require_task_count(
    audit: TaskDatabaseAuditResult, expected: int, label: str
) -> None:
    if not audit.audit_passed or audit.example_count != expected:
        raise ValueError(
            f"{label} task database expected {expected} examples, "
            f"observed {audit.example_count}"
        )


def _artifact_state() -> dict[str, object]:
    return {
        "oversampled_exists": OVERSAMPLED.is_file(),
        "deterministic_exists": DETERMINISTIC.is_file(),
        "determinism_audit_exists": DETERMINISM_AUDIT.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--status-only", action="store_true")
    arguments = parser.parse_args()
    if not 5 <= arguments.poll_seconds <= 300:
        parser.error("poll-seconds must be between 5 and 300")

    admission = _admission_status()
    if arguments.status_only:
        print(json.dumps({"admission": admission, **_artifact_state()}, indent=2, sort_keys=True))
        return 0

    while admission["catalog_exhausted"] is not True:
        process_id = _admission_process_id()
        if not _process_is_active(process_id):
            raise RuntimeError(
                "V7 admission stopped before catalog exhaustion; preserved data may be resumed"
            )
        print(
            json.dumps(
                {
                    "stage": "waiting_for_admission",
                    "admission_pid": process_id,
                    **admission,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(arguments.poll_seconds)
        admission = _admission_status()

    binding = NativeTokenizerBinding.load()
    if not OVERSAMPLED.exists():
        _run_script("build_balanced_curriculum.py")
    oversampled_audit = audit_task_database(binding, OVERSAMPLED)
    _require_task_count(
        oversampled_audit, EXPECTED_OVERSAMPLED_EXAMPLES, "oversampled"
    )

    if not DETERMINISTIC.exists():
        _run_script("derive_deterministic_curriculum.py")
    deterministic_audit = audit_task_database(binding, DETERMINISTIC)
    _require_task_count(
        deterministic_audit, EXPECTED_DETERMINISTIC_EXAMPLES, "deterministic"
    )

    if not DETERMINISM_AUDIT.exists():
        _run_script("analyze_task_prompt_determinism.py")
    determinism = json.loads(DETERMINISM_AUDIT.read_text(encoding="utf-8"))
    summary = determinism.get("summary")
    if not (
        determinism.get("audit_complete") is True
        and determinism.get("prompt_targets_deterministic") is True
        and determinism.get("heldout_prompt_isolation") is True
        and determinism.get("task_database_identity_sha256")
        == deterministic_audit.database_identity_sha256
        and type(summary) is dict
        and summary.get("example_count") == EXPECTED_DETERMINISTIC_EXAMPLES
        and summary.get("unique_prompt_count") == EXPECTED_DETERMINISTIC_EXAMPLES
        and summary.get("duplicate_prompt_group_count") == 0
        and summary.get("conflicting_prompt_group_count") == 0
        and summary.get("cross_split_prompt_group_count") == 0
    ):
        raise ValueError("V7 deterministic curriculum audit did not pass")

    payload: dict[str, object] = {
        "schema_version": "bosscrafts-v7-data-pipeline-status-v1",
        "data_pipeline_ready": True,
        "admission": admission,
        "oversampled_task_database": str(OVERSAMPLED),
        "oversampled_task_database_identity_sha256": (
            oversampled_audit.database_identity_sha256
        ),
        "oversampled_example_count": oversampled_audit.example_count,
        "deterministic_task_database": str(DETERMINISTIC),
        "deterministic_task_database_identity_sha256": (
            deterministic_audit.database_identity_sha256
        ),
        "deterministic_example_count": deterministic_audit.example_count,
        "determinism_audit": str(DETERMINISM_AUDIT),
        "prompt_targets_deterministic": True,
        "heldout_prompt_isolation": True,
    }
    _write_json_atomic(arguments.status, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
