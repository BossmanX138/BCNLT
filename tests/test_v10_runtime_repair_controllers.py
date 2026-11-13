from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v3_runtime_repair_controllers_form_an_isolated_pipeline() -> None:
    admission = importlib.import_module(
        "scripts.run_v10_frozen_training_admission_v3_runtime_repair"
    )
    audit = importlib.import_module(
        "scripts.run_v10_training_semantic_audit_v3_runtime_repair"
    )

    assert audit.DEFAULT_ADMISSION == admission.DEFAULT_DATABASE
    assert audit.DEFAULT_FREEZE == admission.DEFAULT_FREEZE
    assert audit.admission_host_code_identity() == admission.host_code_identity()
    for path in (admission.DEFAULT_DATABASE, audit.DEFAULT_AUDIT):
        assert "v3-runtime-repair" in path.as_posix()
        assert "v2-code-audit" not in path.as_posix()


def test_v3_runtime_repair_controllers_have_executable_help() -> None:
    for name in (
        "run_v10_frozen_training_admission_v3_runtime_repair.py",
        "run_v10_training_semantic_audit_v3_runtime_repair.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / name), "--help"],
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=15,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--status-only" in completed.stdout

