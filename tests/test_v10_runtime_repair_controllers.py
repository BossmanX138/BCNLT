from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from scripts import audit_v10_task_loss_masks_v3_runtime_repair as loss_masks
from scripts import (
    materialize_v10_evaluation_eligibility_v3_runtime_repair as eval_eligibility,
)
from scripts import (
    materialize_v10_training_eligibility_v3_runtime_repair as train_eligibility,
)
from scripts import pack_v10_audited_training_tasks_v3_runtime_repair as train_packing
from scripts import pack_v10_frozen_evaluation_tasks_v3_runtime_repair as eval_packing
from scripts import (
    run_v10_frozen_evaluation_admission_v3_runtime_repair as eval_admission,
)
from scripts import (
    run_v10_frozen_evaluation_semantic_audit_v3_runtime_repair as eval_audit,
)
from scripts import (
    run_v10_frozen_training_admission_v3_runtime_repair as train_admission,
)
from scripts import run_v10_training_semantic_audit_v3_runtime_repair as train_audit

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
    names = (
        "run_v10_frozen_training_admission_v3_runtime_repair.py",
        "run_v10_training_semantic_audit_v3_runtime_repair.py",
        "materialize_v10_training_eligibility_v3_runtime_repair.py",
        "pack_v10_audited_training_tasks_v3_runtime_repair.py",
        "run_v10_frozen_evaluation_admission_v3_runtime_repair.py",
        "run_v10_frozen_evaluation_semantic_audit_v3_runtime_repair.py",
        "materialize_v10_evaluation_eligibility_v3_runtime_repair.py",
        "pack_v10_frozen_evaluation_tasks_v3_runtime_repair.py",
        "audit_v10_task_loss_masks_v3_runtime_repair.py",
    )
    for name in names:
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / name), "--help"],
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=15,
            check=False,
        )
        assert completed.returncode == 0, f"{name}: {completed.stderr}"
        assert "usage:" in completed.stdout

    status_scripts = (
        "run_v10_frozen_training_admission_v3_runtime_repair.py",
        "run_v10_training_semantic_audit_v3_runtime_repair.py",
        "pack_v10_audited_training_tasks_v3_runtime_repair.py",
        "run_v10_frozen_evaluation_admission_v3_runtime_repair.py",
        "run_v10_frozen_evaluation_semantic_audit_v3_runtime_repair.py",
        "pack_v10_frozen_evaluation_tasks_v3_runtime_repair.py",
    )
    for name in status_scripts:
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / name), "--help"],
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=15,
            check=False,
        )
        assert completed.returncode == 0, f"{name}: {completed.stderr}"
        assert "--status-only" in completed.stdout


def test_training_lineage_is_shared_exactly() -> None:
    assert train_eligibility.DEFAULT_ADMISSION == train_audit.DEFAULT_ADMISSION
    assert train_eligibility.DEFAULT_AUDIT == train_audit.DEFAULT_AUDIT
    assert train_packing.DEFAULT_ADMISSION == train_admission.DEFAULT_DATABASE
    assert train_packing.DEFAULT_ELIGIBILITY == train_eligibility.DEFAULT_OUTPUT
    assert loss_masks.DEFAULT_TRAIN == train_packing.DEFAULT_DATABASE
    assert train_eligibility.AUDIT_HOST_CODE_FILES == train_audit.AUDIT_HOST_CODE_FILES
    assert (
        train_eligibility.audit_host_code_identity()
        == train_audit.audit_host_code_identity()
    )


def test_all_outputs_are_isolated_to_repaired_v3() -> None:
    paths = (
        train_audit.DEFAULT_AUDIT,
        train_eligibility.DEFAULT_OUTPUT,
        train_packing.DEFAULT_DATABASE,
        loss_masks.DEFAULT_TRAIN,
        loss_masks.DEFAULT_VALIDATION,
        loss_masks.DEFAULT_TEST,
        loss_masks.DEFAULT_OUTPUT,
    )
    for path in paths:
        assert "v3-runtime-repair" in path.as_posix()
        assert "v2-code-audit" not in path.as_posix()


def test_evaluation_pipeline_uses_frozen_source_and_repaired_runtime() -> None:
    assert eval_admission.DEFAULT_FREEZE == train_admission.DEFAULT_FREEZE
    assert (
        eval_admission.FREEZE_MANIFEST_SHA256 == train_admission.FREEZE_MANIFEST_SHA256
    )
    assert eval_audit.DEFAULT_FREEZE == train_admission.DEFAULT_FREEZE
    assert eval_audit.FREEZE_MANIFEST_SHA256 == train_admission.FREEZE_MANIFEST_SHA256
    assert eval_audit.QUALIFICATION_FILE_SHA256 == train_audit.QUALIFICATION_FILE_SHA256
    assert (
        eval_eligibility.audit_host_code_identity()
        == eval_audit.audit_host_code_identity()
    )
    training_admission_path = (
        "scripts/run_v10_frozen_training_admission_v3_runtime_repair.py"
    )
    assert training_admission_path in eval_admission.HOST_CODE_FILES
    assert training_admission_path in eval_audit.AUDIT_HOST_CODE_FILES


def test_loss_mask_inputs_are_exact_evaluation_packing_defaults() -> None:
    root = eval_packing.PROJECT_ROOT / "artifacts" / "corpora"
    assert loss_masks.DEFAULT_VALIDATION == (
        root / "bcnlt-v10-frozen-validation-tasks-v3-runtime-repair" / "tasks.sqlite"
    )
    assert loss_masks.DEFAULT_TEST == (
        root / "bcnlt-v10-frozen-test-tasks-v3-runtime-repair" / "tasks.sqlite"
    )
