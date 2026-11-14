from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bcnlt.full_run_gate import prepare_full_run


REQUIRED = (
    "source_registry",
    "tokenizer_manifest",
    "corpus_manifest",
    "corpus_validation",
    "base_identity",
    "adapter_manifest",
    "smoke_manifest",
    "base_evaluation",
    "adapter_evaluation",
    "smoke_promotion",
    "training_config",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handoff(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for role in REQUIRED:
        path = tmp_path / f"{role}.json"
        payload = {"role": role}
        if role in {"base_evaluation", "adapter_evaluation"}:
            subject_role = (
                "base_identity" if role == "base_evaluation" else "adapter_manifest"
            )
            payload = {
                "schema_version": "bcnlt-model-evaluation-v1",
                "evaluated_artifact": {
                    "role": subject_role,
                    "sha256": "PENDING",
                },
                "identity": role,
                "settings": {"seed": 1729, "temperature": 0.0, "max_new_tokens": 64},
                "results": [
                    {
                        "case_id": "case-1",
                        "category": "raw_code",
                        "prompt": "prompt",
                        "expected": "expected",
                        "output": "expected",
                        "passed": True,
                        "error": None,
                    }
                ],
                "metrics": {"raw_code_accuracy": 1.0},
                "runtime_failures": [],
                "heldout_loss": 1.0,
            }
        if role == "corpus_validation":
            payload["passed"] = True
        if role == "smoke_manifest":
            payload.update({"finite_loss": True, "completed_training_examples": 1})
        if role == "smoke_promotion":
            payload.update({"promoted": True, "failure_codes": []})
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifacts[role] = {"path": str(path), "sha256": _sha(path)}
    for evaluation_role, subject_role in (
        ("base_evaluation", "base_identity"),
        ("adapter_evaluation", "adapter_manifest"),
    ):
        path = Path(artifacts[evaluation_role]["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["evaluated_artifact"]["sha256"] = artifacts[subject_role]["sha256"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifacts[evaluation_role]["sha256"] = _sha(path)
    command = ["python", "-m", "trainer", "--full"]
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "schema_version": "bcnlt-full-run-handoff-v1",
                "artifacts": artifacts,
                "requirements": {
                    "minimum_disk_free_bytes": 100,
                    "minimum_gpu_memory_mib": 50,
                },
                "training": {
                    "command": command,
                    "command_sha256": hashlib.sha256(
                        json.dumps(command, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "estimated_steps": 123,
                    "estimated_disk_bytes": 456,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return handoff


@pytest.mark.parametrize("missing_role", REQUIRED)
def test_gate_refuses_every_missing_artifact(tmp_path: Path, missing_role: str):
    handoff = _handoff(tmp_path)
    raw = json.loads(handoff.read_text())
    del raw["artifacts"][missing_role]
    handoff.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match=missing_role):
        prepare_full_run(handoff, disk_free_bytes=1000, gpu_memory_mib=1000)


def test_gate_refuses_changed_artifact_disk_gpu_and_resume_hashes(tmp_path: Path):
    handoff = _handoff(tmp_path)
    raw = json.loads(handoff.read_text())
    changed = Path(raw["artifacts"]["tokenizer_manifest"]["path"])
    changed.write_text("changed")
    with pytest.raises(ValueError, match="tokenizer_manifest hash"):
        prepare_full_run(handoff, disk_free_bytes=1000, gpu_memory_mib=1000)

    handoff = _handoff(tmp_path / "fresh")
    with pytest.raises(RuntimeError, match="disk"):
        prepare_full_run(handoff, disk_free_bytes=99, gpu_memory_mib=1000)
    with pytest.raises(RuntimeError, match="GPU"):
        prepare_full_run(handoff, disk_free_bytes=1000, gpu_memory_mib=49)

    resume = tmp_path / "resume.pt"
    resume.write_bytes(b"resume")
    with pytest.raises(ValueError, match="resume hash"):
        prepare_full_run(
            handoff,
            disk_free_bytes=1000,
            gpu_memory_mib=1000,
            resume_path=resume,
            expected_resume_sha256="0" * 64,
        )


def test_gate_requires_passing_corpus_smoke_and_promotion(tmp_path: Path):
    for role, update, message in (
        ("corpus_validation", {"passed": False}, "corpus validation"),
        ("smoke_manifest", {"finite_loss": False}, "smoke manifest"),
        ("smoke_promotion", {"promoted": False}, "smoke promotion"),
    ):
        root = tmp_path / role
        handoff = _handoff(root)
        raw = json.loads(handoff.read_text())
        artifact = Path(raw["artifacts"][role]["path"])
        payload = json.loads(artifact.read_text())
        payload.update(update)
        artifact.write_text(json.dumps(payload))
        raw["artifacts"][role]["sha256"] = _sha(artifact)
        handoff.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match=message):
            prepare_full_run(handoff, disk_free_bytes=1000, gpu_memory_mib=1000)


def test_default_is_validation_only_and_explicit_start_uses_exact_command(tmp_path: Path):
    handoff = _handoff(tmp_path)
    calls = []

    proposal = prepare_full_run(
        handoff,
        disk_free_bytes=1000,
        gpu_memory_mib=1000,
        executor=lambda command: calls.append(command) or 0,
    )
    assert not proposal.started
    assert not calls
    assert proposal.command == ("python", "-m", "trainer", "--full")
    assert proposal.estimated_steps == 123
    assert proposal.manifest_hashes

    started = prepare_full_run(
        handoff,
        disk_free_bytes=1000,
        gpu_memory_mib=1000,
        start_training=True,
        executor=lambda command: calls.append(command) or 0,
    )
    assert started.started
    assert calls == [list(proposal.command)]


@pytest.mark.parametrize(
    ("role", "mutation", "message"),
    (
        (
            "base_evaluation",
            lambda payload: payload.update({"schema_version": "unknown"}),
            "base evaluation schema",
        ),
        (
            "adapter_evaluation",
            lambda payload: payload.update({"results": []}),
            "adapter evaluation results",
        ),
        (
            "base_evaluation",
            lambda payload: payload["evaluated_artifact"].update(
                {"sha256": "0" * 64}
            ),
            "base evaluation artifact identity",
        ),
    ),
)
def test_gate_validates_evaluation_schema_and_subject_identity(
    tmp_path: Path, role: str, mutation, message: str
):
    handoff = _handoff(tmp_path)
    raw = json.loads(handoff.read_text(encoding="utf-8"))
    artifact = Path(raw["artifacts"][role]["path"])
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    mutation(payload)
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    raw["artifacts"][role]["sha256"] = _sha(artifact)
    handoff.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        prepare_full_run(handoff, disk_free_bytes=1000, gpu_memory_mib=1000)


def test_gate_refuses_a_training_command_that_does_not_match_its_identity(tmp_path: Path):
    handoff = _handoff(tmp_path)
    raw = json.loads(handoff.read_text(encoding="utf-8"))
    raw["training"]["command"][-1] = "--different"
    handoff.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="training command identity"):
        prepare_full_run(handoff, disk_free_bytes=1000, gpu_memory_mib=1000)
