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
        if role == "corpus_validation":
            payload["passed"] = True
        if role == "smoke_manifest":
            payload.update({"finite_loss": True, "completed_training_examples": 1})
        if role == "smoke_promotion":
            payload.update({"promoted": True, "failure_codes": []})
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifacts[role] = {"path": str(path), "sha256": _sha(path)}
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
                    "command": ["python", "-m", "trainer", "--full"],
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
