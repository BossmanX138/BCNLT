from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


REQUIRED_ARTIFACT_ROLES = (
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


@dataclass(frozen=True)
class FullRunProposal:
    command: tuple[str, ...]
    estimated_steps: int
    estimated_disk_bytes: int
    manifest_hashes: dict[str, str]
    started: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "estimated_steps": self.estimated_steps,
            "estimated_disk_bytes": self.estimated_disk_bytes,
            "manifest_hashes": dict(sorted(self.manifest_hashes.items())),
            "started": self.started,
        }


def prepare_full_run(
    handoff_path: str | Path,
    *,
    disk_free_bytes: int,
    gpu_memory_mib: int,
    start_training: bool = False,
    resume_path: str | Path | None = None,
    expected_resume_sha256: str | None = None,
    executor: Callable[[list[str]], int] | None = None,
) -> FullRunProposal:
    handoff_file = Path(handoff_path)
    raw = json.loads(handoff_file.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "bcnlt-full-run-handoff-v1":
        raise ValueError("unsupported full-run handoff schema")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("handoff artifacts must be an object")
    hashes = {}
    resolved = {}
    for role in REQUIRED_ARTIFACT_ROLES:
        if role not in artifacts:
            raise ValueError(f"missing required artifact: {role}")
        entry = artifacts[role]
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError(f"invalid artifact entry: {role}")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = handoff_file.parent / path
        if not path.is_file():
            raise ValueError(f"missing {role} file: {path}")
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"{role} hash does not match the handoff")
        hashes[role] = actual
        resolved[role] = path

    requirements = raw.get("requirements", {})
    minimum_disk = int(requirements.get("minimum_disk_free_bytes", 0))
    minimum_gpu = int(requirements.get("minimum_gpu_memory_mib", 0))
    if disk_free_bytes < minimum_disk:
        raise RuntimeError("available disk space is below the full-run requirement")
    if gpu_memory_mib < minimum_gpu:
        raise RuntimeError("available GPU memory is below the full-run requirement")

    corpus_validation = _load_object(resolved["corpus_validation"])
    if corpus_validation.get("passed") is not True:
        raise ValueError("corpus validation has not passed")
    smoke = _load_object(resolved["smoke_manifest"])
    if smoke.get("finite_loss") is not True or int(
        smoke.get("completed_training_examples", 0)
    ) <= 0:
        raise ValueError("smoke manifest does not prove a finite completed training step")
    promotion = _load_object(resolved["smoke_promotion"])
    if promotion.get("promoted") is not True or promotion.get("failure_codes"):
        raise ValueError("smoke promotion has not passed")

    if resume_path is not None:
        resume = Path(resume_path)
        if not resume.is_file() or not expected_resume_sha256:
            raise ValueError("resume hash and checkpoint are both required")
        if _sha256(resume) != expected_resume_sha256:
            raise ValueError("resume hash does not match the requested checkpoint")

    training = raw.get("training")
    if not isinstance(training, dict) or not isinstance(training.get("command"), list):
        raise ValueError("handoff training command is missing")
    command = tuple(str(part) for part in training["command"])
    if not command:
        raise ValueError("handoff training command is empty")
    started = False
    if start_training:
        runner = executor or _execute
        exit_code = runner(list(command))
        if exit_code != 0:
            raise RuntimeError(f"full training command failed with exit code {exit_code}")
        started = True
    return FullRunProposal(
        command=command,
        estimated_steps=int(training.get("estimated_steps", 0)),
        estimated_disk_bytes=int(training.get("estimated_disk_bytes", 0)),
        manifest_hashes=hashes,
        started=started,
    )


def _execute(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def _load_object(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or launch the guarded BCNLT full run")
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--disk-free-bytes", required=True, type=int)
    parser.add_argument("--gpu-memory-mib", required=True, type=int)
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--resume-sha256")
    args = parser.parse_args(argv)
    proposal = prepare_full_run(
        args.handoff,
        disk_free_bytes=args.disk_free_bytes,
        gpu_memory_mib=args.gpu_memory_mib,
        start_training=args.start_training,
        resume_path=args.resume,
        expected_resume_sha256=args.resume_sha256,
    )
    print(json.dumps(proposal.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
