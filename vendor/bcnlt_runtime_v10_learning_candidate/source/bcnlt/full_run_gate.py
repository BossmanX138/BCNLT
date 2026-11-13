from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeGuard


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
MAX_HANDOFF_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_JSON_BYTES = 16 * 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
    expected_handoff_sha256: str | None = None,
    resume_path: str | Path | None = None,
    expected_resume_sha256: str | None = None,
    executor: Callable[[list[str]], int] | None = None,
) -> FullRunProposal:
    if not _is_int(disk_free_bytes) or disk_free_bytes < 0:
        raise ValueError("available disk space must be a non-negative integer")
    if not _is_int(gpu_memory_mib) or gpu_memory_mib < 0:
        raise ValueError("available GPU memory must be a non-negative integer")
    handoff_file = Path(handoff_path)
    handoff_bytes = handoff_file.read_bytes()
    if not handoff_bytes or len(handoff_bytes) > MAX_HANDOFF_BYTES:
        raise ValueError("full-run handoff has an invalid byte length")
    actual_handoff_sha256 = hashlib.sha256(handoff_bytes).hexdigest()
    if start_training and (
        not _is_sha256(expected_handoff_sha256)
        or expected_handoff_sha256 != actual_handoff_sha256
    ):
        raise ValueError("starting training requires the externally approved handoff SHA-256")
    try:
        raw = json.loads(
            handoff_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("full-run handoff is not valid unambiguous UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("full-run handoff must contain a JSON object")
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
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
            or not _is_sha256(entry.get("sha256"))
        ):
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
    if not isinstance(requirements, dict):
        raise ValueError("handoff requirements must be an object")
    minimum_disk = requirements.get("minimum_disk_free_bytes", 0)
    minimum_gpu = requirements.get("minimum_gpu_memory_mib", 0)
    if not _is_int(minimum_disk) or minimum_disk < 0:
        raise ValueError("minimum disk requirement must be a non-negative integer")
    if not _is_int(minimum_gpu) or minimum_gpu < 0:
        raise ValueError("minimum GPU requirement must be a non-negative integer")
    if disk_free_bytes < minimum_disk:
        raise RuntimeError("available disk space is below the full-run requirement")
    if gpu_memory_mib < minimum_gpu:
        raise RuntimeError("available GPU memory is below the full-run requirement")

    corpus_validation = _load_object(resolved["corpus_validation"])
    if corpus_validation.get("passed") is not True:
        raise ValueError("corpus validation has not passed")
    _validate_evaluation(
        _load_object(resolved["base_evaluation"]),
        label="base evaluation",
        expected_role="base_identity",
        expected_sha256=hashes["base_identity"],
    )
    _validate_evaluation(
        _load_object(resolved["adapter_evaluation"]),
        label="adapter evaluation",
        expected_role="adapter_manifest",
        expected_sha256=hashes["adapter_manifest"],
    )
    smoke = _load_object(resolved["smoke_manifest"])
    completed_training_examples = smoke.get("completed_training_examples")
    if (
        smoke.get("finite_loss") is not True
        or not _is_int(completed_training_examples)
        or completed_training_examples <= 0
    ):
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
    if not training["command"] or not all(
        isinstance(part, str) and part and "\x00" not in part
        for part in training["command"]
    ):
        raise ValueError("handoff training command is empty")
    command = tuple(training["command"])
    command_sha256 = training.get("command_sha256")
    actual_command_sha256 = hashlib.sha256(
        json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not _is_sha256(command_sha256) or command_sha256 != actual_command_sha256:
        raise ValueError("training command identity does not match the handoff")
    estimated_steps = training.get("estimated_steps", 0)
    estimated_disk_bytes = training.get("estimated_disk_bytes", 0)
    if not _is_int(estimated_steps) or estimated_steps < 0:
        raise ValueError("estimated training steps must be a non-negative integer")
    if not _is_int(estimated_disk_bytes) or estimated_disk_bytes < 0:
        raise ValueError("estimated training disk bytes must be a non-negative integer")
    started = False
    if start_training:
        runner = executor or _execute
        exit_code = runner(list(command))
        if exit_code != 0:
            raise RuntimeError(f"full training command failed with exit code {exit_code}")
        started = True
    return FullRunProposal(
        command=command,
        estimated_steps=estimated_steps,
        estimated_disk_bytes=estimated_disk_bytes,
        manifest_hashes=hashes,
        started=started,
    )


def _execute(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def _load_object(path: Path) -> dict[str, object]:
    encoded = path.read_bytes()
    if not encoded or len(encoded) > MAX_ARTIFACT_JSON_BYTES:
        raise ValueError(f"artifact has an invalid byte length: {path}")
    try:
        raw = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"artifact is not valid unambiguous UTF-8 JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return raw


def _validate_evaluation(
    payload: dict[str, object],
    *,
    label: str,
    expected_role: str,
    expected_sha256: str,
) -> None:
    if payload.get("schema_version") != "bcnlt-model-evaluation-v1":
        raise ValueError(f"{label} schema is unsupported")
    evaluated = payload.get("evaluated_artifact")
    if (
        not isinstance(evaluated, dict)
        or evaluated.get("role") != expected_role
        or evaluated.get("sha256") != expected_sha256
    ):
        raise ValueError(f"{label} artifact identity does not match the handoff")
    if not isinstance(payload.get("identity"), str) or not payload["identity"]:
        raise ValueError(f"{label} identity is missing")
    settings = payload.get("settings")
    if (
        not isinstance(settings, dict)
        or not _is_int(settings.get("seed"))
        or not _is_finite_number(settings.get("temperature"))
        or not _is_int(settings.get("max_new_tokens"))
        or int(settings["max_new_tokens"]) <= 0
    ):
        raise ValueError(f"{label} settings are invalid")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{label} results are missing")
    case_ids: list[str] = []
    for result in results:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("case_id"), str)
            or not result["case_id"]
            or not isinstance(result.get("category"), str)
            or not isinstance(result.get("prompt"), str)
            or not isinstance(result.get("expected"), str)
            or not isinstance(result.get("output"), str)
            or type(result.get("passed")) is not bool
            or not (result.get("error") is None or isinstance(result.get("error"), str))
        ):
            raise ValueError(f"{label} results contain an invalid record")
        case_ids.append(result["case_id"])
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{label} results contain duplicate case IDs")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics or any(
        not isinstance(name, str) or not _is_finite_number(value)
        for name, value in metrics.items()
    ):
        raise ValueError(f"{label} metrics are invalid")
    failures = payload.get("runtime_failures")
    if not isinstance(failures, list) or not all(
        isinstance(failure, str) for failure in failures
    ):
        raise ValueError(f"{label} runtime failures are invalid")
    if not _is_finite_number(payload.get("heldout_loss")):
        raise ValueError(f"{label} heldout loss is invalid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or launch the guarded BCNLT full run")
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--disk-free-bytes", required=True, type=int)
    parser.add_argument("--gpu-memory-mib", required=True, type=int)
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--handoff-sha256")
    parser.add_argument("--resume")
    parser.add_argument("--resume-sha256")
    args = parser.parse_args(argv)
    proposal = prepare_full_run(
        args.handoff,
        disk_free_bytes=args.disk_free_bytes,
        gpu_memory_mib=args.gpu_memory_mib,
        start_training=args.start_training,
        expected_handoff_sha256=args.handoff_sha256,
        resume_path=args.resume,
        expected_resume_sha256=args.resume_sha256,
    )
    print(json.dumps(proposal.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
