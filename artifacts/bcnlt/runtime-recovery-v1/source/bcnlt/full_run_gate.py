from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .languages import CANONICAL_LANGUAGE_IDS
from .multilingual_training import MultilingualTrainingConfig


REQUIRED_ARTIFACT_ROLES = (
    "parser_bundle",
    "semantic_specs",
    "source_registry",
    "tokenizer_manifest",
    "corpus_manifest",
    "corpus_validation",
    "per_language_validation",
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
    estimated_tokens: int
    estimated_disk_bytes: int
    manifest_hashes: dict[str, str]
    started: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "estimated_steps": self.estimated_steps,
            "estimated_tokens": self.estimated_tokens,
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
    if raw.get("schema_version") != "bcnlt-full-run-handoff-v2":
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
    language_validation = _load_object(resolved["per_language_validation"])
    _require_language_reports(
        language_validation,
        "per-language validation",
        minimum_examples=256,
    )
    semantic_specs = _load_object(resolved["semantic_specs"])
    hashes_by_language = semantic_specs.get("semantic_spec_sha256")
    if not isinstance(hashes_by_language, dict) or tuple(hashes_by_language) != CANONICAL_LANGUAGE_IDS:
        raise ValueError("semantic specs must contain all 14 languages in canonical order")
    if any(not _is_sha256(value) for value in hashes_by_language.values()):
        raise ValueError("semantic specs contain an invalid SHA-256 digest")
    source_registry = _load_object(resolved["source_registry"])
    source_languages = {
        item.get("canonical_language")
        for item in source_registry.get("sources", [])
        if isinstance(item, dict)
    }
    if source_languages != set(CANONICAL_LANGUAGE_IDS):
        raise ValueError("source registry does not cover all 14 canonical languages")
    training_config = MultilingualTrainingConfig.load(resolved["training_config"])
    if training_config.run_kind != "full":
        raise ValueError("full-run handoff must reference the full training config")
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
    if smoke.get("finite_loss") is not True or int(
        smoke.get("completed_training_examples", 0)
    ) <= 0:
        raise ValueError("smoke manifest does not prove a finite completed training step")
    if int(smoke.get("optimizer_steps", 0)) != 128:
        raise ValueError("smoke manifest does not prove 128 optimizer steps")
    validation_examples = smoke.get("validation_examples")
    if not isinstance(validation_examples, dict) or any(
        int(validation_examples.get(language, 0)) != 256
        for language in CANONICAL_LANGUAGE_IDS
    ):
        raise ValueError("smoke manifest does not prove 256 validation examples per language")
    promotion = _load_object(resolved["smoke_promotion"])
    if promotion.get("promoted") is not True or promotion.get("failure_codes"):
        raise ValueError("smoke promotion has not passed")

    if resume_path is not None:
        resume = Path(resume_path)
        if not resume.is_file() or not expected_resume_sha256:
            raise ValueError("resume hash and checkpoint are both required")
        if _hash_artifact(resume) != expected_resume_sha256:
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
    if resume_path is not None:
        command = (
            *command,
            "--resume",
            str(Path(resume_path)),
            "--resume-sha256",
            str(expected_resume_sha256),
        )
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
        estimated_tokens=int(training.get("estimated_tokens", 0)),
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


def _hash_artifact(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise ValueError(f"lineage artifact is missing: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"lineage directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_language_reports(
    payload: dict[str, object],
    label: str,
    *,
    minimum_examples: int | None = None,
) -> None:
    reports = payload.get("language_reports")
    if not isinstance(reports, list):
        raise ValueError(f"{label} does not contain language reports")
    by_language = {
        str(item.get("language")): item
        for item in reports
        if isinstance(item, dict) and item.get("language")
    }
    if set(by_language) != set(CANONICAL_LANGUAGE_IDS):
        raise ValueError(f"{label} does not cover all 14 canonical languages")
    for language in CANONICAL_LANGUAGE_IDS:
        report = by_language[language]
        if report.get("passed") is not True:
            raise ValueError(f"{label} failed for {language}")
        if minimum_examples is not None and int(report.get("examples", 0)) < minimum_examples:
            raise ValueError(f"{label} has too few examples for {language}")


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
