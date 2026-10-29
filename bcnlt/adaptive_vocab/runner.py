from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .lineage import _paths_overlap, _resolve_platform_path


_BASE_PHASES = (
    "lineage",
    "audit",
    "candidates",
    "inventories",
    "tokenizer_eval",
    "entropy",
)
_TRAINING_PHASES = ("paired_data", "paired_training", "model_eval")
_STATUS_FILE = ".phase-result.json"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_hash(path: Path, *, exclude_status: bool = False) -> str:
    if path.is_file():
        return _file_hash(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in sorted(
        (value for value in path.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(path).as_posix(),
    ):
        relative = item.relative_to(path).as_posix()
        if exclude_status and relative == _STATUS_FILE:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_hash(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentInputs:
    output_root: Path
    protected_paths: tuple[Path, ...]
    config_path: Path
    code_revision: str
    phase_commands: tuple[tuple[str, tuple[str, ...]], ...]
    phase_inputs: tuple[tuple[str, tuple[Path, ...]], ...] = ()
    legacy_corpus: bool = False
    start_training: bool = False


@dataclass(frozen=True)
class ExperimentPhase:
    name: str
    command: tuple[str, ...]
    inputs: tuple[Path, ...]
    output: Path
    code_revision: str
    requires_training_gate: bool = False

    def to_dict(self) -> dict[str, object]:
        command_sha256 = hashlib.sha256(
            json.dumps(list(self.command), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "name": self.name,
            "command_executable": self.command[0] if self.command else None,
            "command_sha256": command_sha256,
            "argument_count": len(self.command),
            "inputs": [str(path) for path in self.inputs],
            "output": str(self.output),
            "code_revision": self.code_revision,
            "requires_training_gate": self.requires_training_gate,
        }


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    status: str
    return_code: int
    identity_sha256: str
    input_sha256: str
    output_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "bcnlt-adaptive-phase-result-v1",
            "phase": self.phase,
            "status": self.status,
            "return_code": self.return_code,
            "identity_sha256": self.identity_sha256,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
        }


@dataclass(frozen=True)
class ExperimentRunResult:
    phases: tuple[PhaseResult, ...]
    return_code: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "bcnlt-adaptive-experiment-run-v1",
            "return_code": self.return_code,
            "phases": [phase.to_dict() for phase in self.phases],
        }


def plan_experiment(inputs: ExperimentInputs) -> tuple[ExperimentPhase, ...]:
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", inputs.code_revision):
        raise ValueError("code revision must be an immutable lowercase Git hash")
    config = inputs.config_path.expanduser().resolve(strict=True)
    output = _resolve_platform_path(inputs.output_root)
    for protected in inputs.protected_paths:
        preserved = _resolve_platform_path(protected)
        if _paths_overlap(output, preserved):
            raise ValueError(f"output overlaps preserved artifact: {output} and {preserved}")
    commands = dict(inputs.phase_commands)
    phase_inputs = dict(inputs.phase_inputs)
    names = list(_BASE_PHASES)
    if inputs.legacy_corpus:
        names.insert(1, "preliminary_corpus_rebuild")
    if inputs.start_training:
        names.extend(_TRAINING_PHASES)
    missing = [name for name in names if name not in commands]
    if missing:
        raise ValueError(f"phase commands missing: {missing}")
    phases = []
    for name in names:
        command = tuple(
            argument.replace("{config}", str(config)) for argument in commands[name]
        )
        if not command or any("\x00" in argument for argument in command):
            raise ValueError(f"invalid command argument list for phase {name}")
        phases.append(
            ExperimentPhase(
                name=name,
                command=command,
                inputs=(config, *phase_inputs.get(name, ())),
                output=output / name,
                code_revision=inputs.code_revision,
                requires_training_gate=name in _TRAINING_PHASES,
            )
        )
    return tuple(phases)


def _phase_identity(phase: ExperimentPhase) -> tuple[str, str]:
    input_rows = []
    digest = hashlib.sha256()
    for path in phase.inputs:
        resolved = path.expanduser().resolve(strict=True)
        item_hash = _path_hash(resolved)
        input_rows.append((str(resolved), item_hash))
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
    input_sha256 = digest.hexdigest()
    identity = hashlib.sha256(
        json.dumps(
            {
                "phase": phase.name,
                "command": list(phase.command),
                "inputs": input_rows,
                "output": str(phase.output.expanduser().resolve(strict=False)),
                "code_revision": phase.code_revision,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return identity, input_sha256


def _load_result(path: Path) -> PhaseResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.pop("schema_version", None) != "bcnlt-adaptive-phase-result-v1":
        raise ValueError("phase result has an unsupported schema")
    return PhaseResult(**raw)


def run_phase(phase: ExperimentPhase, *, resume: bool) -> PhaseResult:
    identity, input_sha256 = _phase_identity(phase)
    output = phase.output.expanduser().resolve(strict=False)
    status_path = output / _STATUS_FILE
    if output.exists():
        if not resume:
            raise FileExistsError(f"phase output already exists: {output}")
        if not status_path.is_file():
            raise ValueError("resume output has no atomic phase status")
        recorded = _load_result(status_path)
        if recorded.identity_sha256 != identity or recorded.input_sha256 != input_sha256:
            raise ValueError("resume phase identity does not match current inputs")
        if recorded.output_sha256 != _path_hash(output, exclude_status=True):
            raise ValueError("resume phase output hash does not match recorded output")
        return recorded

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    command = tuple(
        argument.replace("{output}", str(temporary)) for argument in phase.command
    )
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            check=False,
        )
        if completed.returncode not in (0, 2) or not temporary.is_dir():
            shutil.rmtree(temporary, ignore_errors=True)
            return PhaseResult(
                phase.name, "failed", completed.returncode or 1,
                identity, input_sha256, None,
            )
        output_sha256 = _path_hash(temporary)
        result = PhaseResult(
            phase.name,
            "completed" if completed.returncode == 0 else "hard_stop",
            completed.returncode,
            identity,
            input_sha256,
            output_sha256,
        )
        (temporary / _STATUS_FILE).write_text(
            json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.rename(output)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_experiment(
    phases: Sequence[ExperimentPhase],
    *,
    resume: bool,
    start_training: bool,
    tokenizer_selection_path: Path | None,
) -> ExperimentRunResult:
    results: list[PhaseResult] = []
    for phase in phases:
        if phase.requires_training_gate:
            if not start_training:
                raise ValueError("training phase requires the explicit start-training flag")
            if tokenizer_selection_path is None or not tokenizer_selection_path.is_file():
                return ExperimentRunResult(tuple(results), 2)
            selection = json.loads(tokenizer_selection_path.read_text(encoding="utf-8"))
            if not isinstance(selection, dict) or selection.get("training_eligible") is not True:
                return ExperimentRunResult(tuple(results), 2)
        result = run_phase(phase, resume=resume)
        results.append(result)
        if result.status == "hard_stop":
            return ExperimentRunResult(tuple(results), 2)
        if result.status == "failed":
            return ExperimentRunResult(tuple(results), 1)
    return ExperimentRunResult(tuple(results), 0)
