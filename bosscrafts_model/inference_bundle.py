"""Immutable inference-only BossCrafts model bundles."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import torch

from .model import EngineeringModel
from .tokenizer import NativeTokenizerBinding
from .training_checkpoint import load_model_checkpoint_for_inference


INFERENCE_BUNDLE_SCHEMA = "bosscrafts-inference-bundle-v1"
MODEL_FILE = "model.pt"
MANIFEST_FILE = "manifest.json"
TOKENIZER_DIRECTORY = "tokenizer"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class InferenceBundleStatus:
    bundle_path: Path
    bundle_sha256: str
    model_file_sha256: str
    model_file_bytes: int
    checkpoint_step: int
    task_database_identity_sha256: str
    model_fingerprint: str


@dataclass(frozen=True)
class InferenceBundleState:
    bundle_path: Path
    bundle_sha256: str
    checkpoint_step: int
    epoch_index: int
    target_epochs: int
    task_database_identity_sha256: str
    binding_sha256: str
    tokenizer_sha256: str
    model_fingerprint: str
    training_plan_sha256: str
    parameter_dtype: str


def export_inference_bundle(
    checkpoint_path: str | Path,
    bundle_path: str | Path,
    model: EngineeringModel,
    *,
    binding: NativeTokenizerBinding,
    expected_task_database_identity_sha256: str,
    expected_model_fingerprint: str,
) -> InferenceBundleStatus:
    if type(model) is not EngineeringModel:
        raise TypeError("model must be an EngineeringModel")
    if type(binding) is not NativeTokenizerBinding:
        raise TypeError("binding must be a verified NativeTokenizerBinding")
    if model.config.fingerprint() != expected_model_fingerprint:
        raise ValueError("model fingerprint mismatch")
    destination = Path(bundle_path).resolve()
    if destination.exists():
        raise ValueError("inference bundle already exists")
    checkpoint = Path(checkpoint_path).resolve()
    state = load_model_checkpoint_for_inference(
        checkpoint,
        model,
        expected_task_database_identity_sha256=expected_task_database_identity_sha256,
        expected_binding_sha256=binding.binding_sha256,
        expected_model_fingerprint=expected_model_fingerprint,
    )
    if state.epoch_index < state.target_epochs or state.cursor_after_example_id is not None:
        raise ValueError("checkpoint training is incomplete")
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("model has no parameters")
    parameter_dtype = str(parameter.dtype).removeprefix("torch.")
    checkpoint_sha256 = _sha256_file(checkpoint)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ValueError("temporary inference bundle already exists")
    temporary.mkdir()
    try:
        tokenizer_destination = temporary / TOKENIZER_DIRECTORY
        tokenizer_destination.mkdir()
        tokenizer_files: dict[str, dict[str, object]] = {}
        for source in sorted(binding.bundle_path.iterdir(), key=lambda path: path.name):
            if not source.is_file():
                raise ValueError("tokenizer bundle contains a non-file entry")
            copied = tokenizer_destination / source.name
            shutil.copy2(source, copied)
            tokenizer_files[source.name] = {
                "bytes": copied.stat().st_size,
                "sha256": _sha256_file(copied),
            }

        model_payload = {
            "schema_version": INFERENCE_BUNDLE_SCHEMA,
            "checkpoint_step": state.step,
            "epoch_index": state.epoch_index,
            "target_epochs": state.target_epochs,
            "task_database_identity_sha256": state.task_database_identity_sha256,
            "binding_sha256": state.binding_sha256,
            "tokenizer_sha256": binding.tokenizer_sha256,
            "model_fingerprint": state.model_fingerprint,
            "training_plan_sha256": state.training_plan_sha256,
            "parameter_dtype": parameter_dtype,
            "source_checkpoint_sha256": checkpoint_sha256,
            "model_state_dict": model.state_dict(),
        }
        model_file = temporary / MODEL_FILE
        torch.save(model_payload, model_file)
        model_file_sha256 = _sha256_file(model_file)
        model_file_bytes = model_file.stat().st_size
        manifest = {
            "schema_version": INFERENCE_BUNDLE_SCHEMA,
            "binding_sha256": binding.binding_sha256,
            "checkpoint_step": state.step,
            "epoch_index": state.epoch_index,
            "model_file": MODEL_FILE,
            "model_file_bytes": model_file_bytes,
            "model_file_sha256": model_file_sha256,
            "model_fingerprint": state.model_fingerprint,
            "parameter_dtype": parameter_dtype,
            "source_checkpoint_sha256": checkpoint_sha256,
            "target_epochs": state.target_epochs,
            "task_database_identity_sha256": state.task_database_identity_sha256,
            "tokenizer_directory": TOKENIZER_DIRECTORY,
            "tokenizer_files": tokenizer_files,
            "tokenizer_sha256": binding.tokenizer_sha256,
            "training_plan_sha256": state.training_plan_sha256,
        }
        bundle_sha256 = hashlib.sha256(
            _canonical_json(manifest).encode("utf-8")
        ).hexdigest()
        manifest["bundle_sha256"] = bundle_sha256
        (temporary / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return InferenceBundleStatus(
        bundle_path=destination,
        bundle_sha256=bundle_sha256,
        model_file_sha256=model_file_sha256,
        model_file_bytes=model_file_bytes,
        checkpoint_step=state.step,
        task_database_identity_sha256=state.task_database_identity_sha256,
        model_fingerprint=state.model_fingerprint,
    )


def load_inference_bundle(
    bundle_path: str | Path,
    model: EngineeringModel,
    *,
    expected_task_database_identity_sha256: str | None,
    expected_binding_sha256: str,
    expected_model_fingerprint: str,
) -> InferenceBundleState:
    if type(model) is not EngineeringModel:
        raise TypeError("model must be an EngineeringModel")
    root = Path(bundle_path).resolve()
    manifest_path = root / MANIFEST_FILE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("inference bundle manifest is invalid") from error
    if type(manifest) is not dict:
        raise ValueError("inference bundle manifest is invalid")
    claimed_bundle_sha256 = manifest.pop("bundle_sha256", None)
    actual_bundle_sha256 = hashlib.sha256(
        _canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    if claimed_bundle_sha256 != actual_bundle_sha256:
        raise ValueError("inference bundle manifest SHA-256 mismatch")
    if manifest.get("schema_version") != INFERENCE_BUNDLE_SCHEMA:
        raise ValueError("inference bundle schema mismatch")
    if (
        (
            expected_task_database_identity_sha256 is not None
            and manifest.get("task_database_identity_sha256")
            != expected_task_database_identity_sha256
        )
        or manifest.get("binding_sha256") != expected_binding_sha256
        or manifest.get("model_fingerprint") != expected_model_fingerprint
        or model.config.fingerprint() != expected_model_fingerprint
    ):
        raise ValueError("inference bundle lineage mismatch")
    if (
        not _is_sha256(manifest.get("tokenizer_sha256"))
        or not _is_sha256(manifest.get("training_plan_sha256"))
        or not _is_sha256(manifest.get("source_checkpoint_sha256"))
    ):
        raise ValueError("inference bundle manifest digest is invalid")

    model_file = root / str(manifest.get("model_file"))
    if (
        manifest.get("model_file") != MODEL_FILE
        or not model_file.is_file()
        or model_file.stat().st_size != manifest.get("model_file_bytes")
        or _sha256_file(model_file) != manifest.get("model_file_sha256")
    ):
        raise ValueError("inference bundle model file SHA-256 mismatch")
    tokenizer_directory = root / TOKENIZER_DIRECTORY
    contracts = manifest.get("tokenizer_files")
    if type(contracts) is not dict or not contracts:
        raise ValueError("inference bundle tokenizer inventory is invalid")
    actual_names = {
        path.name for path in tokenizer_directory.iterdir() if path.is_file()
    } if tokenizer_directory.is_dir() else set()
    if actual_names != set(contracts):
        raise ValueError("inference bundle tokenizer inventory mismatch")
    for name, contract in contracts.items():
        path = tokenizer_directory / name
        if (
            type(contract) is not dict
            or path.stat().st_size != contract.get("bytes")
            or _sha256_file(path) != contract.get("sha256")
        ):
            raise ValueError("inference bundle tokenizer file mismatch")
    binding = NativeTokenizerBinding.load(tokenizer_directory)
    if (
        binding.binding_sha256 != expected_binding_sha256
        or binding.tokenizer_sha256 != manifest["tokenizer_sha256"]
    ):
        raise ValueError("inference bundle tokenizer lineage mismatch")

    payload = torch.load(model_file, map_location="cpu", weights_only=True)
    if type(payload) is not dict or payload.get("schema_version") != INFERENCE_BUNDLE_SCHEMA:
        raise ValueError("inference bundle model payload is invalid")
    scalar_keys = (
        "checkpoint_step",
        "epoch_index",
        "target_epochs",
        "task_database_identity_sha256",
        "binding_sha256",
        "tokenizer_sha256",
        "model_fingerprint",
        "training_plan_sha256",
        "parameter_dtype",
        "source_checkpoint_sha256",
    )
    if any(payload.get(key) != manifest.get(key) for key in scalar_keys):
        raise ValueError("inference bundle payload lineage mismatch")
    checkpoint_step = payload.get("checkpoint_step")
    epoch_index = payload.get("epoch_index")
    target_epochs = payload.get("target_epochs")
    if (
        type(checkpoint_step) is not int
        or checkpoint_step < 1
        or type(epoch_index) is not int
        or type(target_epochs) is not int
        or target_epochs < 1
        or epoch_index < target_epochs
    ):
        raise ValueError("inference bundle training state is incomplete")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("inference bundle state dictionary is invalid")
    parameter = next(model.parameters(), None)
    if (
        parameter is None
        or str(parameter.dtype).removeprefix("torch.") != payload.get("parameter_dtype")
    ):
        raise ValueError("inference bundle model dtype mismatch")
    model.load_state_dict(state_dict, strict=True)
    return InferenceBundleState(
        bundle_path=root,
        bundle_sha256=actual_bundle_sha256,
        checkpoint_step=checkpoint_step,
        epoch_index=epoch_index,
        target_epochs=target_epochs,
        task_database_identity_sha256=str(payload["task_database_identity_sha256"]),
        binding_sha256=str(payload["binding_sha256"]),
        tokenizer_sha256=str(payload["tokenizer_sha256"]),
        model_fingerprint=str(payload["model_fingerprint"]),
        training_plan_sha256=str(payload["training_plan_sha256"]),
        parameter_dtype=str(payload["parameter_dtype"]),
    )
