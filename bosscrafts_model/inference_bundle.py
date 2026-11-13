"""Immutable inference-only BossCrafts model bundles."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import EngineeringModel
from .tokenizer import NativeTokenizerBinding
from .training_checkpoint import load_model_checkpoint_for_inference

INFERENCE_BUNDLE_SCHEMA = "bosscrafts-inference-bundle-v1"
MODEL_FILE = "model.pt"
MANIFEST_FILE = "manifest.json"
TOKENIZER_DIRECTORY = "tokenizer"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MANIFEST_FIELDS = frozenset(
    {
        "binding_sha256",
        "bundle_sha256",
        "checkpoint_step",
        "epoch_index",
        "model_file",
        "model_file_bytes",
        "model_file_sha256",
        "model_fingerprint",
        "parameter_dtype",
        "schema_version",
        "source_checkpoint_sha256",
        "target_epochs",
        "task_database_identity_sha256",
        "tokenizer_directory",
        "tokenizer_files",
        "tokenizer_sha256",
        "training_plan_sha256",
    }
)
_MODEL_PAYLOAD_FIELDS = frozenset(
    {
        "binding_sha256",
        "checkpoint_step",
        "epoch_index",
        "model_fingerprint",
        "model_state_dict",
        "parameter_dtype",
        "schema_version",
        "source_checkpoint_sha256",
        "target_epochs",
        "task_database_identity_sha256",
        "tokenizer_sha256",
        "training_plan_sha256",
    }
)


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


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate inference bundle manifest key: {key}")
        payload[key] = value
    return payload


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        content = path.read_bytes()
        if len(content) > MAX_MANIFEST_BYTES:
            raise ValueError("inference bundle manifest exceeds size limit")
        manifest = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("inference bundle manifest is invalid") from error
    if type(manifest) is not dict:
        raise ValueError("inference bundle manifest is invalid")
    return manifest


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
    if (
        not _is_sha256(expected_task_database_identity_sha256)
        or not _is_sha256(expected_model_fingerprint)
    ):
        raise ValueError("expected inference bundle lineage is invalid")
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
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if not parameter_dtypes:
        raise ValueError("model has no parameters")
    if len(parameter_dtypes) != 1:
        raise ValueError("model parameters do not have one export dtype")
    parameter_dtype = str(next(iter(parameter_dtypes))).removeprefix("torch.")
    checkpoint_sha256 = _sha256_file(checkpoint)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
    )
    try:
        tokenizer_destination = temporary / TOKENIZER_DIRECTORY
        tokenizer_destination.mkdir()
        tokenizer_files: dict[str, dict[str, object]] = {}
        for source in sorted(binding.bundle_path.iterdir(), key=lambda path: path.name):
            if not source.is_file():
                raise ValueError("tokenizer bundle contains a non-file entry")
            copied = tokenizer_destination / source.name
            shutil.copy2(source, copied)
            _fsync_file(copied)
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
        _fsync_file(model_file)
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
        manifest_path = temporary / MANIFEST_FILE
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
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
    expected_bundle_sha256: str | None = None,
) -> InferenceBundleState:
    if type(model) is not EngineeringModel:
        raise TypeError("model must be an EngineeringModel")
    if (
        (
            expected_task_database_identity_sha256 is not None
            and not _is_sha256(expected_task_database_identity_sha256)
        )
        or not _is_sha256(expected_binding_sha256)
        or not _is_sha256(expected_model_fingerprint)
        or (
            expected_bundle_sha256 is not None
            and not _is_sha256(expected_bundle_sha256)
        )
    ):
        raise ValueError("expected inference bundle lineage is invalid")
    root = Path(bundle_path).resolve()
    if not root.is_dir():
        raise ValueError("inference bundle directory is invalid")
    root_entries = {entry.name for entry in root.iterdir()}
    if root_entries != {MANIFEST_FILE, MODEL_FILE, TOKENIZER_DIRECTORY}:
        raise ValueError("inference bundle root inventory mismatch")
    manifest_path = root / MANIFEST_FILE
    manifest = _load_manifest(manifest_path)
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("inference bundle manifest fields are invalid")
    unsigned_manifest = dict(manifest)
    claimed_bundle_sha256 = unsigned_manifest.pop("bundle_sha256")
    actual_bundle_sha256 = hashlib.sha256(
        _canonical_json(unsigned_manifest).encode("utf-8")
    ).hexdigest()
    if claimed_bundle_sha256 != actual_bundle_sha256:
        raise ValueError("inference bundle manifest SHA-256 mismatch")
    if (
        expected_bundle_sha256 is not None
        and actual_bundle_sha256 != expected_bundle_sha256
    ):
        raise ValueError("inference bundle identity mismatch")
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

    model_file_bytes = manifest.get("model_file_bytes")
    model_file_sha256 = manifest.get("model_file_sha256")
    model_file = root / str(manifest.get("model_file"))
    if (
        manifest.get("model_file") != MODEL_FILE
        or type(model_file_bytes) is not int
        or model_file_bytes < 1
        or not _is_sha256(model_file_sha256)
        or not model_file.is_file()
        or model_file.stat().st_size != model_file_bytes
        or _sha256_file(model_file) != model_file_sha256
    ):
        raise ValueError("inference bundle model file SHA-256 mismatch")
    tokenizer_directory = root / TOKENIZER_DIRECTORY
    contracts = manifest.get("tokenizer_files")
    if type(contracts) is not dict or not contracts:
        raise ValueError("inference bundle tokenizer inventory is invalid")
    tokenizer_entries = (
        tuple(tokenizer_directory.iterdir()) if tokenizer_directory.is_dir() else ()
    )
    actual_names = {path.name for path in tokenizer_entries}
    if any(not path.is_file() for path in tokenizer_entries):
        raise ValueError("inference bundle tokenizer inventory mismatch")
    if actual_names != set(contracts):
        raise ValueError("inference bundle tokenizer inventory mismatch")
    for name, contract in contracts.items():
        path = tokenizer_directory / name
        if (
            not name
            or Path(name).name != name
            or type(contract) is not dict
            or set(contract) != {"bytes", "sha256"}
            or type(contract.get("bytes")) is not int
            or int(contract["bytes"]) < 0
            or not _is_sha256(contract.get("sha256"))
            or path.stat().st_size != contract["bytes"]
            or _sha256_file(path) != contract["sha256"]
        ):
            raise ValueError("inference bundle tokenizer file mismatch")
    binding = NativeTokenizerBinding.load(tokenizer_directory)
    if (
        binding.binding_sha256 != expected_binding_sha256
        or binding.tokenizer_sha256 != manifest["tokenizer_sha256"]
    ):
        raise ValueError("inference bundle tokenizer lineage mismatch")

    try:
        payload = torch.load(model_file, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("inference bundle model payload is unreadable") from error
    if (
        type(payload) is not dict
        or set(payload) != _MODEL_PAYLOAD_FIELDS
        or payload.get("schema_version") != INFERENCE_BUNDLE_SCHEMA
    ):
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
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if (
        len(parameter_dtypes) != 1
        or str(next(iter(parameter_dtypes))).removeprefix("torch.")
        != payload.get("parameter_dtype")
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
