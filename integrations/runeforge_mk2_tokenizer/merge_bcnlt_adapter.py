from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from .embedding_extension import ExtendedEmbedding, ExtendedLMHead


@contextmanager
def _temporary_dequantized_merge_base(model: nn.Module, *, enabled: bool):
    if not enabled:
        yield
        return
    model_type = str(
        getattr(getattr(model, "config", None), "model_type", "")
    ).strip().lower()
    if not model_type:
        raise ValueError("dequantized merge base requires a declared model type")
    saving_utils = importlib.import_module("unsloth_zoo.saving_utils")
    attribute = "_DEQUANT_MERGE_BASE_MODEL_TYPES"
    current = getattr(saving_utils, attribute, None)
    if not isinstance(current, frozenset):
        raise RuntimeError("installed Unsloth does not expose the dequant merge-base gate")
    setattr(saving_utils, attribute, current | {model_type})
    try:
        yield
    finally:
        setattr(saving_utils, attribute, current)


def build_merge_source_hashes(
    *,
    base_model_path: str | Path,
    adapter_path: str | Path,
    tokenizer_path: str | Path,
    training_config_path: str | Path,
) -> dict[str, str]:
    base = Path(base_model_path)
    adapter = Path(adapter_path)
    tokenizer = Path(tokenizer_path)
    training_config = Path(training_config_path)
    if not base.is_dir() or not adapter.is_dir() or not tokenizer.is_dir():
        raise FileNotFoundError("merge base, adapter, and tokenizer directories are required")
    required = (
        adapter / "adapter_model.pt",
        adapter / "artifact_manifest.json",
        adapter / "lineage.json",
        tokenizer / "bcnlt_tokenizer_manifest.json",
        training_config,
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("merge source artifact is missing")
    files = [path for path in base.rglob("*") if path.is_file()]
    files.extend(path for path in adapter.rglob("*") if path.is_file())
    files.extend(path for path in tokenizer.rglob("*") if path.is_file())
    files.append(training_config)
    unique = {path.resolve(): path for path in files}
    return {
        str(resolved): _sha256(path)
        for resolved, path in sorted(unique.items(), key=lambda item: str(item[0]))
    }


def validate_merge_destination(
    output_path: str | Path,
    *,
    source_roots: tuple[str | Path, ...],
) -> Path:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"merged output already exists: {output}")
    for source_value in source_roots:
        source = Path(source_value).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError(f"merged output overlaps source artifact: {source}")
    return output


def materialize_extended_tables(model: nn.Module) -> nn.Module:
    embedding = model.get_input_embeddings()  # type: ignore[attr-defined]
    output = model.get_output_embeddings()  # type: ignore[attr-defined]
    if not isinstance(embedding, ExtendedEmbedding) or not isinstance(output, ExtendedLMHead):
        raise ValueError("model has no attached BCNLT extension")
    merged_embedding = nn.Embedding(
        embedding.num_embeddings,
        embedding.embedding_dim,
        device=embedding.base.weight.device,
        dtype=embedding.base.weight.dtype,
    )
    merged_head = nn.Linear(
        output.in_features,
        output.out_features,
        bias=False,
        device=output.base.weight.device,
        dtype=output.base.weight.dtype,
    )
    with torch.no_grad():
        merged_embedding.weight.copy_(
            torch.cat((embedding.base.weight, embedding.extension.weight), dim=0)
        )
        merged_head.weight.copy_(
            torch.cat((output.base.weight, output.extension.weight), dim=0)
        )
    model.set_input_embeddings(merged_embedding)  # type: ignore[attr-defined]
    model.set_output_embeddings(merged_head)  # type: ignore[attr-defined]
    return model


def _tokenizer_vocab_bounds(tokenizer_path: Path) -> tuple[int, int]:
    manifest_path = tokenizer_path / "bcnlt_tokenizer_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_vocab_size = int(payload["base_vocab_size"])
    added_tokens = payload.get("added_tokens")
    if not isinstance(added_tokens, list) or not added_tokens:
        raise ValueError("BCNLT tokenizer manifest must contain added tokens")
    expected_ids = list(range(base_vocab_size, base_vocab_size + len(added_tokens)))
    actual_ids = [int(item["id"]) for item in added_tokens]
    if actual_ids != expected_ids:
        raise ValueError("BCNLT tokenizer IDs are not a contiguous extension")
    return base_vocab_size, base_vocab_size + len(added_tokens)


def _embed_base_tokenizer(
    *,
    tokenizer_path: Path,
    base_tokenizer_path: Path,
    destination: Path,
) -> None:
    payload = json.loads(
        (tokenizer_path / "bcnlt_tokenizer_manifest.json").read_text(encoding="utf-8")
    )
    declared_files = payload.get("base_tokenizer", {}).get("files")
    if not isinstance(declared_files, list) or not declared_files:
        raise ValueError("BCNLT tokenizer manifest has no base tokenizer file identity")
    destination.mkdir()
    for item in declared_files:
        relative = Path(str(item["path"]))
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != str(item["path"])
        ):
            raise ValueError("base tokenizer manifest contains an unsafe path")
        source = base_tokenizer_path / relative
        expected = str(item["sha256"])
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"base tokenizer source hash mismatch: {relative}")
        shutil.copyfile(source, destination / relative)
        if _sha256(destination / relative) != expected:
            raise ValueError(f"embedded base tokenizer hash mismatch: {relative}")


def _capture_extension_rows(
    model: nn.Module,
    *,
    base_vocab_size: int,
    total_vocab_size: int,
) -> dict[str, torch.Tensor]:
    embedding = model.get_input_embeddings()  # type: ignore[attr-defined]
    output = model.get_output_embeddings()  # type: ignore[attr-defined]
    tables = {
        "model.embed_tokens.weight": embedding.weight,
        "lm_head.weight": output.weight,
    }
    extension_rows: dict[str, torch.Tensor] = {}
    for name, weight in tables.items():
        if weight.ndim != 2 or weight.shape[0] != total_vocab_size:
            raise ValueError(
                f"materialized {name} has {weight.shape[0]} rows; "
                f"expected {total_vocab_size}"
            )
        extension_rows[name] = (
            weight[base_vocab_size:total_vocab_size].detach().cpu().contiguous().clone()
        )
    return extension_rows


def _restore_exported_extension_rows(
    checkpoint_path: Path,
    *,
    extension_rows: dict[str, torch.Tensor],
    base_vocab_size: int,
    total_vocab_size: int,
) -> None:
    index_path = checkpoint_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError("merged safetensors index is required for BCNLT repair")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("merged safetensors index has no weight map")

    shard_replacements: dict[str, dict[str, torch.Tensor]] = {}
    added_bytes = 0
    for tensor_name, extension in extension_rows.items():
        shard_name = weight_map.get(tensor_name)
        if not isinstance(shard_name, str):
            raise ValueError(f"merged checkpoint is missing {tensor_name}")
        shard_path = checkpoint_path / shard_name
        tensors = load_file(shard_path, device="cpu")
        saved = tensors.get(tensor_name)
        if saved is None:
            raise ValueError(f"merged shard is missing {tensor_name}")
        if saved.ndim != 2 or saved.shape[1] != extension.shape[1]:
            raise ValueError(f"merged {tensor_name} has incompatible dimensions")
        if saved.shape[0] == base_vocab_size:
            replacement = torch.cat((saved, extension.to(dtype=saved.dtype)), dim=0)
            added_bytes += extension.numel() * saved.element_size()
        elif saved.shape[0] == total_vocab_size:
            expected = extension.to(dtype=saved.dtype)
            if not torch.equal(saved[base_vocab_size:], expected):
                raise ValueError(f"merged {tensor_name} extension rows changed during export")
            replacement = saved
        else:
            raise ValueError(
                f"merged {tensor_name} has {saved.shape[0]} rows; expected "
                f"{base_vocab_size} or {total_vocab_size}"
            )
        shard_replacements.setdefault(shard_name, {})[tensor_name] = replacement
        del tensors

    for shard_name, replacements in shard_replacements.items():
        shard_path = checkpoint_path / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        tensors = load_file(shard_path, device="cpu")
        tensors.update(replacements)
        temporary = shard_path.with_suffix(shard_path.suffix + ".bcnlt.tmp")
        save_file(tensors, temporary, metadata=metadata)
        temporary.replace(shard_path)

    metadata = index.setdefault("metadata", {})
    if "total_size" in metadata:
        metadata["total_size"] = int(metadata["total_size"]) + added_bytes
    temporary_index = index_path.with_suffix(index_path.suffix + ".bcnlt.tmp")
    temporary_index.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_index.replace(index_path)

    for tensor_name in extension_rows:
        shard_path = checkpoint_path / weight_map[tensor_name]
        tensor = load_file(shard_path, device="cpu")[tensor_name]
        if tensor.shape[0] != total_vocab_size:
            raise ValueError(f"BCNLT repair failed for {tensor_name}")


def save_merged_checkpoint(
    model: nn.Module,
    output_path: str | Path,
    *,
    adapter_path: str | Path,
    tokenizer_path: str | Path,
    base_tokenizer_path: str | Path,
    source_hashes: dict[str, str],
    save_method: str = "merged_16bit",
    merge_base: str = "original_16bit",
    release_status: str = "provisional",
    hf_tokenizer=None,
) -> Path:
    output = Path(output_path)
    adapter = Path(adapter_path)
    tokenizer = Path(tokenizer_path)
    base_tokenizer = Path(base_tokenizer_path)
    if save_method not in {"merged_16bit", "merged_4bit"}:
        raise ValueError("BCNLT merged checkpoint must use a full-weight save method")
    if merge_base not in {"original_16bit", "dequantized_4bit"}:
        raise ValueError("unsupported BCNLT merge base")
    if merge_base == "dequantized_4bit" and save_method != "merged_16bit":
        raise ValueError("dequantized merge base requires merged_16bit output")
    if not release_status.strip():
        raise ValueError("merged checkpoint release status is required")
    if output.exists():
        raise FileExistsError(f"merged output already exists: {output}")
    for path_text, expected in source_hashes.items():
        path = Path(path_text)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source hash mismatch before merge: {path}")
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"merge staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    materialize_extended_tables(model)
    base_vocab_size, total_vocab_size = _tokenizer_vocab_bounds(tokenizer)
    extension_rows = _capture_extension_rows(
        model,
        base_vocab_size=base_vocab_size,
        total_vocab_size=total_vocab_size,
    )
    if not hasattr(model, "save_pretrained_merged"):
        raise TypeError("merged model must implement full-weight save_pretrained_merged")
    with _temporary_dequantized_merge_base(
        model,
        enabled=merge_base == "dequantized_4bit",
    ):
        model.save_pretrained_merged(  # type: ignore[attr-defined]
            staging,
            tokenizer=hf_tokenizer,
            save_method=save_method,
            safe_serialization=True,
        )
    _restore_exported_extension_rows(
        staging,
        extension_rows=extension_rows,
        base_vocab_size=base_vocab_size,
        total_vocab_size=total_vocab_size,
    )
    shutil.copytree(tokenizer, staging / "tokenizer")
    _embed_base_tokenizer(
        tokenizer_path=tokenizer,
        base_tokenizer_path=base_tokenizer,
        destination=staging / "base_tokenizer",
    )
    for path_text, expected in source_hashes.items():
        if _sha256(Path(path_text)) != expected:
            raise ValueError(f"source hash changed during merge: {path_text}")
    output_hashes = {
        path.relative_to(staging).as_posix(): _sha256(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    if not any(
        name.endswith((".safetensors", ".bin")) for name in output_hashes
    ):
        raise ValueError("full-weight merge produced no model weight files")
    (staging / "merge_lineage.json").write_text(
        json.dumps(
            {
                "adapter_path": str(adapter.resolve()),
                "release_status": release_status,
                "save_method": save_method,
                "merge_base": merge_base,
                "output_hashes": output_hashes,
                "source_hashes": dict(sorted(source_hashes.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    staging.rename(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge one verified BCNLT adapter into a new Runeforge checkpoint"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--save-method",
        choices=("merged_16bit", "merged_4bit"),
        default="merged_16bit",
    )
    parser.add_argument(
        "--merge-base",
        choices=("original_16bit", "dequantized_4bit"),
        default="original_16bit",
    )
    parser.add_argument("--release-status", default="provisional")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    base_model = Path(args.base_model)
    tokenizer_root = Path(args.tokenizer_artifact)
    adapter_root = Path(args.adapter)
    training_config_path = Path(args.training_config)
    output = validate_merge_destination(
        args.output,
        source_roots=(base_model, tokenizer_root, adapter_root),
    )

    import unsloth  # noqa: F401
    from transformers import AutoTokenizer

    from bcnlt.tokenizer import BCNLTTokenizer
    from .train_bcnlt_adapter import (
        QLoRAConfig,
        TrainingLineage,
        _default_definition,
        hash_artifact,
        load_bcnlt_adapter_state,
        load_qlora_model,
    )

    config_raw = json.loads(training_config_path.read_text(encoding="utf-8"))
    config = QLoRAConfig(
        target_modules=tuple(config_raw["target_modules"]),
        rank=int(config_raw.get("rank", config_raw.get("lora_rank", 16))),
        alpha=int(config_raw.get("alpha", config_raw.get("lora_alpha", 32))),
        dropout=float(config_raw.get("dropout", config_raw.get("lora_dropout", 0.0))),
        maximum_sequence_tokens=int(config_raw.get("maximum_sequence_tokens", 4096)),
        learning_rate=float(config_raw.get("learning_rate", 2e-4)),
        gradient_accumulation_steps=int(config_raw.get("gradient_accumulation_steps", 8)),
    )
    lineage = TrainingLineage(
        **json.loads((adapter_root / "lineage.json").read_text(encoding="utf-8"))
    )
    if hash_artifact(base_model) != lineage.base_model_sha256:
        raise ValueError("merge base model does not match adapter lineage")
    tokenizer_manifest_path = tokenizer_root / "bcnlt_tokenizer_manifest.json"
    if hash_artifact(tokenizer_manifest_path) != lineage.tokenizer_manifest_sha256:
        raise ValueError("merge tokenizer does not match adapter lineage")
    if hash_artifact(training_config_path) != lineage.training_config_sha256:
        raise ValueError("merge training config does not match adapter lineage")

    tokenizer = BCNLTTokenizer.from_pretrained(tokenizer_root, base_model)
    definitions = {
        item.token: _default_definition(item.token)
        for item in tokenizer.manifest.added_tokens
    }
    model, engine = load_qlora_model(
        base_model,
        manifest=tokenizer.manifest,
        definitions=definitions,
        config=config,
        seed=args.seed,
        engine=lineage.engine,
    )
    if engine != lineage.engine:
        raise ValueError("merge engine does not match adapter lineage")
    load_bcnlt_adapter_state(model, adapter_root, expected=lineage)
    source_hashes = build_merge_source_hashes(
        base_model_path=base_model,
        adapter_path=adapter_root,
        tokenizer_path=tokenizer_root,
        training_config_path=training_config_path,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=False,
    )
    merged = save_merged_checkpoint(
        model,
        output,
        adapter_path=adapter_root,
        tokenizer_path=tokenizer_root,
        base_tokenizer_path=base_model,
        source_hashes=source_hashes,
        save_method=args.save_method,
        merge_base=args.merge_base,
        release_status=args.release_status,
        hf_tokenizer=hf_tokenizer,
    )
    print(str(merged))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
