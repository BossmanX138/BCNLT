from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from torch import nn

from .embedding_extension import ExtendedEmbedding, ExtendedLMHead


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


def save_merged_checkpoint(
    model: nn.Module,
    output_path: str | Path,
    *,
    adapter_path: str | Path,
    tokenizer_path: str | Path,
    source_hashes: dict[str, str],
) -> Path:
    output = Path(output_path)
    adapter = Path(adapter_path)
    tokenizer = Path(tokenizer_path)
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
    if not hasattr(model, "save_pretrained"):
        raise TypeError("merged model must implement save_pretrained")
    model.save_pretrained(staging)  # type: ignore[attr-defined]
    shutil.copytree(tokenizer, staging / "tokenizer")
    for path_text, expected in source_hashes.items():
        if _sha256(Path(path_text)) != expected:
            raise ValueError(f"source hash changed during merge: {path_text}")
    (staging / "merge_lineage.json").write_text(
        json.dumps(
            {
                "adapter_path": str(adapter.resolve()),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
