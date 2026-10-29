from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


class ExtendedEmbedding(nn.Module):
    def __init__(
        self,
        base: nn.Embedding,
        definitions: Sequence[str],
        *,
        seed: int,
        expected_base_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        if not definitions or any(not definition.strip() for definition in definitions):
            raise ValueError("at least one non-empty token definition is required")
        if expected_base_vocab_size is not None and base.num_embeddings != expected_base_vocab_size:
            raise ValueError("embedding rows do not match the base vocabulary")
        self.base = base
        self.base.requires_grad_(False)
        self.extension = nn.Embedding(
            len(definitions),
            base.embedding_dim,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        _initialize_rows(self.extension.weight, base.weight, definitions, seed, "embedding")
        self.base_vocab_size = base.num_embeddings
        self.num_embeddings = base.num_embeddings + len(definitions)
        self.embedding_dim = base.embedding_dim

    @property
    def weight(self) -> torch.Tensor:
        """Expose the trainable rows for runtimes that inspect embedding gradients."""
        return self.extension.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _validate_ids(input_ids, self.num_embeddings)
        base_mask = input_ids < self.base_vocab_size
        base_ids = input_ids.clamp(max=self.base_vocab_size - 1)
        extension_ids = (input_ids - self.base_vocab_size).clamp(min=0)
        base_values = self.base(base_ids)
        extension_values = self.extension(extension_ids)
        return torch.where(base_mask.unsqueeze(-1), base_values, extension_values)


class ExtendedLMHead(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        definitions: Sequence[str],
        *,
        seed: int,
        expected_base_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        if not definitions or any(not definition.strip() for definition in definitions):
            raise ValueError("at least one non-empty token definition is required")
        if expected_base_vocab_size is not None and base.out_features != expected_base_vocab_size:
            raise ValueError("language-model output rows do not match the base vocabulary")
        self.base = base
        self.base.requires_grad_(False)
        self.extension = nn.Linear(
            base.in_features,
            len(definitions),
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        _initialize_rows(self.extension.weight, base.weight, definitions, seed, "lm_head")
        self.in_features = base.in_features
        self.out_features = base.out_features + len(definitions)

    @property
    def weight(self) -> torch.Tensor:
        """Present the complete vocabulary table to fused causal-LM loss paths."""
        return torch.cat((self.base.weight, self.extension.weight), dim=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.base(hidden_states), self.extension(hidden_states)), dim=-1)


def attach_bcnlt_extension(
    model: nn.Module,
    manifest: Any,
    definitions: Mapping[str, str],
    seed: int,
) -> nn.Module:
    base_vocab_size = getattr(manifest, "base_vocab_size", None)
    added_tokens = tuple(getattr(manifest, "added_tokens", ()))
    vocab_size = getattr(manifest, "vocab_size", None)
    if not isinstance(base_vocab_size, int) or vocab_size != base_vocab_size + len(added_tokens):
        raise ValueError("invalid tokenizer manifest vocabulary dimensions")
    ordered_definitions = []
    for offset, item in enumerate(added_tokens):
        token = getattr(item, "token", None)
        token_id = getattr(item, "id", None)
        if token_id != base_vocab_size + offset or token not in definitions:
            raise ValueError("every added token requires an exact base vocabulary ID and definition")
        ordered_definitions.append(definitions[token])
    if set(definitions) != {item.token for item in added_tokens}:
        raise ValueError("token definitions must exactly match the tokenizer manifest")

    embedding = model.get_input_embeddings()  # type: ignore[attr-defined]
    output = model.get_output_embeddings()  # type: ignore[attr-defined]
    if isinstance(embedding, ExtendedEmbedding) or isinstance(output, ExtendedLMHead):
        raise ValueError("BCNLT extension is already attached")
    if not isinstance(embedding, nn.Embedding) or not isinstance(output, nn.Linear):
        raise TypeError("BCNLT extension requires Embedding input and Linear output modules")
    model.set_input_embeddings(  # type: ignore[attr-defined]
        ExtendedEmbedding(
            embedding,
            ordered_definitions,
            seed=seed,
            expected_base_vocab_size=base_vocab_size,
        )
    )
    model.set_output_embeddings(  # type: ignore[attr-defined]
        ExtendedLMHead(
            output,
            ordered_definitions,
            seed=seed,
            expected_base_vocab_size=base_vocab_size,
        )
    )
    config = getattr(model, "config", None)
    if config is not None:
        config.vocab_size = vocab_size
    return model


def _initialize_rows(
    destination: torch.Tensor,
    base_weight: torch.Tensor,
    definitions: Sequence[str],
    seed: int,
    namespace: str,
) -> None:
    with torch.no_grad():
        mean = base_weight.detach().float().mean().item()
        std = base_weight.detach().float().std().item()
        scale = std if std > 0 else 0.02
        for index, definition in enumerate(definitions):
            material = f"{namespace}\0{seed}\0{index}\0{definition}".encode()
            row_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(row_seed)
            row = torch.empty(destination.shape[1], dtype=torch.float32, device="cpu")
            row.normal_(mean=mean, std=scale, generator=generator)
            destination[index].copy_(row.to(device=destination.device, dtype=destination.dtype))


def _validate_ids(input_ids: torch.Tensor, vocab_size: int) -> None:
    if input_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("input IDs must use an integer tensor dtype")
    if input_ids.numel() and (
        torch.any(input_ids < 0).item() or torch.any(input_ids >= vocab_size).item()
    ):
        raise IndexError("token ID outside extended vocabulary")
