from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from integrations.runeforge_mk2_tokenizer.embedding_extension import (
    ExtendedEmbedding,
    ExtendedLMHead,
    attach_bcnlt_extension,
    grow_bcnlt_extension,
)
from bcnlt.adaptive_vocab.manifest import AdaptiveVocabularyManifest, MacroToken
from integrations.runeforge_mk2_tokenizer.merge_bcnlt_adapter import (
    materialize_extended_tables,
)


@dataclass(frozen=True)
class Added:
    token: str
    id: int


class Manifest:
    base_vocab_size = 5
    vocab_size = 7
    added_tokens = (Added("<|bcnlt:a|>", 5), Added("<|bcnlt:b|>", 6))
    manifest_sha256 = "a" * 64


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(9)
        self.embedding = nn.Embedding(5, 4)
        self.lm_head = nn.Linear(4, 5, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value


DEFINITIONS = {
    "<|bcnlt:a|>": "begin a semantic field",
    "<|bcnlt:b|>": "end a semantic field",
}


def test_base_rows_are_exact_and_added_ids_use_only_extension_rows():
    model = TinyCausalModel()
    original_embedding = model.embedding.weight.detach().clone()
    original_head = model.lm_head.weight.detach().clone()
    hidden = torch.randn(2, 4)
    original_logits = model.lm_head(hidden).detach().clone()

    attach_bcnlt_extension(model, Manifest(), DEFINITIONS, seed=17)

    ids = torch.tensor([[0, 4, 5, 6]])
    embedded = model.get_input_embeddings()(ids)
    logits = model.get_output_embeddings()(hidden)
    assert torch.equal(embedded[0, :2], original_embedding[torch.tensor([0, 4])])
    assert torch.equal(logits[:, :5], original_logits)
    assert torch.equal(model.get_input_embeddings().base.weight, original_embedding)
    assert torch.equal(model.get_output_embeddings().base.weight, original_head)
    assert torch.equal(
        embedded[0, 2:], model.get_input_embeddings().extension.weight
    )


def test_initialization_is_deterministic_from_definitions_and_seed():
    first = TinyCausalModel()
    second = TinyCausalModel()
    attach_bcnlt_extension(first, Manifest(), DEFINITIONS, seed=33)
    attach_bcnlt_extension(second, Manifest(), DEFINITIONS, seed=33)

    assert torch.equal(
        first.get_input_embeddings().extension.weight,
        second.get_input_embeddings().extension.weight,
    )
    assert torch.equal(
        first.get_output_embeddings().extension.weight,
        second.get_output_embeddings().extension.weight,
    )

    third = TinyCausalModel()
    attach_bcnlt_extension(third, Manifest(), DEFINITIONS, seed=34)
    assert not torch.equal(
        first.get_input_embeddings().extension.weight,
        third.get_input_embeddings().extension.weight,
    )


def test_gradients_touch_only_extension_rows_and_out_of_range_ids_fail():
    model = TinyCausalModel()
    attach_bcnlt_extension(model, Manifest(), DEFINITIONS, seed=17)
    ids = torch.tensor([[0, 5, 6]])
    loss = model.get_input_embeddings()(ids).sum()
    loss.backward()

    assert model.get_input_embeddings().base.weight.grad is None
    assert model.get_input_embeddings().extension.weight.grad is not None
    assert model.get_output_embeddings().base.weight.requires_grad is False
    with pytest.raises(IndexError, match="outside extended vocabulary"):
        model.get_input_embeddings()(torch.tensor([[7]]))
    with pytest.raises(IndexError, match="outside extended vocabulary"):
        model.get_input_embeddings()(torch.tensor([[-1]]))


def test_unsloth_weight_contract_exposes_trainable_input_and_complete_output():
    model = TinyCausalModel()
    attach_bcnlt_extension(model, Manifest(), DEFINITIONS, seed=17)
    embedding = model.get_input_embeddings()
    output = model.get_output_embeddings()
    hidden = torch.randn(2, 4)

    assert embedding.weight is embedding.extension.weight
    assert embedding.weight.requires_grad
    assert output.weight.shape == (Manifest.vocab_size, 4)
    assert torch.equal(output.weight[: Manifest.base_vocab_size], output.base.weight)
    assert torch.equal(output.weight[Manifest.base_vocab_size :], output.extension.weight)

    torch.nn.functional.linear(hidden, output.weight).sum().backward()
    assert output.base.weight.grad is None
    assert output.extension.weight.grad is not None


def test_wrappers_validate_dimensions_and_manifest_definitions():
    base_embedding = nn.Embedding(5, 4)
    base_head = nn.Linear(4, 5, bias=False)
    with pytest.raises(ValueError, match="definition"):
        ExtendedEmbedding(base_embedding, (), seed=1)
    with pytest.raises(ValueError, match="output rows"):
        ExtendedLMHead(
            nn.Linear(4, 4, bias=False),
            ("a", "b"),
            seed=1,
            expected_base_vocab_size=5,
        )


def test_materialized_tables_preserve_base_rows_and_append_extensions():
    model = TinyCausalModel()
    base_embedding = model.embedding.weight.detach().clone()
    base_head = model.lm_head.weight.detach().clone()
    attach_bcnlt_extension(model, Manifest(), DEFINITIONS, seed=3)
    extension_embedding = model.embedding.extension.weight.detach().clone()
    extension_head = model.lm_head.extension.weight.detach().clone()

    materialize_extended_tables(model)

    assert model.embedding.num_embeddings == 7
    assert model.lm_head.out_features == 7
    assert torch.equal(model.embedding.weight[:5], base_embedding)
    assert torch.equal(model.embedding.weight[5:], extension_embedding)
    assert torch.equal(model.lm_head.weight[:5], base_head)
    assert torch.equal(model.lm_head.weight[5:], extension_head)
    with pytest.raises(ValueError, match="base vocabulary"):
        attach_bcnlt_extension(
            TinyCausalModel(),
            Manifest(),
            {"<|bcnlt:a|>": "a"},
            seed=1,
        )


def test_growth_preserves_all_old_rows_and_initializes_only_appended_macros():
    model = TinyCausalModel()
    attach_bcnlt_extension(model, Manifest(), DEFINITIONS, seed=3)
    old_embedding = torch.cat(
        (model.embedding.base.weight, model.embedding.extension.weight), dim=0
    ).detach().clone()
    old_head = torch.cat(
        (model.lm_head.base.weight, model.lm_head.extension.weight), dim=0
    ).detach().clone()
    adaptive = AdaptiveVocabularyManifest.create(
        v1_manifest_sha256=Manifest.manifest_sha256,
        candidate_report_sha256="b" * 64,
        config_sha256="c" * 64,
        lineage_sha256="d" * 64,
        base_vocab_size=Manifest.base_vocab_size,
        v1_vocab_size=Manifest.vocab_size,
        macros=(MacroToken("<|bcnlt:macro:0000|>", 7, (1, 2), ("surface",)),),
    )

    grow_bcnlt_extension(
        model,
        Manifest(),
        adaptive,
        {"<|bcnlt:macro:0000|>": "common surface sequence"},
        seed=17,
    )

    assert torch.equal(model.embedding.base.weight, old_embedding[:5])
    assert torch.equal(model.lm_head.base.weight, old_head[:5])
    assert torch.equal(model.embedding.extension.weight[:2], old_embedding[5:])
    assert torch.equal(model.lm_head.extension.weight[:2], old_head[5:])
    assert model.embedding.extension.weight.requires_grad
    assert model.lm_head.extension.weight.requires_grad
    assert model.embedding.base.weight.requires_grad is False

    with pytest.raises(ValueError, match="v1 vocabulary|append"):
        grow_bcnlt_extension(
            model,
            Manifest(),
            adaptive,
            {"<|bcnlt:macro:0000|>": "common surface sequence"},
            seed=17,
        )
