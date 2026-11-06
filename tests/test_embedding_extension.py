from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from integrations.runeforge_mk2_tokenizer.embedding_extension import (
    ExtendedEmbedding,
    ExtendedLMHead,
    attach_bcnlt_extension,
    grow_bcnlt_extension,
)
from bcnlt.adaptive_vocab.manifest import AdaptiveVocabularyManifest, MacroToken
from integrations.runeforge_mk2_tokenizer.merge_bcnlt_adapter import (
    _temporary_dequantized_merge_base,
    build_merge_source_hashes,
    materialize_extended_tables,
    save_merged_checkpoint,
    validate_merge_destination,
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


def test_merged_checkpoint_uses_full_weight_export_and_records_derived_hashes(
    tmp_path: Path,
):
    class MergeableModel(TinyCausalModel):
        def __init__(self):
            super().__init__()
            self.export_call = None

        def save_pretrained(self, path):
            raise AssertionError("adapter-style save_pretrained must not be used")

        def save_pretrained_merged(
            self,
            path,
            tokenizer,
            save_method,
            safe_serialization,
        ):
            self.export_call = (tokenizer, save_method, safe_serialization)
            destination = Path(path)
            (destination / "config.json").write_text('{"vocab_size":7}\n')
            save_file(
                {
                    "model.embed_tokens.weight": self.embedding.weight[:5].detach().cpu(),
                    "lm_head.weight": self.lm_head.weight[:5].detach().cpu(),
                },
                destination / "model.safetensors",
            )
            (destination / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 160},
                        "weight_map": {
                            "model.embed_tokens.weight": "model.safetensors",
                            "lm_head.weight": "model.safetensors",
                        },
                    }
                )
            )

    model = MergeableModel()
    attach_bcnlt_extension(model, Manifest(), DEFINITIONS, seed=3)
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    base_tokenizer = tmp_path / "base-tokenizer"
    base_tokenizer.mkdir()
    base_tokenizer_files = {
        "config.json": b'{"model_type":"tiny"}\n',
        "tokenizer.json": b'{"version":"1.0"}\n',
        "tokenizer_config.json": b'{"tokenizer_class":"Tiny"}\n',
    }
    for name, content in base_tokenizer_files.items():
        (base_tokenizer / name).write_bytes(content)
    (tokenizer / "bcnlt_tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "base_vocab_size": 5,
                "base_tokenizer": {
                    "files": [
                        {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
                        for name, content in sorted(base_tokenizer_files.items())
                    ]
                },
                "added_tokens": [
                    {"token": "<|bcnlt:a|>", "id": 5},
                    {"token": "<|bcnlt:b|>", "id": 6},
                ],
            }
        )
        + "\n"
    )
    output = tmp_path / "merged"

    save_merged_checkpoint(
        model,
        output,
        adapter_path=adapter,
        tokenizer_path=tokenizer,
        base_tokenizer_path=base_tokenizer,
        source_hashes={str(adapter): hashlib.sha256(b"adapter").hexdigest()},
        save_method="merged_16bit",
        release_status="provisional",
    )

    assert model.export_call == (None, "merged_16bit", True)
    assert (output / "tokenizer" / "bcnlt_tokenizer_manifest.json").is_file()
    assert {
        path.name: path.read_bytes() for path in (output / "base_tokenizer").iterdir()
    } == base_tokenizer_files
    tensors = load_file(output / "model.safetensors")
    assert tensors["model.embed_tokens.weight"].shape == (7, 4)
    assert tensors["lm_head.weight"].shape == (7, 4)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 224
    lineage = json.loads((output / "merge_lineage.json").read_text())
    assert lineage["release_status"] == "provisional"
    assert lineage["save_method"] == "merged_16bit"
    assert lineage["merge_base"] == "original_16bit"
    assert lineage["output_hashes"]["model.safetensors"] == hashlib.sha256(
        (output / "model.safetensors").read_bytes()
    ).hexdigest()


def test_merge_source_hashes_cover_base_adapter_tokenizer_and_config(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_bytes(b"base-config")
    (base / "model.safetensors").write_bytes(b"base-weights")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.pt").write_bytes(b"adapter-weights")
    (adapter / "artifact_manifest.json").write_bytes(b"adapter-manifest")
    (adapter / "lineage.json").write_bytes(b"adapter-lineage")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "bcnlt_tokenizer_manifest.json").write_bytes(b"tokenizer")
    config = tmp_path / "training.json"
    config.write_bytes(b"training-config")

    hashes = build_merge_source_hashes(
        base_model_path=base,
        adapter_path=adapter,
        tokenizer_path=tokenizer,
        training_config_path=config,
    )

    assert set(hashes) == {
        str((base / "config.json").resolve()),
        str((base / "model.safetensors").resolve()),
        str((adapter / "adapter_model.pt").resolve()),
        str((adapter / "artifact_manifest.json").resolve()),
        str((adapter / "lineage.json").resolve()),
        str((tokenizer / "bcnlt_tokenizer_manifest.json").resolve()),
        str(config.resolve()),
    }
    assert hashes[str((adapter / "adapter_model.pt").resolve())] == hashlib.sha256(
        b"adapter-weights"
    ).hexdigest()


def test_merge_destination_must_be_new_and_outside_every_source(tmp_path: Path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    tokenizer = tmp_path / "tokenizer"
    for path in (base, adapter, tokenizer):
        path.mkdir()

    assert validate_merge_destination(
        tmp_path / "merged",
        source_roots=(base, adapter, tokenizer),
    ) == (tmp_path / "merged").resolve()
    with pytest.raises(ValueError, match="overlaps"):
        validate_merge_destination(base / "merged", source_roots=(base, adapter, tokenizer))
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        validate_merge_destination(existing, source_roots=(base, adapter, tokenizer))


def test_dequantized_merge_base_override_is_explicit_and_scoped(monkeypatch):
    saving_utils = SimpleNamespace(
        _DEQUANT_MERGE_BASE_MODEL_TYPES=frozenset({"falcon_h1"})
    )
    monkeypatch.setitem(sys.modules, "unsloth_zoo.saving_utils", saving_utils)
    model = SimpleNamespace(config=SimpleNamespace(model_type="mistral"))

    with _temporary_dequantized_merge_base(model, enabled=True):
        assert saving_utils._DEQUANT_MERGE_BASE_MODEL_TYPES == frozenset(
            {"falcon_h1", "mistral"}
        )

    assert saving_utils._DEQUANT_MERGE_BASE_MODEL_TYPES == frozenset({"falcon_h1"})


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
