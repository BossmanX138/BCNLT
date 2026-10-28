from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from integrations.runeforge_mk2_tokenizer.embedding_extension import attach_bcnlt_extension
from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import (
    TrainingLineage,
    save_bcnlt_adapter,
    verify_training_preflight,
)

from test_embedding_extension import DEFINITIONS, Manifest, TinyCausalModel


class TrainableTinyModel(TinyCausalModel):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2, 2))


def _lineage() -> TrainingLineage:
    return TrainingLineage(
        base_model_sha256="1" * 64,
        tokenizer_manifest_sha256="2" * 64,
        corpus_manifest_sha256="3" * 64,
        training_config_sha256="4" * 64,
        engine="transformers-peft-bitsandbytes",
    )


def _model():
    model = TrainableTinyModel()
    attach_bcnlt_extension(model, Manifest(), DEFINITIONS, seed=7)
    return model


def test_preflight_requires_exact_hashes_cuda_extensions_and_lora():
    model = _model()
    report = verify_training_preflight(
        model,
        expected=_lineage(),
        observed=_lineage(),
        cuda_available=True,
    )

    assert report.passed
    assert report.extension_trainable_parameters > 0
    assert report.lora_trainable_parameters > 0
    assert report.frozen_base_parameters > 0

    bad = TrainingLineage(**{**_lineage().__dict__, "corpus_manifest_sha256": "9" * 64})
    with pytest.raises(ValueError, match="corpus_manifest_sha256"):
        verify_training_preflight(
            model, expected=_lineage(), observed=bad, cuda_available=True
        )
    with pytest.raises(RuntimeError, match="CUDA"):
        verify_training_preflight(
            model, expected=_lineage(), observed=_lineage(), cuda_available=False
        )


def test_resume_identity_is_exact_and_whole_tables_cannot_be_trainable():
    model = _model()
    bad_resume = TrainingLineage(
        **{**_lineage().__dict__, "training_config_sha256": "8" * 64}
    )
    with pytest.raises(ValueError, match="resume training_config_sha256"):
        verify_training_preflight(
            model,
            expected=_lineage(),
            observed=_lineage(),
            resume=bad_resume,
            cuda_available=True,
        )

    model.get_input_embeddings().base.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="whole base table"):
        verify_training_preflight(
            model,
            expected=_lineage(),
            observed=_lineage(),
            cuda_available=True,
        )


def test_adapter_save_keeps_sparse_and_lora_weights_with_lineage(tmp_path: Path):
    model = _model()
    tokenizer = tmp_path / "tokenizer-source"
    tokenizer.mkdir()
    (tokenizer / "bcnlt_tokenizer_manifest.json").write_text("{}")
    output = tmp_path / "adapter"

    artifact = save_bcnlt_adapter(model, output, tokenizer, _lineage())

    state = torch.load(artifact.weights_path, weights_only=True)
    assert state
    assert all("extension" in key or "lora_" in key for key in state)
    assert json.loads(artifact.lineage_path.read_text())["base_model_sha256"] == "1" * 64
    assert (output / "tokenizer" / "bcnlt_tokenizer_manifest.json").is_file()
    assert tokenizer.is_dir()
    with pytest.raises(FileExistsError):
        save_bcnlt_adapter(model, output, tokenizer, _lineage())
