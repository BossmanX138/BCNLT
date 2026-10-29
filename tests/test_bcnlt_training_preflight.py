from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from integrations.runeforge_mk2_tokenizer.embedding_extension import (
    ExtendedLMHead,
    attach_bcnlt_extension,
)
from integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter import (
    TrainingLineage,
    backward_split_causal_loss,
    evaluate_packed_loss,
    load_bcnlt_adapter_state,
    save_bcnlt_adapter,
    verify_training_preflight,
    QLoRAConfig,
    run_bounded_training_trace,
    run_matched_continuation,
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


def test_adapter_load_requires_exact_hash_and_restores_every_trainable_tensor(tmp_path: Path):
    tokenizer = tmp_path / "tokenizer-source"
    tokenizer.mkdir()
    (tokenizer / "bcnlt_tokenizer_manifest.json").write_text("{}")
    source = _model()
    output = tmp_path / "adapter"
    save_bcnlt_adapter(source, output, tokenizer, _lineage())
    restored = _model()
    with torch.no_grad():
        restored.lora_A.zero_()
        restored.get_input_embeddings().extension.weight.zero_()
        restored.get_output_embeddings().extension.weight.zero_()

    lineage = load_bcnlt_adapter_state(restored, output, expected=_lineage())

    assert lineage == _lineage()
    assert torch.equal(restored.lora_A, source.lora_A)
    assert torch.equal(
        restored.get_input_embeddings().extension.weight,
        source.get_input_embeddings().extension.weight,
    )
    assert torch.equal(
        restored.get_output_embeddings().extension.weight,
        source.get_output_embeddings().extension.weight,
    )

    with (output / "adapter_model.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_bcnlt_adapter_state(_model(), output, expected=_lineage())


def test_packed_loss_is_measured_from_model_outputs_with_a_fixed_limit():
    class LossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, *, input_ids, labels, attention_mask, use_cache):
            assert labels.shape == input_ids.shape == attention_mask.shape
            assert use_cache is False
            return SimpleNamespace(loss=input_ids.float().mean() + self.anchor * 0)

    dataset = [
        {"input_ids": [2, 2], "labels": [2, 2], "attention_mask": [1, 1]},
        {"input_ids": [4, 4], "labels": [4, 4], "attention_mask": [1, 1]},
    ]

    assert evaluate_packed_loss(LossModel(), dataset) == pytest.approx(3.0)
    assert evaluate_packed_loss(LossModel(), dataset, maximum_examples=1) == pytest.approx(2.0)


def test_split_head_chunked_loss_matches_dense_cross_entropy_gradients():
    torch.manual_seed(23)
    reference_head = ExtendedLMHead(
        nn.Linear(4, 5, bias=False), ("first", "second"), seed=7
    )
    chunked_head = ExtendedLMHead(
        nn.Linear(4, 5, bias=False), ("first", "second"), seed=7
    )
    chunked_head.load_state_dict(reference_head.state_dict())
    reference_hidden = torch.randn(1, 6, 4, requires_grad=True)
    chunked_hidden = reference_hidden.detach().clone().requires_grad_(True)
    labels = torch.tensor([[-100, -100, 5, 6, 2, 1]])

    logits = reference_head(reference_hidden[:, :-1])
    reference_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    reference_loss.backward()

    measured = backward_split_causal_loss(
        chunked_hidden,
        labels,
        chunked_head,
        gradient_scale=1.0,
        chunk_tokens=2,
    )

    assert measured == pytest.approx(float(reference_loss.detach()), rel=1e-6)
    assert torch.allclose(chunked_hidden.grad, reference_hidden.grad, atol=1e-6)
    assert torch.allclose(
        chunked_head.extension.weight.grad,
        reference_head.extension.weight.grad,
        atol=1e-6,
    )
    assert chunked_head.base.weight.grad is None


def test_bounded_trace_stops_at_step_limit_validates_and_checks_resume_identity():
    class LoopModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8)

        def get_output_embeddings(self):
            return self.head

        def forward(self, *, input_ids, labels, attention_mask, use_cache):
            logits = self.head(self.embedding(input_ids))
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1)
            )
            return SimpleNamespace(loss=loss)

    dataset = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1]}
        for _ in range(5)
    ]
    validations = []
    trace = run_bounded_training_trace(
        LoopModel(),
        dataset,
        QLoRAConfig(target_modules=("head",), gradient_accumulation_steps=1),
        maximum_optimizer_steps=2,
        maximum_epochs=1,
        validation_interval=1,
        validation_callback=lambda step: validations.append(step),
        run_identity="same",
        resume_identity="same",
        require_cuda=False,
    )

    assert trace.optimizer_steps == 2
    assert trace.completed_epochs <= 1
    assert trace.validation_steps == (1, 2)
    assert validations == [1, 2]
    assert all(torch.isfinite(torch.tensor(trace.losses)))

    with pytest.raises(ValueError, match="resume identity"):
        run_bounded_training_trace(
            LoopModel(),
            dataset,
            QLoRAConfig(target_modules=("head",), gradient_accumulation_steps=1),
            maximum_optimizer_steps=1,
            maximum_epochs=1,
            validation_interval=64,
            run_identity="new",
            resume_identity="old",
            require_cuda=False,
        )


def test_matched_continuation_uses_identical_order_and_shared_step_limit():
    class LoopModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8)

        def get_output_embeddings(self):
            return self.head

        def forward(self, *, input_ids, labels, attention_mask, use_cache):
            logits = self.head(self.embedding(input_ids))
            return SimpleNamespace(
                loss=F.cross_entropy(logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1))
            )

    rows = [
        {"record_id": f"r{i}", "input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1]}
        for i in range(3)
    ]
    left, right = run_matched_continuation(
        LoopModel(), LoopModel(), rows, list(rows),
        QLoRAConfig(target_modules=("head",), gradient_accumulation_steps=1),
        maximum_optimizer_steps=2, maximum_epochs=1, validation_interval=1,
        run_identity="pair", require_cuda=False,
    )
    assert left.optimizer_steps == right.optimizer_steps == 2
    assert left.validation_steps == right.validation_steps == (1, 2)

    mismatched = [dict(rows[0], record_id="different"), *rows[1:]]
    with pytest.raises(ValueError, match="record order"):
        run_matched_continuation(
            LoopModel(), LoopModel(), rows, mismatched,
            QLoRAConfig(target_modules=("head",), gradient_accumulation_steps=1),
            maximum_optimizer_steps=1, run_identity="pair", require_cuda=False,
        )


def test_production_pair_driver_loads_v4_twice_and_uses_the_matched_loop():
    source = Path("scripts/train_bcnlt_adaptive_pair.py").read_text(encoding="utf-8")

    assert source.count("load_bcnlt_adapter_state(") == 2
    assert "run_matched_continuation(" in source
    assert "grow_bcnlt_extension(" in source
