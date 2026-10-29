from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from bcnlt.adaptive_vocab.boundary_entropy import (
    align_target_boundaries,
    assert_files_unchanged,
    snapshot_file_hashes,
    token_entropies,
    collect_boundary_report,
)


def test_uniform_logits_equal_log2_vocabulary_and_one_hot_approaches_zero():
    uniform = torch.zeros((1, 3, 4), dtype=torch.float32)
    labels = torch.tensor([[0, 1, 2]])
    attention = torch.ones_like(labels)
    peaked = uniform.clone()
    peaked[0, 0, 1] = 100.0

    uniform_values = token_entropies(uniform, labels, attention)
    peaked_values = token_entropies(peaked, labels, attention)

    assert uniform_values == pytest.approx((2.0, 2.0, None))
    assert peaked_values[0] == pytest.approx(0.0, abs=1e-6)


def test_alignment_uses_next_target_and_masks_padding_ignore_and_nonfinite():
    logits = torch.zeros((2, 4, 3), dtype=torch.float32)
    labels = torch.tensor([[0, 1, -100, 2], [0, 2, 1, 0]])
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    logits[1, 0, 2] = float("nan")

    values = token_entropies(logits, labels, attention)

    assert len(values) == 8
    assert values[0] == pytest.approx(torch.log2(torch.tensor(3.0)).item())
    assert values[1] is None
    assert values[2] is None
    assert values[3] is None
    assert values[4] is None
    assert values[5] is None
    assert values[6] is None
    assert values[7] is None


def test_boundary_flags_attach_to_target_token_not_predictor():
    offsets = [(0, 1), (1, 3), (3, 4), (0, 0)]

    flags = align_target_boundaries(offsets, {1, 3})

    assert flags == (True, True, False, None)


def test_diagnostic_snapshot_helpers_detect_no_mutation(tmp_path: Path):
    paths = []
    for name in ("learned.jsonl", "corpus.sqlite", "tokenizer.json", "candidates.jsonl"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths.append(path)
    before = snapshot_file_hashes(paths)

    (tmp_path / "boundary-entropy.jsonl").write_text("{}\n", encoding="utf-8")

    assert_files_unchanged(before)
    assert before == {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def test_collector_uses_frozen_token_streams_and_emits_target_boundaries(tmp_path: Path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, *, input_ids, attention_mask):
            return type("Output", (), {"logits": torch.zeros((*input_ids.shape, 4))})()

    class SemanticTokenizer:
        def encode_source(self, source):
            raise AssertionError("stateful preprocessing must not run")

    class RawTokenizer:
        def encode_with_offsets(self, text):
            return [0, 1, 2], [(0, 1), (1, 2), (2, 3)]

    report = collect_boundary_report(
        v4_model=Model(),
        base_model=Model(),
        v1_tokenizer=SemanticTokenizer(),
        raw_tokenizer=RawTokenizer(),
        heldout_documents=(
            {
                "document_id": "doc",
                "content": "abc",
                "semantic_token_ids": [0, 1, 2],
                "semantic_boundary_token_indices": [1],
                "raw_boundary_positions": [1],
            },
        ),
        output=tmp_path / "entropy",
    )

    assert report.records
    assert any(item.target_boundary is True for item in report.records)
    assert all(item.target_boundary is not None for item in report.records if item.token_index < 2)


def test_entropy_cli_loads_the_custom_sparse_adapter_format():
    source = Path("scripts/run_bcnlt_boundary_entropy.py").read_text(encoding="utf-8")

    assert "load_bcnlt_adapter_state" in source
    assert "load_qlora_model" in source
    assert "PeftModel.from_pretrained" not in source
    assert 'parser.add_argument("--training-config"' in source
