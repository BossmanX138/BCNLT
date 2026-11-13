from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


def token_entropies(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[float | None, ...]:
    if logits.ndim != 3 or labels.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("entropy inputs require [batch, sequence, vocabulary] logits")
    if logits.shape[:2] != labels.shape or labels.shape != attention_mask.shape:
        raise ValueError("entropy logits, labels, and attention masks are misaligned")
    batch, sequence, _ = logits.shape
    result: list[float | None] = []
    with torch.inference_mode():
        values = logits.detach().to(dtype=torch.float32)
        log_probabilities = torch.log_softmax(values, dim=-1)
        probabilities = torch.exp(log_probabilities)
        entropy = -(probabilities * log_probabilities).sum(dim=-1) / torch.log(
            torch.tensor(2.0, dtype=torch.float32, device=values.device)
        )
        finite_rows = torch.isfinite(values).all(dim=-1) & torch.isfinite(entropy)
        for batch_index in range(batch):
            for position in range(sequence):
                if position + 1 >= sequence:
                    result.append(None)
                    continue
                target = int(labels[batch_index, position + 1].item())
                valid = (
                    int(attention_mask[batch_index, position].item()) != 0
                    and int(attention_mask[batch_index, position + 1].item()) != 0
                    and target != -100
                    and bool(finite_rows[batch_index, position].item())
                )
                result.append(
                    float(entropy[batch_index, position].item()) if valid else None
                )
    return tuple(result)


def align_target_boundaries(
    offsets: Sequence[tuple[int, int]], boundary_positions: set[int]
) -> tuple[bool | None, ...]:
    result: list[bool | None] = []
    seen_target_spans: set[tuple[int, int]] = set()
    for position in range(len(offsets)):
        if position + 1 >= len(offsets):
            result.append(None)
            continue
        start, end = offsets[position + 1]
        span = (start, end)
        is_duplicate = span in seen_target_spans
        if end > start:
            seen_target_spans.add(span)
        result.append(
            end > start and not is_duplicate and start in boundary_positions
        )
    return tuple(result)


def align_target_token_boundaries(
    token_count: int, boundary_token_indices: set[int]
) -> tuple[bool | None, ...]:
    return tuple(
        None if position + 1 >= token_count else position + 1 in boundary_token_indices
        for position in range(token_count)
    )


def snapshot_file_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    return {
        str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in paths
    }


def assert_files_unchanged(snapshot: dict[str, str]) -> None:
    for raw_path, expected in snapshot.items():
        path = Path(raw_path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"boundary entropy diagnostic mutated input: {path}")


@dataclass(frozen=True, slots=True)
class BoundaryEntropyRecord:
    model_role: str
    document_id: str
    token_index: int
    entropy_bits: float | None
    target_boundary: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "model_role": self.model_role,
            "document_id": self.document_id,
            "token_index": self.token_index,
            "entropy_bits": self.entropy_bits,
            "target_boundary": self.target_boundary,
        }


@dataclass(frozen=True, slots=True)
class BoundaryEntropyReport:
    records: tuple[BoundaryEntropyRecord, ...]
    output_jsonl: str
    summary_path: str

    def to_dict(self) -> dict[str, object]:
        by_role = {}
        for role in sorted({item.model_role for item in self.records}):
            values = [
                item.entropy_bits
                for item in self.records
                if item.model_role == role and item.entropy_bits is not None
            ]
            by_role[role] = {
                "token_count": len(values),
                "mean_entropy_bits": sum(values) / len(values) if values else None,
            }
        return {
            "schema_version": "bcnlt-boundary-entropy-v1",
            "record_count": len(self.records),
            "models": by_role,
        }


def collect_boundary_report(
    *,
    v4_model,
    base_model,
    v1_tokenizer,
    raw_tokenizer,
    heldout_documents: Sequence[object],
    output: str | Path,
) -> BoundaryEntropyReport:
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "boundary-entropy.jsonl"
    summary_path = output_root / "boundary-entropy-summary.json"
    if jsonl_path.exists() or summary_path.exists():
        raise FileExistsError("boundary entropy output already exists")
    records: list[BoundaryEntropyRecord] = []
    for ordinal, document in enumerate(heldout_documents):
        source = str(getattr(document, "content", document.get("content", "") if isinstance(document, dict) else ""))
        document_id = str(getattr(document, "document_id", document.get("document_id", ordinal) if isinstance(document, dict) else ordinal))
        if not isinstance(document, dict):
            raise ValueError("heldout boundary documents require frozen token annotations")
        semantic_ids = document.get("semantic_token_ids")
        semantic_boundaries = document.get("semantic_boundary_token_indices")
        raw_boundaries = document.get("raw_boundary_positions")
        if (
            not isinstance(semantic_ids, list)
            or not all(isinstance(item, int) for item in semantic_ids)
            or not isinstance(semantic_boundaries, list)
            or not all(isinstance(item, int) for item in semantic_boundaries)
            or not isinstance(raw_boundaries, list)
            or not all(isinstance(item, int) for item in raw_boundaries)
        ):
            raise ValueError("heldout document lacks frozen boundary annotations")
        raw_ids, raw_offsets = raw_tokenizer.encode_with_offsets(source)
        inputs = (
            (
                "v4_adapter_semantic_v1",
                v4_model,
                semantic_ids,
                align_target_token_boundaries(len(semantic_ids), set(semantic_boundaries)),
            ),
            (
                "runeforge_base_raw_code",
                base_model,
                raw_ids,
                align_target_boundaries(raw_offsets, set(raw_boundaries)),
            ),
        )
        for role, model, ids, boundaries in inputs:
            device = next(model.parameters()).device
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention = torch.ones_like(input_ids)
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=attention).logits
            entropies = token_entropies(logits, input_ids, attention)
            for index, (entropy, boundary) in enumerate(zip(entropies, boundaries, strict=True)):
                records.append(
                    BoundaryEntropyRecord(role, document_id, index, entropy, boundary)
                )
    report = BoundaryEntropyReport(tuple(records), str(jsonl_path), str(summary_path))
    jsonl_path.write_text(
        "".join(
            json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            for item in report.records
        ),
        encoding="utf-8",
        newline="\n",
    )
    summary_path.write_text(
        json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
