from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from bcnlt.tokenizer_eval import EvaluationRun, EvaluationSuite, evaluate_model


_STATE_NAMES = ("frozen_v4", "untrained_v5", "v1_control", "trained_v5")


def _target_digest(record_ids: Sequence[str], targets: Sequence[bytes]) -> str:
    if len(record_ids) != len(targets):
        raise ValueError("record IDs and canonical targets differ in length")
    digest = hashlib.sha256()
    for record_id, target in zip(record_ids, targets, strict=True):
        encoded_id = record_id.encode("utf-8")
        digest.update(len(encoded_id).to_bytes(8, "big"))
        digest.update(encoded_id)
        digest.update(len(target).to_bytes(8, "big"))
        digest.update(target)
    return digest.hexdigest()


@dataclass(frozen=True)
class NLLReport:
    total_nll_bits: float
    total_target_bytes: int
    bits_per_byte: float
    mean_nll_bits_per_token: float
    scored_tokens: int
    target_sha256: str
    record_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_nll_bits": self.total_nll_bits,
            "total_target_bytes": self.total_target_bytes,
            "bits_per_byte": self.bits_per_byte,
            "mean_nll_bits_per_token": self.mean_nll_bits_per_token,
            "scored_tokens": self.scored_tokens,
            "target_sha256": self.target_sha256,
            "record_ids": list(self.record_ids),
        }


def evaluate_nll_bits_per_byte(
    model: Any,
    dataset: Sequence[Mapping[str, Any]],
    target_byte_lengths: Sequence[int],
    *,
    maximum_examples: int | None = None,
) -> NLLReport:
    count = len(dataset) if maximum_examples is None else min(len(dataset), maximum_examples)
    if count <= 0:
        raise ValueError("NLL evaluation requires at least one example")
    if len(target_byte_lengths) < count:
        raise ValueError("target byte lengths do not cover the evaluation dataset")
    device = next(model.parameters()).device
    total_bits = 0.0
    scored_tokens = 0
    record_ids: list[str] = []
    targets: list[bytes] = []
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.inference_mode():
            for index in range(count):
                item = dataset[index]
                target = item.get("target_bytes")
                record_id = item.get("record_id")
                expected_length = target_byte_lengths[index]
                if not isinstance(expected_length, int) or expected_length <= 0:
                    raise ValueError("canonical target byte lengths must be positive")
                if not isinstance(target, bytes) or len(target) != expected_length:
                    raise ValueError("target byte length does not match canonical sidecar")
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError("NLL rows require stable record IDs")
                input_ids = torch.tensor(
                    [item["input_ids"]], dtype=torch.long, device=device
                )
                labels = torch.tensor([item["labels"]], dtype=torch.long, device=device)
                attention_mask = torch.tensor(
                    [item.get("attention_mask", [1] * len(item["input_ids"]))],
                    dtype=torch.long,
                    device=device,
                )
                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = output.logits[:, :-1].contiguous()
                shifted_labels = labels[:, 1:].contiguous()
                valid = shifted_labels != -100
                tokens = int(valid.sum().item())
                if tokens:
                    nats = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        shifted_labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    if not torch.isfinite(nats):
                        raise FloatingPointError("non-finite heldout negative log likelihood")
                    total_bits += float(nats.cpu()) / math.log(2.0)
                    scored_tokens += tokens
                record_ids.append(record_id)
                targets.append(target)
    finally:
        model.train(was_training)
    if not scored_tokens:
        raise ValueError("NLL evaluation has no unmasked target tokens")
    total_bytes = sum(target_byte_lengths[:count])
    return NLLReport(
        total_nll_bits=total_bits,
        total_target_bytes=total_bytes,
        bits_per_byte=total_bits / total_bytes,
        mean_nll_bits_per_token=total_bits / scored_tokens,
        scored_tokens=scored_tokens,
        target_sha256=_target_digest(record_ids, targets),
        record_ids=tuple(record_ids),
    )


@dataclass(frozen=True)
class PairedHeldout:
    record_ids: tuple[str, ...]
    target_bytes: tuple[bytes, ...]

    @property
    def target_sha256(self) -> str:
        return _target_digest(self.record_ids, self.target_bytes)


@dataclass(frozen=True)
class AdaptiveModelInput:
    identity: str
    model: Any
    dataset: Sequence[Mapping[str, Any]]
    record_ids: tuple[str, ...]
    target_bytes: tuple[bytes, ...]
    generate: Callable[[str, Any], str]
    semantic_input_tokens: int
    training_losses: tuple[float, ...] = ()
    adapter_reloadable: bool = True
    latency_seconds: float | None = None
    peak_vram_bytes: int | None = None
    tokens_per_second: float | None = None


@dataclass(frozen=True)
class AdaptiveEvaluationConfig:
    minimum_semantic_compression: float = 0.10
    maximum_control_regression: float = 0.02
    minimum_structural_accuracy: float = 0.90
    minimum_required_accuracy: float = 1.0


@dataclass(frozen=True)
class AdaptiveStateReport:
    identity: str
    generation: EvaluationRun
    nll: NLLReport
    semantic_input_tokens: int
    training_losses: tuple[float, ...]
    adapter_reloadable: bool
    latency_seconds: float | None
    peak_vram_bytes: int | None
    tokens_per_second: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "generation": self.generation.to_dict(),
            "nll": self.nll.to_dict(),
            "semantic_input_tokens": self.semantic_input_tokens,
            "training_losses": list(self.training_losses),
            "adapter_reloadable": self.adapter_reloadable,
            "latency_seconds": self.latency_seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass(frozen=True)
class AdaptiveModelEvaluation:
    states: tuple[AdaptiveStateReport, ...]
    paired_target_sha256: str
    semantic_compression: float
    failure_codes: tuple[str, ...]

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(state.identity for state in self.states)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "bcnlt-adaptive-model-evaluation-v1",
            "states": [state.to_dict() for state in self.states],
            "paired_target_sha256": self.paired_target_sha256,
            "semantic_compression": self.semantic_compression,
            "failure_codes": list(self.failure_codes),
            "passed": not self.failure_codes,
        }


@dataclass(frozen=True)
class PromotionProposal:
    eligible: bool
    promoted: bool
    failure_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "bcnlt-adaptive-promotion-proposal-v1",
            "eligible": self.eligible,
            "promoted": False,
            "failure_codes": list(self.failure_codes),
        }


def evaluate_adaptive_matrix(
    frozen_v4: AdaptiveModelInput,
    untrained_v5: AdaptiveModelInput,
    v1_control: AdaptiveModelInput,
    trained_v5: AdaptiveModelInput,
    suite: EvaluationSuite,
    paired_heldout: PairedHeldout,
    config: AdaptiveEvaluationConfig,
) -> AdaptiveModelEvaluation:
    inputs = (frozen_v4, untrained_v5, v1_control, trained_v5)
    failures: set[str] = set()
    if tuple(item.identity for item in inputs) != _STATE_NAMES:
        failures.add("missing_or_misnamed_model_state")
    reports: list[AdaptiveStateReport] = []
    for item in inputs:
        row_targets = tuple(row.get("target_bytes") for row in item.dataset)
        row_ids = tuple(row.get("record_id") for row in item.dataset)
        if (
            item.record_ids != paired_heldout.record_ids
            or item.target_bytes != paired_heldout.target_bytes
            or row_ids != paired_heldout.record_ids
            or row_targets != paired_heldout.target_bytes
        ):
            failures.add("target_hash_mismatch")
        generation = evaluate_model(
            item.identity,
            suite.cases,
            suite.settings,
            item.generate,
            heldout_loss=0.0,
        )
        nll = evaluate_nll_bits_per_byte(
            item.model,
            item.dataset,
            [len(value) for value in row_targets],
        )
        if nll.record_ids != paired_heldout.record_ids:
            failures.add("heldout_order_mismatch")
        if nll.target_sha256 != paired_heldout.target_sha256:
            failures.add("target_hash_mismatch")
        reports.append(
            AdaptiveStateReport(
                item.identity,
                generation,
                nll,
                item.semantic_input_tokens,
                item.training_losses,
                item.adapter_reloadable,
                item.latency_seconds,
                item.peak_vram_bytes,
                item.tokens_per_second,
            )
        )
    by_name = {item.identity: item for item in reports}
    baseline_tokens = frozen_v4.semantic_input_tokens
    if baseline_tokens <= 0 or trained_v5.semantic_input_tokens <= 0:
        failures.add("invalid_semantic_token_count")
        compression = 0.0
    else:
        compression = 1.0 - trained_v5.semantic_input_tokens / baseline_tokens
        if compression + 1e-12 < config.minimum_semantic_compression:
            failures.add("semantic_compression_below_threshold")

    trained = by_name.get("trained_v5")
    frozen = by_name.get("frozen_v4")
    control = by_name.get("v1_control")
    if trained and frozen and trained.nll.bits_per_byte > frozen.nll.bits_per_byte + 1e-12:
        failures.add("bits_per_byte_regression_frozen_v4")
    if trained and control and trained.nll.bits_per_byte > control.nll.bits_per_byte + 1e-12:
        failures.add("bits_per_byte_regression_v1_control")
    if trained:
        metrics = trained.generation.metrics
        if metrics["structural_accuracy"] < config.minimum_structural_accuracy:
            failures.add("structural_accuracy_below_threshold")
        for category in ("tokenizer", "semantic_to_code", "raw_code", "ordinary_instruction"):
            if metrics[f"{category}_accuracy"] < config.minimum_required_accuracy:
                failures.add(f"{category}_accuracy_below_threshold")
        if metrics["exact_match_accuracy"] < 1.0:
            failures.add("exact_output_regression")
        if trained.generation.runtime_failures:
            failures.add("runtime_failure")
        if not trained.adapter_reloadable:
            failures.add("adapter_not_reloadable")
        if any(not math.isfinite(value) for value in trained.training_losses):
            failures.add("non_finite_training_trace")
        if frozen and (
            metrics["parseability_accuracy"] < frozen.generation.metrics["parseability_accuracy"]
            or metrics["ast_equivalence_accuracy"]
            < frozen.generation.metrics["ast_equivalence_accuracy"]
        ):
            failures.add("ast_or_parseability_regression")
    if control and any(not math.isfinite(value) for value in control.training_losses):
        failures.add("non_finite_training_trace")
    return AdaptiveModelEvaluation(
        tuple(reports), paired_heldout.target_sha256, compression, tuple(sorted(failures))
    )


def evaluate_adaptive_promotion(
    report: AdaptiveModelEvaluation, tokenizer_selection: Mapping[str, Any]
) -> PromotionProposal:
    failures = set(report.failure_codes)
    if tokenizer_selection.get("training_eligible") is not True:
        failures.add("tokenizer_selection_not_eligible")
    manifest = tokenizer_selection.get("selected_manifest_sha256")
    if not isinstance(manifest, str) or len(manifest) != 64:
        failures.add("tokenizer_selection_manifest_missing")
    ordered = tuple(sorted(failures))
    return PromotionProposal(eligible=not ordered, promoted=False, failure_codes=ordered)
