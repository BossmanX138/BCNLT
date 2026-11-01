from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .languages import CANONICAL_LANGUAGE_IDS


TRAINING_FAMILIES = (
    "semantic_continuation",
    "semantic_to_code",
    "code_to_semantic",
    "structural_reasoning",
    "mixed_instruction",
    "knowledge_boundary",
)
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class MultilingualTrainingConfig:
    schema_version: str
    run_kind: str
    languages: tuple[str, ...]
    families: tuple[str, ...]
    seed: int
    microbatch_size: int
    gradient_accumulation_steps: int
    maximum_sequence_tokens: int
    learning_rate: float
    optimizer: str
    epochs: int
    warmup_ratio: float
    scheduler: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    validation_interval_optimizer_steps: int
    checkpoint_interval_optimizer_steps: int
    maximum_optimizer_steps: int | None
    validation_examples_per_language: int
    minimum_global_improvement: float = 0.02
    maximum_language_loss_ratio: float = 1.25

    def __post_init__(self) -> None:
        if self.schema_version != "bcnlt-multilingual-training-v1":
            raise ValueError("unsupported multilingual training config schema")
        if self.run_kind not in {"smoke", "full"}:
            raise ValueError("run_kind must be smoke or full")
        if self.languages != CANONICAL_LANGUAGE_IDS:
            raise ValueError("training config must list all 14 canonical languages in order")
        if self.families != TRAINING_FAMILIES:
            raise ValueError("training config must list all six curriculum families in order")
        exact = {
            "seed": (self.seed, 1729),
            "microbatch_size": (self.microbatch_size, 1),
            "gradient_accumulation_steps": (self.gradient_accumulation_steps, 8),
            "maximum_sequence_tokens": (self.maximum_sequence_tokens, 4096),
            "optimizer": (self.optimizer, "adamw"),
            "epochs": (self.epochs, 1),
            "scheduler": (self.scheduler, "cosine"),
            "lora_rank": (self.lora_rank, 16),
            "lora_alpha": (self.lora_alpha, 32),
            "lora_dropout": (self.lora_dropout, 0.0),
            "target_modules": (self.target_modules, LORA_TARGET_MODULES),
        }
        for name, (actual, expected) in exact.items():
            if actual != expected:
                raise ValueError(f"{name} must equal the authoritative value {expected!r}")
        if self.learning_rate != 2e-4 or self.warmup_ratio != 0.03:
            raise ValueError("learning_rate and warmup_ratio must equal 2e-4 and 0.03")
        if self.validation_examples_per_language != 256:
            raise ValueError("validation_examples_per_language must equal 256")
        if self.run_kind == "smoke" and self.maximum_optimizer_steps != 128:
            raise ValueError("smoke training must use exactly 128 optimizer steps")
        if self.run_kind == "full" and self.maximum_optimizer_steps is not None:
            raise ValueError("full training is bounded by one epoch, not a smoke step limit")
        if self.run_kind == "full" and (
            self.validation_interval_optimizer_steps != 500
            or self.checkpoint_interval_optimizer_steps != 500
        ):
            raise ValueError("full training validation and checkpoints must run every 500 steps")
        if self.validation_interval_optimizer_steps <= 0:
            raise ValueError("validation interval must be positive")
        if self.checkpoint_interval_optimizer_steps <= 0:
            raise ValueError("checkpoint interval must be positive")
        if not 0.0 < self.minimum_global_improvement < 1.0:
            raise ValueError("minimum_global_improvement must be between zero and one")
        if self.maximum_language_loss_ratio < 1.0:
            raise ValueError("maximum_language_loss_ratio must be at least one")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MultilingualTrainingConfig":
        values = dict(raw)
        for field in ("languages", "families", "target_modules"):
            values[field] = tuple(values[field])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> "MultilingualTrainingConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("multilingual training config must contain a JSON object")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LanguageBalancedSampler:
    """Deterministic near-equal round-robin over language/family cells."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        languages: tuple[str, ...] = CANONICAL_LANGUAGE_IDS,
        families: tuple[str, ...] = TRAINING_FAMILIES,
        seed: int = 1729,
    ) -> None:
        self.records = records
        self.languages = languages
        self.families = families
        self.seed = seed
        buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index in range(len(records)):
            metadata = getattr(records, "metadata", None)
            record = metadata(index) if callable(metadata) else records[index]
            language = str(record.get("language") or "")
            family = str(record.get("family") or "")
            if language in languages and family in families:
                buckets[(language, family)].append(index)
        missing = [
            f"{language}/{family}"
            for language in languages
            for family in families
            if not buckets[(language, family)]
        ]
        if missing:
            raise ValueError("balanced sampler is missing cells: " + ", ".join(missing))
        self._buckets = {key: tuple(value) for key, value in buckets.items()}

    def sample_indices(self, total_examples: int) -> tuple[int, ...]:
        if total_examples <= 0:
            raise ValueError("total_examples must be positive")
        randomizer = random.Random(self.seed)
        language_order = list(self.languages)
        family_order = list(self.families)
        randomizer.shuffle(language_order)
        randomizer.shuffle(family_order)
        available = [
            (language, family) for language in language_order for family in family_order
        ]
        priority = {cell: index for index, cell in enumerate(available)}
        language_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        cells: list[tuple[str, str]] = []
        while available:
            cell = min(
                available,
                key=lambda item: (
                    language_counts[item[0]],
                    family_counts[item[1]],
                    priority[item],
                ),
            )
            available.remove(cell)
            cells.append(cell)
            language_counts[cell[0]] += 1
            family_counts[cell[1]] += 1
        positions = {cell: 0 for cell in cells}
        shuffled = {cell: list(self._buckets[cell]) for cell in cells}
        for values in shuffled.values():
            randomizer.shuffle(values)
        selected: list[int] = []
        while len(selected) < total_examples:
            for cell in cells:
                values = shuffled[cell]
                position = positions[cell]
                if position and position % len(values) == 0:
                    randomizer.shuffle(values)
                selected.append(values[position % len(values)])
                positions[cell] = position + 1
                if len(selected) == total_examples:
                    break
        return tuple(selected)

    def balance_counts(self, indices: Sequence[int]) -> dict[str, dict[str, int]]:
        language_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        for index in indices:
            row = self.records[index]
            language_counts[str(row["language"])] += 1
            family_counts[str(row["family"])] += 1
        return {
            "languages": dict(sorted(language_counts.items())),
            "families": dict(sorted(family_counts.items())),
        }


class IndexedDatasetView(Sequence[Mapping[str, Any]]):
    def __init__(self, dataset: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.dataset[self.indices[index]]


@dataclass(frozen=True)
class SmokeMetrics:
    optimizer_steps: int
    global_step_losses: tuple[float, ...]
    training_losses: tuple[float, ...]
    initial_language_losses: dict[str, float]
    final_language_losses: dict[str, float]
    validation_examples: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SmokeGateDecision:
    promoted: bool
    failure_codes: tuple[str, ...]
    initial_global_mean: float | None
    final_global_mean: float | None
    global_improvement: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_smoke_gate(
    metrics: SmokeMetrics,
    config: MultilingualTrainingConfig,
) -> SmokeGateDecision:
    if config.run_kind != "smoke":
        raise ValueError("smoke metrics require a smoke training config")
    failures: list[str] = []
    if metrics.optimizer_steps != 128:
        failures.append("optimizer_step_count")
    all_losses = (*metrics.training_losses, *metrics.global_step_losses)
    if not all_losses or any(not math.isfinite(value) for value in all_losses):
        failures.append("non_finite_loss")
    initial_mean = final_mean = improvement = None
    if len(metrics.global_step_losses) < 64:
        failures.append("global_loss_window_count")
    else:
        initial_mean = sum(metrics.global_step_losses[:32]) / 32
        final_mean = sum(metrics.global_step_losses[-32:]) / 32
        improvement = (initial_mean - final_mean) / initial_mean if initial_mean > 0 else None
        if improvement is None or improvement < config.minimum_global_improvement:
            failures.append("global_loss_improvement_below_threshold")
    expected = set(config.languages)
    observed = set(metrics.initial_language_losses) & set(metrics.final_language_losses)
    if observed != expected:
        failures.append("missing_language_metrics")
    for language in config.languages:
        if metrics.validation_examples.get(language) != config.validation_examples_per_language:
            failures.append(f"validation_examples:{language}")
        initial = metrics.initial_language_losses.get(language)
        final = metrics.final_language_losses.get(language)
        if initial is not None and final is not None:
            if not math.isfinite(initial) or not math.isfinite(final):
                failures.append(f"non_finite_language_loss:{language}")
            elif final > initial * config.maximum_language_loss_ratio:
                failures.append(f"language_loss_regression:{language}")
    failures = list(dict.fromkeys(failures))
    return SmokeGateDecision(
        promoted=not failures,
        failure_codes=tuple(failures),
        initial_global_mean=initial_mean,
        final_global_mean=final_mean,
        global_improvement=improvement,
    )


def optimizer_step_means(
    losses: Sequence[float], gradient_accumulation_steps: int
) -> tuple[float, ...]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    result = []
    for start in range(0, len(losses), gradient_accumulation_steps):
        window = losses[start : start + gradient_accumulation_steps]
        if window:
            result.append(sum(window) / len(window))
    return tuple(result)


def audit_packed_multilingual_dataset(
    root: str | Path,
    config: MultilingualTrainingConfig,
) -> dict[str, object]:
    packed_root = Path(root)
    manifest = json.loads(
        (packed_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    train_languages: Counter[str] = Counter()
    train_families: Counter[str] = Counter()
    train_cells: Counter[tuple[str, str]] = Counter()
    validation_languages: Counter[str] = Counter()
    for shard in manifest.get("shards", []):
        split = str(shard["split"])
        records_path = packed_root / str(shard["records_path"])
        for line in records_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            language = str(row["language"])
            family = str(row["family"])
            if split == "train":
                train_languages[language] += 1
                train_families[family] += 1
                train_cells[(language, family)] += 1
            elif split == "validation":
                validation_languages[language] += 1
    failures: list[str] = []
    for language in config.languages:
        for family in config.families:
            if train_cells[(language, family)] == 0:
                failures.append(f"missing_training_cell:{language}:{family}")
        if validation_languages[language] < config.validation_examples_per_language:
            failures.append(f"insufficient_validation:{language}")
    if set(train_languages) != set(config.languages):
        failures.append("training_language_coverage")
    if set(train_families) != set(config.families):
        failures.append("training_family_coverage")
    if config.run_kind == "full":
        total_records = sum(train_languages.values())
        token_count = sum(
            int(shard.get("token_count", 0))
            for shard in manifest.get("shards", [])
            if shard.get("split") == "train"
        )
        if total_records < 140_000:
            failures.append("insufficient_training_records")
        if token_count < 50_000_000:
            failures.append("insufficient_training_tokens")
        for language in config.languages:
            count = train_languages[language]
            if count < 10_000:
                failures.append(f"insufficient_training_records:{language}")
            if total_records and count / total_records > 0.12:
                failures.append(f"language_fraction_above_limit:{language}")
    reports = [
        {
            "language": language,
            "passed": (
                validation_languages[language]
                >= config.validation_examples_per_language
                and all(train_cells[(language, family)] > 0 for family in config.families)
            ),
            "training_records": train_languages[language],
            "examples": validation_languages[language],
            "family_counts": {
                family: train_cells[(language, family)] for family in config.families
            },
        }
        for language in config.languages
    ]
    return {
        "schema_version": "bcnlt-multilingual-packed-audit-v1",
        "run_kind": config.run_kind,
        "passed": not failures,
        "failure_codes": failures,
        "language_reports": reports,
        "training_records": sum(train_languages.values()),
        "training_tokens": sum(
            int(shard.get("token_count", 0))
            for shard in manifest.get("shards", [])
            if shard.get("split") == "train"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a packed multilingual training corpus")
    parser.add_argument("--packed", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_packed_multilingual_dataset(
        args.packed,
        MultilingualTrainingConfig.load(args.config),
    )
    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
