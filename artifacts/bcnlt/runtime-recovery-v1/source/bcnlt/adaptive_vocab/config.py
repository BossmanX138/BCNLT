from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bcnlt.languages import CANONICAL_LANGUAGE_IDS


SCHEMA_VERSION = "bcnlt-adaptive-vocab-v5"
_SECRET_PATTERN_NAMES = (
    "private_key",
    "aws_access_key",
    "github_token",
    "openai_style_key",
    "jwt",
    "uri_userinfo",
    "hex_24",
    "base64_24",
)


@dataclass(frozen=True, slots=True)
class SecretPattern:
    name: str
    expression: str

    def matches(self, value: str) -> bool:
        return re.search(self.expression, value) is not None


@dataclass(frozen=True, slots=True)
class AdaptiveVocabularyConfig:
    schema_version: str
    canonical_languages: tuple[str, ...]
    minimum_documents_per_language: int
    minimum_repositories_per_language: int
    minimum_payload_tokens_per_language: int
    candidate_min_tokens: int
    candidate_max_tokens: int
    maximum_occurrences_per_document: int
    minimum_supporting_languages: int
    minimum_supporting_repositories: int
    minimum_supporting_documents: int
    maximum_repository_fraction: float
    maximum_document_fraction: float
    minimum_candidate_score: int
    vocabulary_budgets: tuple[int, ...]
    minimum_heldout_compression: float
    maximum_p95_latency_slowdown: float
    minimum_macro_training_occurrences: int
    maximum_training_examples: int
    maximum_optimizer_steps: int
    validation_interval_steps: int
    minimum_secret_run_length: int
    minimum_secret_entropy: float
    random_seed: int
    default_output_root: str
    secret_patterns: tuple[SecretPattern, ...]

    @classmethod
    def load(cls, path: str | Path) -> AdaptiveVocabularyConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("adaptive vocabulary configuration must be an object")
        expected_keys = {
            field
            for field in cls.__dataclass_fields__
        }
        if set(raw) != expected_keys:
            missing = sorted(expected_keys.difference(raw))
            extra = sorted(set(raw).difference(expected_keys))
            raise ValueError(f"adaptive configuration keys mismatch: missing={missing}, extra={extra}")
        patterns_raw = raw["secret_patterns"]
        if not isinstance(patterns_raw, list):
            raise ValueError("secret_patterns must be a list")
        patterns: list[SecretPattern] = []
        for item in patterns_raw:
            if not isinstance(item, dict) or set(item) != {"name", "expression"}:
                raise ValueError("secret pattern entries require name and expression")
            pattern = SecretPattern(str(item["name"]), str(item["expression"]))
            try:
                re.compile(pattern.expression)
            except re.error as exc:
                raise ValueError(f"invalid secret pattern {pattern.name}: {exc}") from exc
            patterns.append(pattern)
        config = cls(
            **{
                key: value
                for key, value in raw.items()
                if key not in {"canonical_languages", "vocabulary_budgets", "secret_patterns"}
            },
            canonical_languages=tuple(raw["canonical_languages"]),
            vocabulary_budgets=tuple(raw["vocabulary_budgets"]),
            secret_patterns=tuple(patterns),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        exact: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "canonical_languages": CANONICAL_LANGUAGE_IDS,
            "minimum_documents_per_language": 100,
            "minimum_repositories_per_language": 3,
            "minimum_payload_tokens_per_language": 100_000,
            "candidate_min_tokens": 2,
            "candidate_max_tokens": 16,
            "maximum_occurrences_per_document": 8,
            "minimum_supporting_languages": 3,
            "minimum_supporting_repositories": 3,
            "minimum_supporting_documents": 5,
            "maximum_repository_fraction": 0.50,
            "maximum_document_fraction": 0.10,
            "minimum_candidate_score": 25,
            "vocabulary_budgets": (256, 512, 1024),
            "minimum_heldout_compression": 0.10,
            "maximum_p95_latency_slowdown": 0.10,
            "minimum_macro_training_occurrences": 8,
            "maximum_training_examples": 10_000,
            "maximum_optimizer_steps": 512,
            "validation_interval_steps": 64,
            "minimum_secret_run_length": 24,
            "minimum_secret_entropy": 3.5,
            "random_seed": 1729,
            "default_output_root": "D:\\BCNLT-runs\\runeforge-mk2-bcnlt-v5-adaptive-vocab",
        }
        for field, expected in exact.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} must equal the production value {expected!r}")
        names = tuple(pattern.name for pattern in self.secret_patterns)
        if names != _SECRET_PATTERN_NAMES:
            raise ValueError(f"secret_patterns must use the exact names {_SECRET_PATTERN_NAMES!r}")
        if len(set(self.vocabulary_budgets)) != len(self.vocabulary_budgets):
            raise ValueError("vocabulary_budgets must not contain duplicates")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonical_languages": list(self.canonical_languages),
            "minimum_documents_per_language": self.minimum_documents_per_language,
            "minimum_repositories_per_language": self.minimum_repositories_per_language,
            "minimum_payload_tokens_per_language": self.minimum_payload_tokens_per_language,
            "candidate_min_tokens": self.candidate_min_tokens,
            "candidate_max_tokens": self.candidate_max_tokens,
            "maximum_occurrences_per_document": self.maximum_occurrences_per_document,
            "minimum_supporting_languages": self.minimum_supporting_languages,
            "minimum_supporting_repositories": self.minimum_supporting_repositories,
            "minimum_supporting_documents": self.minimum_supporting_documents,
            "maximum_repository_fraction": self.maximum_repository_fraction,
            "maximum_document_fraction": self.maximum_document_fraction,
            "minimum_candidate_score": self.minimum_candidate_score,
            "vocabulary_budgets": list(self.vocabulary_budgets),
            "minimum_heldout_compression": self.minimum_heldout_compression,
            "maximum_p95_latency_slowdown": self.maximum_p95_latency_slowdown,
            "minimum_macro_training_occurrences": self.minimum_macro_training_occurrences,
            "maximum_training_examples": self.maximum_training_examples,
            "maximum_optimizer_steps": self.maximum_optimizer_steps,
            "validation_interval_steps": self.validation_interval_steps,
            "minimum_secret_run_length": self.minimum_secret_run_length,
            "minimum_secret_entropy": self.minimum_secret_entropy,
            "random_seed": self.random_seed,
            "default_output_root": self.default_output_root,
            "secret_patterns": [
                {"name": pattern.name, "expression": pattern.expression}
                for pattern in self.secret_patterns
            ],
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
