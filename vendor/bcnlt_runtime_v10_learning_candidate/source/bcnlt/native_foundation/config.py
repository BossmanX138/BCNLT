from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bcnlt.languages import CANONICAL_LANGUAGE_IDS


SCHEMA_VERSION = "bcnlt-native-foundation-tokenizer-v1"


@dataclass(frozen=True, slots=True)
class NativeFoundationConfig:
    schema_version: str
    primary_vocabulary_size: int
    control_vocabulary_size: int
    tied_embeddings: bool
    required_text_domains: tuple[str, ...]
    canonical_code_languages: tuple[str, ...]
    byte_fallback: bool
    preserve_legacy_artifacts: bool
    minimum_documents_per_domain: int
    minimum_sources_per_domain: int
    minimum_payload_units_per_domain: int
    default_output_root: str

    @classmethod
    def load(cls, path: str | Path) -> NativeFoundationConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("native foundation configuration must be an object")
        expected_keys = set(cls.__dataclass_fields__)
        if set(raw) != expected_keys:
            missing = sorted(expected_keys.difference(raw))
            extra = sorted(set(raw).difference(expected_keys))
            raise ValueError(
                "native foundation configuration keys mismatch: "
                f"missing={missing}, extra={extra}"
            )
        config = cls(
            **{
                key: value
                for key, value in raw.items()
                if key not in {"required_text_domains", "canonical_code_languages"}
            },
            required_text_domains=tuple(raw["required_text_domains"]),
            canonical_code_languages=tuple(raw["canonical_code_languages"]),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        exact = {
            "schema_version": SCHEMA_VERSION,
            "primary_vocabulary_size": 65_536,
            "control_vocabulary_size": 32_768,
            "tied_embeddings": True,
            "required_text_domains": ("english",),
            "canonical_code_languages": CANONICAL_LANGUAGE_IDS,
            "byte_fallback": True,
            "preserve_legacy_artifacts": True,
            "minimum_documents_per_domain": 100,
            "minimum_sources_per_domain": 3,
            "minimum_payload_units_per_domain": 100_000,
            "default_output_root": "D:\\BCNLT-runs\\bcnlt-native-foundation-65536",
        }
        for field, expected in exact.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} must equal the production value {expected!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "primary_vocabulary_size": self.primary_vocabulary_size,
            "control_vocabulary_size": self.control_vocabulary_size,
            "tied_embeddings": self.tied_embeddings,
            "required_text_domains": list(self.required_text_domains),
            "canonical_code_languages": list(self.canonical_code_languages),
            "byte_fallback": self.byte_fallback,
            "preserve_legacy_artifacts": self.preserve_legacy_artifacts,
            "minimum_documents_per_domain": self.minimum_documents_per_domain,
            "minimum_sources_per_domain": self.minimum_sources_per_domain,
            "minimum_payload_units_per_domain": self.minimum_payload_units_per_domain,
            "default_output_root": self.default_output_root,
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
