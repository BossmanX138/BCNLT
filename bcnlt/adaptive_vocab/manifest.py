from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from bcnlt.tokenizer_manifest import TokenizerManifest

from .candidates import CandidateDiscoveryReport
from .config import AdaptiveVocabularyConfig


SCHEMA_VERSION = "bcnlt-adaptive-vocab-manifest-v1"
ALLOWED_FIELDS = frozenset(
    {
        "surface",
        "scope",
        "symbol_id",
        "defines",
        "span",
        "ast_relation",
        "ir",
        "recent_history",
        "metadata",
    }
)


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _digest(payload: object) -> str:
    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _is_digest(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True, slots=True)
class MacroToken:
    name: str
    id: int
    expansion: tuple[int, ...]
    allowed_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "id": self.id,
            "expansion": list(self.expansion),
            "allowed_fields": list(self.allowed_fields),
        }


@dataclass(frozen=True, slots=True)
class AdaptiveVocabularyManifest:
    schema_version: str
    v1_manifest_sha256: str
    candidate_report_sha256: str
    config_sha256: str
    lineage_sha256: str
    base_vocab_size: int
    v1_vocab_size: int
    vocab_size: int
    macros: tuple[MacroToken, ...]
    manifest_sha256: str

    @classmethod
    def create(
        cls,
        *,
        v1_manifest_sha256: str,
        candidate_report_sha256: str,
        config_sha256: str,
        lineage_sha256: str,
        base_vocab_size: int,
        v1_vocab_size: int,
        macros: tuple[MacroToken, ...],
    ) -> AdaptiveVocabularyManifest:
        provisional = cls(
            schema_version=SCHEMA_VERSION,
            v1_manifest_sha256=v1_manifest_sha256,
            candidate_report_sha256=candidate_report_sha256,
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            base_vocab_size=base_vocab_size,
            v1_vocab_size=v1_vocab_size,
            vocab_size=v1_vocab_size + len(macros),
            macros=macros,
            manifest_sha256="",
        )
        _validate(provisional, expected_v1=None, validate_digest=False)
        digest = _digest(provisional._body())
        return cls(
            **{
                field: getattr(provisional, field)
                for field in cls.__dataclass_fields__
                if field != "manifest_sha256"
            },
            manifest_sha256=digest,
        )

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "v1_manifest_sha256": self.v1_manifest_sha256,
            "candidate_report_sha256": self.candidate_report_sha256,
            "config_sha256": self.config_sha256,
            "lineage_sha256": self.lineage_sha256,
            "base_vocab_size": self.base_vocab_size,
            "v1_vocab_size": self.v1_vocab_size,
            "vocab_size": self.vocab_size,
            "macros": [macro.to_dict() for macro in self.macros],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "manifest_sha256": self.manifest_sha256}


def _candidate_report_sha256(report: CandidateDiscoveryReport) -> str:
    return _digest(
        {
            "candidates": [item.to_dict() for item in report.candidates],
            "observation_count": report.observation_count,
            "eligible_base_tokens": [list(item) for item in report.eligible_base_tokens],
        }
    )


def build_inventories(
    report: CandidateDiscoveryReport,
    v1_manifest: TokenizerManifest,
    config: AdaptiveVocabularyConfig,
) -> tuple[AdaptiveVocabularyManifest, ...]:
    eligible = report.eligible_candidates
    if len(eligible) < config.vocabulary_budgets[0]:
        raise ValueError("insufficient_eligible_candidates")
    if not _is_digest(report.lineage_sha256):
        raise ValueError("candidate report requires a verified lineage digest")
    candidate_digest = (
        report.candidate_report_sha256
        if _is_digest(report.candidate_report_sha256)
        else _candidate_report_sha256(report)
    )
    manifests = []
    for budget in config.vocabulary_budgets:
        if len(eligible) < budget:
            continue
        macros = tuple(
            MacroToken(
                name=f"<|bcnlt:macro:{index:04d}|>",
                id=v1_manifest.vocab_size + index,
                expansion=candidate.expansion,
                allowed_fields=tuple(
                    sorted(
                        {
                            rate.payload_class
                            for rate in candidate.rates
                            if rate.capped_occurrences > 0
                        }
                    )
                ),
            )
            for index, candidate in enumerate(eligible[:budget])
        )
        manifests.append(
            AdaptiveVocabularyManifest.create(
                v1_manifest_sha256=v1_manifest.manifest_sha256,
                candidate_report_sha256=candidate_digest,
                config_sha256=config.sha256,
                lineage_sha256=report.lineage_sha256,
                base_vocab_size=v1_manifest.base_vocab_size,
                v1_vocab_size=v1_manifest.vocab_size,
                macros=macros,
            )
        )
    return tuple(manifests)


def _validate(
    manifest: AdaptiveVocabularyManifest,
    expected_v1: TokenizerManifest | None,
    *,
    validate_digest: bool = True,
) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported adaptive manifest schema")
    for name in (
        "v1_manifest_sha256",
        "candidate_report_sha256",
        "config_sha256",
        "lineage_sha256",
    ):
        if not _is_digest(getattr(manifest, name)):
            raise ValueError(f"invalid {name} digest")
    if expected_v1 is not None:
        if manifest.v1_manifest_sha256 != expected_v1.manifest_sha256:
            raise ValueError("v1 manifest digest mismatch")
        if (
            manifest.base_vocab_size != expected_v1.base_vocab_size
            or manifest.v1_vocab_size != expected_v1.vocab_size
        ):
            raise ValueError("v1 vocabulary size mismatch")
    ids = [macro.id for macro in manifest.macros]
    names = [macro.name for macro in manifest.macros]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError("macro IDs and names must be unique")
    expected_ids = list(
        range(manifest.v1_vocab_size, manifest.v1_vocab_size + len(manifest.macros))
    )
    if ids != expected_ids:
        raise ValueError("macro IDs must be contiguous after the v1 vocabulary")
    if manifest.vocab_size != manifest.v1_vocab_size + len(manifest.macros):
        raise ValueError("adaptive vocabulary size mismatch")
    for index, macro in enumerate(manifest.macros):
        if macro.name != f"<|bcnlt:macro:{index:04d}|>":
            raise ValueError("macro names must be canonical and rank ordered")
        if len(macro.expansion) < 2:
            raise ValueError("macro expansion requires at least two base tokens")
        if any(
            not isinstance(token_id, int)
            or token_id < 0
            or token_id >= manifest.base_vocab_size
            for token_id in macro.expansion
        ):
            raise ValueError("macro expansion may contain base-token IDs only")
        if (
            not macro.allowed_fields
            or tuple(sorted(set(macro.allowed_fields))) != macro.allowed_fields
            or any(field not in ALLOWED_FIELDS for field in macro.allowed_fields)
        ):
            raise ValueError("macro allowed field is unknown or noncanonical")
    if validate_digest and manifest.manifest_sha256 != _digest(manifest._body()):
        raise ValueError("adaptive manifest digest mismatch")


def write_adaptive_manifest(path: str | Path, manifest: AdaptiveVocabularyManifest) -> None:
    _validate(manifest, expected_v1=None)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical(manifest.to_dict()))


def validate_adaptive_manifest(
    manifest: AdaptiveVocabularyManifest, v1_manifest: TokenizerManifest
) -> None:
    _validate(manifest, expected_v1=v1_manifest)


def load_adaptive_manifest(
    path: str | Path,
    v1_manifest: TokenizerManifest,
    *,
    expected_candidate_report_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_lineage_sha256: str | None = None,
) -> AdaptiveVocabularyManifest:
    source = Path(path)
    raw_bytes = source.read_bytes()
    try:
        payload = json.loads(raw_bytes)
        macros = tuple(
            MacroToken(
                name=str(item["name"]),
                id=int(item["id"]),
                expansion=tuple(item["expansion"]),
                allowed_fields=tuple(item["allowed_fields"]),
            )
            for item in payload["macros"]
        )
        manifest = AdaptiveVocabularyManifest(
            schema_version=str(payload["schema_version"]),
            v1_manifest_sha256=str(payload["v1_manifest_sha256"]),
            candidate_report_sha256=str(payload["candidate_report_sha256"]),
            config_sha256=str(payload["config_sha256"]),
            lineage_sha256=str(payload["lineage_sha256"]),
            base_vocab_size=int(payload["base_vocab_size"]),
            v1_vocab_size=int(payload["v1_vocab_size"]),
            vocab_size=int(payload["vocab_size"]),
            macros=macros,
            manifest_sha256=str(payload["manifest_sha256"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid adaptive manifest: {exc}") from exc
    _validate(manifest, expected_v1=v1_manifest)
    for label, actual, expected in (
        (
            "candidate report",
            manifest.candidate_report_sha256,
            expected_candidate_report_sha256,
        ),
        ("config", manifest.config_sha256, expected_config_sha256),
        ("lineage", manifest.lineage_sha256, expected_lineage_sha256),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"{label} digest mismatch")
    if raw_bytes != _canonical(manifest.to_dict()):
        raise ValueError("adaptive manifest JSON is not canonical")
    return manifest
