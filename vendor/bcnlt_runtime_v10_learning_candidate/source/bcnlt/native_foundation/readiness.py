from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import NativeFoundationConfig


@dataclass(frozen=True, slots=True)
class CorpusReadinessReport:
    passed: bool
    ready_domains: tuple[str, ...]
    missing_domains: tuple[str, ...]
    below_threshold: dict[str, tuple[str, ...]]
    failure_codes: tuple[str, ...]


def evaluate_corpus_readiness(
    config: NativeFoundationConfig,
    audit: Mapping[str, Any],
) -> CorpusReadinessReport:
    per_domain = audit.get("per_domain", audit.get("per_language", {}))
    if not isinstance(per_domain, Mapping):
        per_domain = {}

    required = (*config.required_text_domains, *config.canonical_code_languages)
    ready: list[str] = []
    missing: list[str] = []
    below: dict[str, tuple[str, ...]] = {}

    for domain in required:
        metrics = per_domain.get(domain)
        if not isinstance(metrics, Mapping):
            missing.append(domain)
            continue
        failures: list[str] = []
        if _metric(metrics, "document_count") < config.minimum_documents_per_domain:
            failures.append("document_count")
        if _metric(metrics, "source_count", "repository_count") < config.minimum_sources_per_domain:
            failures.append("source_count")
        if _metric(metrics, "payload_units", "payload_base_tokens") < config.minimum_payload_units_per_domain:
            failures.append("payload_units")
        if failures:
            below[domain] = tuple(failures)
        else:
            ready.append(domain)

    failure_codes: list[str] = []
    if missing:
        failure_codes.append("missing_required_domains")
    if below:
        failure_codes.append("domains_below_threshold")
    return CorpusReadinessReport(
        passed=not failure_codes,
        ready_domains=tuple(ready),
        missing_domains=tuple(missing),
        below_threshold=below,
        failure_codes=tuple(failure_codes),
    )


def _metric(metrics: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0
