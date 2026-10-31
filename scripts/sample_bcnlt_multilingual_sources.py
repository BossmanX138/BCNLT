from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from bcnlt.adapters.carousel import BCNLTMultilingualAdapter
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine
from bcnlt.corpus.filters import evaluate_source_document, normalize_license
from bcnlt.corpus.sources import SourceSpec
from bcnlt.languages import CANONICAL_LANGUAGE_IDS, LANGUAGE_BY_ID


RowProvider = Callable[[SourceSpec, int], Iterable[Mapping[str, object]]]
CONTROL_FLOW_EVENTS = frozenset({"BRANCH", "LOOP", "RETURN", "TRY", "EXCEPT", "MATCH"})


@dataclass(frozen=True)
class LanguageSampleReport:
    language: str
    fetched: int
    licensed: int
    parsed: int
    admitted: int
    rejected: int
    rejection_reasons: dict[str, int]
    coverage_gate_failures: dict[str, int]
    top_unknown_node_types: dict[str, int]
    events: tuple[str, ...]
    nested_scope_document_count: int
    passed: bool
    failure_codes: tuple[str, ...]


@dataclass(frozen=True)
class MultilingualSampleReport:
    schema_version: str
    rows_per_language: int
    language_reports: tuple[LanguageSampleReport, ...]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def source_spec_from_entry(raw: Mapping[str, object]) -> SourceSpec:
    return SourceSpec(
        provider=str(raw["provider"]),
        name=str(raw["name"]),
        revision=str(raw["revision"]),
        upstream_url=str(raw["upstream_url"]),
        canonical_language=str(raw["canonical_language"]),
        data_dir=str(raw["data_dir"]) if raw.get("data_dir") else None,
        dialect=str(raw["dialect"]) if raw.get("dialect") else None,
        registry_entry_sha256=str(raw.get("registry_entry_sha256", "")),
        parser_bundle_sha256=str(raw.get("parser_bundle_sha256", "")),
        grammar_revision=str(raw.get("grammar_revision", "")),
        semantic_spec_sha256=str(raw.get("semantic_spec_sha256", "")),
        extraction_rule_version=str(
            raw.get("extraction_rule_version", "bcnlt-multilingual-extraction-v1")
        ),
    )


def sample_languages(
    specs: Iterable[SourceSpec],
    *,
    row_provider: RowProvider,
    adapter: object,
    rows_per_language: int = 100,
    minimum_admitted: int = 25,
) -> MultilingualSampleReport:
    by_language = {spec.canonical_language: spec for spec in specs}
    if tuple(by_language) != CANONICAL_LANGUAGE_IDS:
        raise ValueError("Multilingual source sample requires all 14 languages in canonical order")
    reports: list[LanguageSampleReport] = []
    for language in CANONICAL_LANGUAGE_IDS:
        spec = by_language[language]
        fetched = licensed = parsed = admitted = nested_scope_documents = 0
        reasons: Counter[str] = Counter()
        coverage_gate_failures: Counter[str] = Counter()
        unknown_node_types: Counter[str] = Counter()
        events: set[str] = set()
        for index, row in enumerate(itertools.islice(row_provider(spec, rows_per_language), rows_per_language)):
            fetched += 1
            content = str(row.get("content") or "")
            extension = LANGUAGE_BY_ID[language].accepted_extensions[0]
            relative_path = str(row.get("path") or f"row-{index}{extension}").replace("\\", "/")
            license_value = row.get("license", row.get("max_stars_repo_licenses"))
            if normalize_license(license_value) is not None:
                licensed += 1
            decision = evaluate_source_document(
                relative_path=relative_path,
                content=content,
                license_value=license_value,
                language=language,
                dialect=spec.dialect,
                adapter=adapter,
            )
            if decision.mapped_ratio is not None:
                parsed += 1
                unknown_node_types.update(decision.unknown_node_types)
            if decision.accepted:
                admitted += 1
                normalized_events = tuple(str(event).upper() for event in decision.event_names)
                events.update(normalized_events)
                # Any explicit scope entered beneath the implicit module/global
                # scope proves nested-scope handling for this document.
                if normalized_events.count("SCOPE_ENTER") >= 1:
                    nested_scope_documents += 1
            else:
                reasons[decision.reason_code or "rejected"] += 1
                if decision.reason_code == "semantic_coverage_failure":
                    if decision.mapped_ratio is not None and decision.mapped_ratio < 0.85:
                        coverage_gate_failures["mapped_ratio_below_0_85"] += 1
                    if decision.unknown_ratio is not None and decision.unknown_ratio > 0.10:
                        coverage_gate_failures["unknown_ratio_above_0_10"] += 1
                    if decision.scopes_balanced is False:
                        coverage_gate_failures["scope_imbalance"] += 1
                    if decision.links_valid is False:
                        coverage_gate_failures["invalid_links"] += 1
        failures: list[str] = []
        if fetched < rows_per_language:
            failures.append("insufficient_fetched_rows")
        if admitted < minimum_admitted:
            failures.append("insufficient_admitted_documents")
        for event in ("DEFINE", "CALL", "REFERENCE"):
            if event not in events:
                failures.append(f"missing_{event.lower()}_event")
        if not events.intersection(CONTROL_FLOW_EVENTS):
            failures.append("missing_control_flow_event")
        if nested_scope_documents == 0:
            failures.append("missing_nested_scope_document")
        reports.append(
            LanguageSampleReport(
                language=language,
                fetched=fetched,
                licensed=licensed,
                parsed=parsed,
                admitted=admitted,
                rejected=fetched - admitted,
                rejection_reasons=dict(sorted(reasons.items())),
                coverage_gate_failures=dict(sorted(coverage_gate_failures.items())),
                top_unknown_node_types=dict(unknown_node_types.most_common(20)),
                events=tuple(sorted(events)),
                nested_scope_document_count=nested_scope_documents,
                passed=not failures,
                failure_codes=tuple(failures),
            )
        )
    return MultilingualSampleReport(
        schema_version="bcnlt-multilingual-source-sample-v1",
        rows_per_language=rows_per_language,
        language_reports=tuple(reports),
        passed=all(item.passed for item in reports),
    )


def _hub_rows(spec: SourceSpec, limit: int) -> Iterable[Mapping[str, object]]:
    if sys.platform == "win32":
        try:
            import truststore
        except ImportError:
            pass
        else:
            truststore.inject_into_ssl()
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem, get_token

    token = get_token()
    if not token:
        raise ValueError("Hugging Face authentication is required for the gated Stack sample")
    print(f"Fetching bounded {spec.canonical_language} preview...", file=sys.stderr, flush=True)
    api = HfApi(token=token)
    files = sorted(
        item.path
        for item in api.list_repo_tree(
            spec.name,
            path_in_repo=spec.data_dir or "",
            recursive=False,
            revision=spec.revision,
            repo_type="dataset",
        )
        if item.path.endswith(".parquet")
    )
    if not files:
        raise ValueError(f"No Parquet shards found for {spec.canonical_language}")
    pinned_path = f"datasets/{spec.name}@{spec.revision}/{files[0]}"
    filesystem = HfFileSystem(token=token)
    with filesystem.open(pinned_path, "rb", block_size=8 * 1024 * 1024) as handle:
        parquet = pq.ParquetFile(handle)
        rows: list[dict[str, object]] = []
        accepted_extensions = LANGUAGE_BY_ID[spec.canonical_language].accepted_extensions
        for row_group in range(parquet.num_row_groups):
            for row in parquet.read_row_group(row_group).to_pylist():
                normalized = _normalize_stack_preview_row(row)
                suffix = Path(str(normalized["path"])).suffix.lower()
                if suffix in accepted_extensions:
                    rows.append(normalized)
                    if len(rows) == limit:
                        return tuple(rows)
    return tuple(rows)


def _normalize_stack_preview_row(row: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(row)
    normalized["path"] = row.get("max_stars_repo_path", row.get("path", ""))
    normalized["repository"] = row.get("max_stars_repo_name", row.get("repository", ""))
    normalized["license"] = row.get("max_stars_repo_licenses", row.get("license"))
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample all 14 pinned BCNLT source partitions")
    parser.add_argument("--config", default="configs/bcnlt_corpus_sources_v2.json")
    parser.add_argument("--rows-per-language", type=int, default=100)
    parser.add_argument("--minimum-admitted", type=int, default=25)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    registry = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if registry.get("schema_version") != "bcnlt-corpus-sources-v2":
        parser.error("source sample requires bcnlt-corpus-sources-v2")
    specs = tuple(
        source_spec_from_entry(raw)
        for raw in registry["sources"]
        if raw.get("name") == "bigcode/the-stack-dedup"
    )
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    report = sample_languages(
        specs,
        row_provider=_hub_rows,
        adapter=adapter,
        rows_per_language=args.rows_per_language,
        minimum_admitted=args.minimum_admitted,
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
