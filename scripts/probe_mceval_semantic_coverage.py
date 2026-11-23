"""Probe every pinned McEval source against the current BCNLT semantic runtime.

This is diagnostic evidence only.  It does not create training examples or
qualified calibration labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.bcnlt_source_pipeline import (  # noqa: E402
    PARSER_RUNTIME_IDENTITY_SHA256,
    BCNLTPreprocessingError,
    BCNLTSourcePipeline,
)
from bosscrafts_model.category_case_sources import (  # noqa: E402
    MCEVAL_LANGUAGE_FILES,
    _mceval_harness,
    _read_mceval_manifest,
    _reject_duplicate_keys,
)
from bosscrafts_model.deterministic_semantic_audit import (  # noqa: E402
    DeterministicSemanticAuditor,
)
from bosscrafts_model.immutable_artifacts import publish_new_files  # noqa: E402
from bosscrafts_model.jsonl import split_jsonl_records  # noqa: E402
from bosscrafts_model.raw_ast_witness import RawAstWitnessClient  # noqa: E402
from bosscrafts_model.resumable_training_semantic_audit import (  # noqa: E402
    ResumableTrainingSemanticAudit,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding  # noqa: E402

PROBE_SCHEMA = "bosscrafts-mceval-semantic-coverage-probe-v1"
ROW_SCHEMA = "bosscrafts-mceval-semantic-coverage-row-v1"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _row_int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if type(value) is not int:
        raise TypeError(f"probe row {key} must be an integer")
    return value


def _row_counts(row: dict[str, object]) -> dict[str, int]:
    value = row.get("finding_code_counts")
    if type(value) is not dict or any(
        type(key) is not str or type(count) is not int for key, count in value.items()
    ):
        raise TypeError("probe row finding_code_counts is invalid")
    return {str(key): int(count) for key, count in value.items()}


def _load_language_tasks(
    source_root: Path,
    language: str,
    filename: str,
    expected_sha256: str,
) -> list[dict[str, object]]:
    content = (source_root / filename).read_bytes()
    if _sha256_bytes(content) != expected_sha256:
        raise ValueError(f"McEval {language} source SHA-256 mismatch")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        if type(row) is not dict:
            raise ValueError(f"McEval {language} line {line_number} is invalid")
        task_id = row.get("task_id")
        prompt = row.get("prompt")
        solution = row.get("canonical_solution")
        tests = row.get("test")
        if any(
            type(value) is not str or not value.strip()
            for value in (task_id, prompt, solution, tests)
        ):
            raise ValueError(f"McEval {language} line {line_number} is incomplete")
        prefix, suffix = _mceval_harness(language, str(prompt), str(tests))
        rows.append(
            {
                "task_id": task_id,
                "source": prefix + str(solution) + suffix,
                "relative_path": f"mceval/{language}/{task_id}",
                "source_line": line_number,
            }
        )
    return rows


def _probe_language(
    language: str,
    tasks: list[dict[str, object]],
) -> list[dict[str, object]]:
    binding = NativeTokenizerBinding.load()
    pipeline = BCNLTSourcePipeline(binding, persistent_sidecars=True)
    auditor = DeterministicSemanticAuditor()
    witness = RawAstWitnessClient()
    output: list[dict[str, object]] = []
    try:
        for task in tasks:
            task_id = str(task["task_id"])
            source = str(task["source"])
            relative_path = str(task["relative_path"])
            source_sha256 = _sha256_bytes(source.encode("utf-8"))
            try:
                prepared = pipeline.prepare_semantic(
                    source,
                    language=language,
                    path=relative_path,
                    allow_raw_fallback=False,
                )
            except (BCNLTPreprocessingError, ValueError) as exc:
                fallback = pipeline.prepare_semantic(
                    source,
                    language=language,
                    path=relative_path,
                    allow_raw_fallback=True,
                )
                if fallback.mode != "raw_code" or not fallback.used_raw_fallback:
                    raise RuntimeError("semantic rejection did not enter raw fallback")
                output.append(
                    {
                        "schema_version": ROW_SCHEMA,
                        "task_id": task_id,
                        "language": language,
                        "source_line": task["source_line"],
                        "source_sha256": source_sha256,
                        "status": "preprocessing_rejected",
                        "semantic_record_count": 0,
                        "failed_record_count": 0,
                        "finding_code_counts": {},
                        "detail": " ".join(str(exc).split())[:512],
                        "fallback_mode": fallback.mode,
                        "fallback_token_count": len(fallback.token_ids),
                        "fallback_reason": fallback.fallback_reason,
                    }
                )
                continue
            if prepared.canonical_jsonl is None:
                raise RuntimeError("semantic preparation omitted canonical JSONL")
            records = [
                json.loads(line)
                for line in split_jsonl_records(prepared.canonical_jsonl)
            ]
            queries: list[dict[str, object]] = []
            finding_codes: Counter[str] = Counter()
            failed_indices: set[int] = set()
            deterministic_evidence: list[dict[str, object]] = []
            for index, record in enumerate(records):
                review_id = f"{task_id}:{index}"
                item = {
                    "review_id": review_id,
                    "canonical_record": record,
                    "source_excerpt": ResumableTrainingSemanticAudit._source_excerpt(
                        source, relative_path, record
                    ),
                }
                result = auditor.audit_item(item)
                codes = [finding.code for finding in result.findings]
                deterministic_evidence.append(
                    {
                        "review_id": review_id,
                        "extracted_surface": result.extracted_surface,
                        "finding_codes": codes,
                        "passed": result.passed,
                    }
                )
                if codes:
                    failed_indices.add(index)
                    finding_codes.update(codes)
                queries.append(
                    {
                        "query_id": review_id,
                        "span": record.get("span"),
                        "ast_relation": record.get("ast_relation"),
                        "scope": record.get("scope"),
                        "surface": record.get("surface"),
                    }
                )
            raw_evidence: list[dict[str, object]] = []
            if queries:
                response = witness.inspect(
                    source,
                    language=language,
                    path=relative_path,
                    dialect=None,
                    queries=tuple(queries),
                )
                witnesses = (
                    response.get("witnesses") if type(response) is dict else None
                )
                if (
                    type(response) is not dict
                    or response.get("source_sha256") != source_sha256
                    or type(witnesses) is not list
                    or len(witnesses) != len(records)
                    or [row.get("query_id") for row in witnesses]
                    != [query["query_id"] for query in queries]
                ):
                    raise ValueError("raw AST witness coverage or provenance mismatch")
                raw_evidence = witnesses
                for index, row in enumerate(witnesses):
                    codes = []
                    if row.get("relation_span_match") is not True:
                        codes.append("ast_relation_span_mismatch")
                    if row.get("scope_match") is False:
                        codes.append("raw_scope_mismatch")
                    if row.get("surface_match") is False:
                        codes.append("raw_surface_mismatch")
                    if codes:
                        failed_indices.add(index)
                        finding_codes.update(codes)
            output.append(
                {
                    "schema_version": ROW_SCHEMA,
                    "task_id": task_id,
                    "language": language,
                    "source_line": task["source_line"],
                    "source_sha256": source_sha256,
                    "status": "audited",
                    "semantic_record_count": len(records),
                    "failed_record_count": len(failed_indices),
                    "finding_code_counts": dict(sorted(finding_codes.items())),
                    "deterministic_evidence_sha256": _sha256_bytes(
                        "".join(
                            f"{_canonical(row)}\n" for row in deterministic_evidence
                        ).encode("utf-8")
                    ),
                    "raw_ast_evidence_sha256": _sha256_bytes(
                        "".join(f"{_canonical(row)}\n" for row in raw_evidence).encode(
                            "utf-8"
                        )
                    ),
                }
            )
    finally:
        witness.close()
        pipeline.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    output = arguments.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output already exists and is not empty")
    source_root = arguments.source_root.resolve()
    manifest, source_manifest_sha256 = _read_mceval_manifest(arguments.source_manifest)
    language_manifest = manifest["languages"]
    if type(language_manifest) is not dict:
        raise AssertionError("validated McEval language manifest is invalid")
    tasks_by_language: dict[str, list[dict[str, object]]] = {}
    for language, filename in MCEVAL_LANGUAGE_FILES.items():
        entry = language_manifest[language]
        if type(entry) is not dict:
            raise AssertionError("validated McEval file entry is invalid")
        tasks_by_language[language] = _load_language_tasks(
            source_root,
            language,
            filename,
            str(entry["sha256"]),
        )

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(_probe_language, language, tasks): language
            for language, tasks in tasks_by_language.items()
        }
        for future in as_completed(futures):
            language = futures[future]
            language_rows = future.result()
            rows.extend(language_rows)
            failed = sum(_row_int(row, "failed_record_count") for row in language_rows)
            rejected = sum(
                row["status"] == "preprocessing_rejected" for row in language_rows
            )
            print(
                f"{language}: documents={len(language_rows)} "
                f"failed_records={failed} source_rejects={rejected}",
                flush=True,
            )

    rows.sort(key=lambda row: (str(row["language"]), str(row["task_id"])))
    rows_text = "".join(f"{_canonical(row)}\n" for row in rows)
    per_language: dict[str, dict[str, Any]] = {}
    finding_counts: Counter[str] = Counter()
    for language in sorted(MCEVAL_LANGUAGE_FILES):
        language_rows = [row for row in rows if row["language"] == language]
        language_findings: Counter[str] = Counter()
        for row in language_rows:
            language_findings.update(_row_counts(row))
        finding_counts.update(language_findings)
        per_language[language] = {
            "documents": len(language_rows),
            "audited_documents": sum(
                row["status"] == "audited" for row in language_rows
            ),
            "source_rejects": sum(
                row["status"] == "preprocessing_rejected" for row in language_rows
            ),
            "raw_fallback_successes": sum(
                row["status"] == "preprocessing_rejected"
                and row.get("fallback_mode") == "raw_code"
                for row in language_rows
            ),
            "semantic_records": sum(
                _row_int(row, "semantic_record_count") for row in language_rows
            ),
            "failed_records": sum(
                _row_int(row, "failed_record_count") for row in language_rows
            ),
            "finding_code_counts": dict(sorted(language_findings.items())),
        }
    unsigned = {
        "schema_version": PROBE_SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "source_revision": manifest["revision"],
        "source_license": manifest["license"],
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "selection_policy": "all pinned McEval generation tasks in all 14 BCNLT languages",
        "document_count": len(rows),
        "audited_document_count": sum(row["status"] == "audited" for row in rows),
        "source_reject_count": sum(
            row["status"] == "preprocessing_rejected" for row in rows
        ),
        "raw_fallback_success_count": sum(
            row["status"] == "preprocessing_rejected"
            and row.get("fallback_mode") == "raw_code"
            for row in rows
        ),
        "hybrid_coverage_count": sum(
            row["status"] == "audited" or row.get("fallback_mode") == "raw_code"
            for row in rows
        ),
        "hybrid_coverage_complete": all(
            row["status"] == "audited" or row.get("fallback_mode") == "raw_code"
            for row in rows
        ),
        "semantic_record_count": sum(
            _row_int(row, "semantic_record_count") for row in rows
        ),
        "failed_record_count": sum(
            _row_int(row, "failed_record_count") for row in rows
        ),
        "finding_code_counts": dict(sorted(finding_counts.items())),
        "per_language": per_language,
        "rows_file": "observations.jsonl",
        "rows_sha256": _sha256_bytes(rows_text.encode("utf-8")),
        "diagnostic_complete": True,
        "calibration_ready": False,
        "training_ready": False,
        "blockers": [
            "diagnostic probe findings are not qualified calibration labels",
            "evaluation-only CC-BY-SA sources are excluded from training",
        ],
    }
    manifest_sha256 = _sha256_bytes(_canonical(unsigned).encode("utf-8"))
    probe_manifest = {**unsigned, "manifest_sha256": manifest_sha256}
    publish_new_files(
        (
            (output / "observations.jsonl", rows_text.encode("utf-8")),
            (
                output / "manifest.json",
                f"{_canonical(probe_manifest)}\n".encode("utf-8"),
            ),
        ),
        artifact_label="McEval semantic coverage probe",
    )
    print(json.dumps(probe_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
