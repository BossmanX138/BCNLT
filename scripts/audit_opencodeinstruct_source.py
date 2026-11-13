"""Audit official NVIDIA OpenCodeInstruct Parquet shards before admission."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.codex_code_curriculum import extract_final_code


SCHEMA = "bosscrafts-opencodeinstruct-source-audit-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(value: object) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0.0 <= score <= 1.0 else None


def _statuses(value: object) -> list[str] | None:
    if isinstance(value, list):
        return [str(item).lower() for item in value]
    if not isinstance(value, str):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    return [str(item).lower() for item in parsed]


def audit(paths: list[Path]) -> dict[str, object]:
    language_counts: Counter[str] = Counter()
    score_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    algorithm_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    supported_code = 0
    rows = 0
    manifests: list[dict[str, object]] = []
    for path in paths:
        resolved = path.resolve()
        parquet = pq.ParquetFile(resolved)
        manifests.append(
            {
                "path": str(resolved),
                "rows": parquet.metadata.num_rows,
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
        required = {
            "id",
            "input",
            "output",
            "domain",
            "generation_algorithm",
            "unit_tests",
            "tests_execution_status",
            "average_test_score",
        }
        columns = set(parquet.schema.names)
        if not required.issubset(columns):
            raise ValueError(f"OpenCodeInstruct shard schema mismatch: {resolved}")
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=2048):
            payload = batch.to_pydict()
            for index in range(batch.num_rows):
                rows += 1
                identifier = str(payload["id"][index])
                if identifier in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(identifier)
                domain_counts[str(payload["domain"][index])] += 1
                algorithm_counts[str(payload["generation_algorithm"][index])] += 1
                extracted = extract_final_code(str(payload["output"][index]))
                if extracted is None:
                    language_counts["unsupported_or_unfenced"] += 1
                else:
                    supported_code += 1
                    language_counts[extracted.language] += 1
                score = _score(payload["average_test_score"][index])
                if score is None:
                    score_counts["invalid"] += 1
                elif score == 1.0:
                    score_counts["1.0"] += 1
                elif score == 0.0:
                    score_counts["0.0"] += 1
                else:
                    score_counts["partial"] += 1
                statuses = _statuses(payload["tests_execution_status"][index])
                if not statuses:
                    status_counts["invalid_or_empty"] += 1
                elif all(status == "pass" for status in statuses):
                    status_counts["all_pass"] += 1
                elif any(status == "pass" for status in statuses):
                    status_counts["mixed"] += 1
                else:
                    status_counts["all_fail"] += 1
    return {
        "schema_version": SCHEMA,
        "source_dataset": "nvidia/OpenCodeInstruct",
        "source_revision": "main-unfrozen-audit",
        "license_spdx": "CC-BY-4.0",
        "files": manifests,
        "row_count": rows,
        "unique_id_count": len(seen_ids),
        "duplicate_id_count": duplicate_ids,
        "supported_final_code_count": supported_code,
        "supported_final_code_fraction": supported_code / rows if rows else 0.0,
        "language_counts": dict(sorted(language_counts.items())),
        "average_test_score_counts": dict(sorted(score_counts.items())),
        "execution_status_counts": dict(sorted(status_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "generation_algorithm_counts": dict(sorted(algorithm_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit(arguments.parquet)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
