"""Freeze one exact upstream benchmark source into acceptance-case JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.category_case_sources import (
    FrozenAegisV1SafetyCases,
    FrozenCategoryCases,
    FrozenMcEvalExecutionCases,
    freeze_aegis_test_cases,
    freeze_aegis_v1_parquet_test_cases,
    freeze_ifeval_cases,
    freeze_mceval_execution_cases,
)
from bosscrafts_model.scientific_sandbox import DockerScientificSandbox


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)


def _result_payload(result: FrozenCategoryCases) -> dict[str, object]:
    return {
        "output": str(result.output),
        "record_count": result.record_count,
        "source_sha256": result.source_sha256,
        "revision": result.revision,
    }


def _mceval_result_payload(
    result: FrozenMcEvalExecutionCases,
) -> dict[str, object]:
    return {
        "output": str(result.output),
        "audit_output": str(result.audit_output),
        "record_count": result.record_count,
        "rejected_count": result.rejected_count,
        "source_manifest_sha256": result.source_manifest_sha256,
        "revision": result.revision,
    }


def _aegis_v1_result_payload(
    result: FrozenAegisV1SafetyCases,
) -> dict[str, object]:
    return {
        "output": str(result.output),
        "audit_output": str(result.audit_output),
        "record_count": result.record_count,
        "rejected_count": result.rejected_count,
        "source_sha256": result.source_sha256,
        "revision": result.revision,
    }


def _print_mceval_progress(completed: int, total: int) -> None:
    if completed != total and completed % 10 != 0:
        return
    print(
        json.dumps({"completed": completed, "total": total}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="source_kind", required=True)
    _add_source_arguments(subparsers.add_parser("ifeval"))
    _add_source_arguments(subparsers.add_parser("aegis-test"))
    aegis_v1 = subparsers.add_parser("aegis-v1-parquet-test")
    _add_source_arguments(aegis_v1)
    aegis_v1.add_argument("--audit-output", type=Path, required=True)
    mceval = subparsers.add_parser("mceval-execution")
    mceval.add_argument("--source-root", type=Path, required=True)
    mceval.add_argument("--source-manifest", type=Path, required=True)
    mceval.add_argument("--sandbox-manifest", type=Path, required=True)
    mceval.add_argument("--output", type=Path, required=True)
    mceval.add_argument("--audit-output", type=Path, required=True)
    mceval.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()

    if arguments.source_kind == "ifeval":
        result = freeze_ifeval_cases(
            arguments.source,
            arguments.output,
            revision=arguments.revision,
            expected_source_sha256=arguments.source_sha256,
        )
    elif arguments.source_kind == "aegis-test":
        result = freeze_aegis_test_cases(
            arguments.source,
            arguments.output,
            revision=arguments.revision,
            expected_source_sha256=arguments.source_sha256,
        )
    elif arguments.source_kind == "aegis-v1-parquet-test":
        aegis_result = freeze_aegis_v1_parquet_test_cases(
            arguments.source,
            arguments.output,
            arguments.audit_output,
            revision=arguments.revision,
            expected_source_sha256=arguments.source_sha256,
        )
        print(json.dumps(_aegis_v1_result_payload(aegis_result), indent=2, sort_keys=True))
        return 0
    elif arguments.source_kind == "mceval-execution":
        mceval_result = freeze_mceval_execution_cases(
            arguments.source_root,
            arguments.source_manifest,
            arguments.output,
            arguments.audit_output,
            sandbox=DockerScientificSandbox(arguments.sandbox_manifest),
            workers=arguments.workers,
            progress=_print_mceval_progress,
        )
        print(json.dumps(_mceval_result_payload(mceval_result), indent=2, sort_keys=True))
        return 0
    else:
        raise AssertionError(f"unhandled source kind: {arguments.source_kind}")
    print(json.dumps(_result_payload(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
