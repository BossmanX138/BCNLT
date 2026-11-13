"""Build one model-acceptance category artifact from immutable case outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.model_acceptance import ACCEPTANCE_CATEGORIES
from bosscrafts_model.model_category_evidence import (
    MAX_CATEGORY_RECORDS_BYTES,
    checkpoint_sha256,
    compile_model_category_evidence,
    publish_model_category_evidence,
)
from bosscrafts_model.resumable_task_packing import audit_task_database
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_bounded(path: Path, label: str) -> bytes:
    try:
        content = path.resolve().read_bytes()
    except OSError as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not content or len(content) > MAX_CATEGORY_RECORDS_BYTES:
        raise ValueError(f"{label} size is invalid: {path}")
    return content


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            _read_bounded(path, label).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _read_records(path: Path) -> list[dict[str, object]]:
    try:
        text = _read_bounded(path, "category records").decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("category records are not UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("category records must be non-empty JSONL")
    records: list[dict[str, object]] = []
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise ValueError(f"category record line {index} is invalid JSON") from error
        if type(value) is not dict:
            raise ValueError(f"category record line {index} is not an object")
        records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=ACCEPTANCE_CATEGORIES, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--task-database", type=Path)
    parser.add_argument("--validation-database", type=Path)
    parser.add_argument("--test-database", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--strict-uncertainty-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    plural_databases = (
        arguments.validation_database is not None or arguments.test_database is not None
    )
    if (arguments.validation_database is None) != (arguments.test_database is None):
        parser.error("validation-database and test-database must be supplied together")
    if (arguments.task_database is None) == (not plural_databases):
        parser.error("supply either task-database or both split databases")

    binding = NativeTokenizerBinding.load()
    database_identity: str | None = None
    database_identities: dict[str, str] | None = None
    if arguments.task_database is not None:
        database_identity = audit_task_database(
            binding, arguments.task_database
        ).database_identity_sha256
    else:
        if arguments.validation_database is None or arguments.test_database is None:
            raise RuntimeError("validated split databases are missing")
        database_identities = {
            "validation": audit_task_database(
                binding, arguments.validation_database
            ).database_identity_sha256,
            "test": audit_task_database(
                binding, arguments.test_database
            ).database_identity_sha256,
        }
    receipt = (
        _read_object(
            arguments.strict_uncertainty_receipt,
            "strict uncertainty receipt",
        )
        if arguments.strict_uncertainty_receipt is not None
        else None
    )
    payload = compile_model_category_evidence(
        category=arguments.category,
        records=_read_records(arguments.records),
        heldout_manifest=_read_object(
            arguments.heldout_manifest,
            "held-out manifest",
        ),
        task_database_identity_sha256=database_identity,
        task_database_identities_sha256=database_identities,
        checkpoint_sha256=checkpoint_sha256(arguments.checkpoint),
        checkpoint_step=arguments.checkpoint_step,
        strict_uncertainty_receipt=receipt,
    )
    publish_model_category_evidence(arguments.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
