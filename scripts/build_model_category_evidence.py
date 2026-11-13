"""Build one model-acceptance category artifact from immutable case outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.code_acceptance import MAX_CODE_ACCEPTANCE_RECEIPT_BYTES
from bosscrafts_model.model_acceptance import ACCEPTANCE_CATEGORIES
from bosscrafts_model.model_category_evidence import (
    MAX_CATEGORY_RECORDS_BYTES,
    checkpoint_sha256,
    compile_model_category_evidence,
    publish_model_category_evidence,
)
from bosscrafts_model.resumable_task_packing import audit_task_database
from bosscrafts_model.semantic_acceptance import (
    MAX_SEMANTIC_ACCEPTANCE_RECEIPT_BYTES,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_bounded(
    path: Path,
    label: str,
    *,
    maximum_bytes: int = MAX_CATEGORY_RECORDS_BYTES,
) -> bytes:
    try:
        source = path.resolve()
        before = source.stat()
        if not 0 < before.st_size <= maximum_bytes:
            raise ValueError(f"{label} size is invalid: {path}")
        content = source.read_bytes()
        after = source.stat()
    except OSError as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if (
        len(content) != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ValueError(f"{label} changed while reading: {path}")
    if not content:
        raise ValueError(f"{label} size is invalid: {path}")
    return content


def _read_object(
    path: Path,
    label: str,
    *,
    maximum_bytes: int = MAX_CATEGORY_RECORDS_BYTES,
) -> dict[str, object]:
    try:
        value = json.loads(
            _read_bounded(path, label, maximum_bytes=maximum_bytes).decode("utf-8"),
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
    parser.add_argument("--records", type=Path)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--task-database", type=Path)
    parser.add_argument("--validation-database", type=Path)
    parser.add_argument("--test-database", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--strict-uncertainty-receipt", type=Path)
    parser.add_argument("--ifeval-batch-receipt", type=Path)
    parser.add_argument("--safety-content-receipt", type=Path)
    parser.add_argument("--containment-receipt", type=Path)
    parser.add_argument("--regression-receipt", type=Path)
    parser.add_argument("--tokenizer-fidelity-receipt", type=Path)
    parser.add_argument("--code-acceptance-receipt", type=Path)
    parser.add_argument("--semantic-acceptance-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    plural_databases = (
        arguments.validation_database is not None or arguments.test_database is not None
    )
    if (arguments.validation_database is None) != (arguments.test_database is None):
        parser.error("validation-database and test-database must be supplied together")
    if (arguments.task_database is None) == (not plural_databases):
        parser.error("supply either task-database or both split databases")
    if arguments.category == "tokenizer_fidelity":
        if arguments.tokenizer_fidelity_receipt is None:
            parser.error(
                "tokenizer_fidelity requires --tokenizer-fidelity-receipt"
            )
        if arguments.records is not None:
            parser.error(
                "tokenizer_fidelity records come from the tokenizer receipt"
            )
    elif arguments.category == "semantic_fidelity":
        if arguments.semantic_acceptance_receipt is None:
            parser.error(
                "semantic_fidelity requires --semantic-acceptance-receipt"
            )
        if arguments.strict_uncertainty_receipt is None:
            parser.error("semantic_fidelity requires --strict-uncertainty-receipt")
        if arguments.records is not None:
            parser.error(
                "semantic_fidelity records come from the semantic receipt"
            )
    elif arguments.category in {"raw_code", "execution_correctness"}:
        if arguments.code_acceptance_receipt is None:
            parser.error(
                f"{arguments.category} requires --code-acceptance-receipt"
            )
        if arguments.records is not None:
            parser.error(
                f"{arguments.category} records come from the code receipt"
            )
    elif arguments.category == "ordinary_instruction":
        if arguments.ifeval_batch_receipt is None:
            parser.error("ordinary_instruction requires --ifeval-batch-receipt")
        if arguments.records is not None:
            parser.error("ordinary_instruction records come from the IFEval receipt")
    elif arguments.category == "safety":
        if arguments.safety_content_receipt is None:
            parser.error("safety requires --safety-content-receipt")
        if arguments.containment_receipt is None:
            parser.error("safety requires --containment-receipt")
        if arguments.records is not None:
            parser.error("safety records come from the safety content receipt")
    elif arguments.category == "regression":
        if arguments.regression_receipt is None:
            parser.error("regression requires --regression-receipt")
        if arguments.records is not None:
            parser.error("regression records come from the regression receipt")
    elif arguments.records is None:
        parser.error("all other categories require --records")

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
    ifeval_receipt = (
        _read_object(arguments.ifeval_batch_receipt, "IFEval batch receipt")
        if arguments.ifeval_batch_receipt is not None
        else None
    )
    safety_receipt = (
        _read_object(arguments.safety_content_receipt, "safety content receipt")
        if arguments.safety_content_receipt is not None
        else None
    )
    containment_receipt = (
        _read_object(arguments.containment_receipt, "containment receipt")
        if arguments.containment_receipt is not None
        else None
    )
    regression_receipt = (
        _read_object(arguments.regression_receipt, "regression receipt")
        if arguments.regression_receipt is not None
        else None
    )
    tokenizer_receipt = (
        _read_object(
            arguments.tokenizer_fidelity_receipt,
            "tokenizer fidelity receipt",
        )
        if arguments.tokenizer_fidelity_receipt is not None
        else None
    )
    code_receipt = (
        _read_object(
            arguments.code_acceptance_receipt,
            "code acceptance receipt",
            maximum_bytes=MAX_CODE_ACCEPTANCE_RECEIPT_BYTES,
        )
        if arguments.code_acceptance_receipt is not None
        else None
    )
    semantic_receipt = (
        _read_object(
            arguments.semantic_acceptance_receipt,
            "semantic acceptance receipt",
            maximum_bytes=MAX_SEMANTIC_ACCEPTANCE_RECEIPT_BYTES,
        )
        if arguments.semantic_acceptance_receipt is not None
        else None
    )
    if arguments.category == "tokenizer_fidelity":
        raw_records = (
            tokenizer_receipt.get("records")
            if tokenizer_receipt is not None
            else None
        )
        if type(raw_records) is not list or any(
            type(record) is not dict for record in raw_records
        ):
            raise ValueError("tokenizer fidelity receipt records are invalid")
        records = raw_records
    elif arguments.category == "semantic_fidelity":
        raw_records = (
            semantic_receipt.get("records")
            if semantic_receipt is not None
            else None
        )
        if type(raw_records) is not list or any(
            type(record) is not dict for record in raw_records
        ):
            raise ValueError("semantic acceptance receipt records are invalid")
        records = raw_records
    elif arguments.category in {"raw_code", "execution_correctness"}:
        raw_records = code_receipt.get("records") if code_receipt is not None else None
        if type(raw_records) is not list or any(
            type(record) is not dict for record in raw_records
        ):
            raise ValueError("code acceptance receipt records are invalid")
        records = raw_records
    elif arguments.category == "ordinary_instruction":
        raw_records = (
            ifeval_receipt.get("records") if ifeval_receipt is not None else None
        )
        if type(raw_records) is not list or any(
            type(record) is not dict for record in raw_records
        ):
            raise ValueError("IFEval batch receipt records are invalid")
        records = raw_records
    elif arguments.category == "safety":
        raw_records = (
            safety_receipt.get("records") if safety_receipt is not None else None
        )
        if type(raw_records) is not list or any(
            type(record) is not dict for record in raw_records
        ):
            raise ValueError("safety content receipt records are invalid")
        records = raw_records
    elif arguments.category == "regression":
        raw_record = (
            regression_receipt.get("category_record")
            if regression_receipt is not None
            else None
        )
        if type(raw_record) is not dict:
            raise ValueError("regression receipt category record is invalid")
        records = [raw_record]
    else:
        if arguments.records is None:
            raise RuntimeError("validated category records path is missing")
        records = _read_records(arguments.records)
    payload = compile_model_category_evidence(
        category=arguments.category,
        records=records,
        heldout_manifest=_read_object(
            arguments.heldout_manifest,
            "held-out manifest",
        ),
        task_database_identity_sha256=database_identity,
        task_database_identities_sha256=database_identities,
        checkpoint_sha256=checkpoint_sha256(arguments.checkpoint),
        checkpoint_step=arguments.checkpoint_step,
        strict_uncertainty_receipt=receipt,
        ifeval_batch_receipt=ifeval_receipt,
        safety_content_receipt=safety_receipt,
        containment_receipt=containment_receipt,
        regression_receipt=regression_receipt,
        tokenizer_fidelity_receipt=tokenizer_receipt,
        code_acceptance_receipt=code_receipt,
        semantic_acceptance_receipt=semantic_receipt,
    )
    publish_model_category_evidence(arguments.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
