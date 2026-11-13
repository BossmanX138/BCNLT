"""Verify held-out and inference evidence for the final BossCrafts checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.immutable_artifacts import publish_new_files
from bosscrafts_model.inference_protocol import HeldoutInferenceCase
from bosscrafts_model.model_acceptance import (
    ACCEPTANCE_CATEGORIES,
    STRUCTURED_TASKS,
    verify_model_acceptance,
)
from bosscrafts_model.model_category_evidence import (
    MAX_MODEL_CATEGORY_EVIDENCE_BYTES,
)
from bosscrafts_model.resumable_task_packing import (
    ResumableTaskDataset,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding

MAX_ACCEPTANCE_EVIDENCE_BYTES = 16 * 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate acceptance evidence key: {key}")
        payload[key] = value
    return payload


def _read_payload(
    path: Path,
    *,
    maximum_bytes: int = MAX_ACCEPTANCE_EVIDENCE_BYTES,
) -> dict[str, object]:
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("acceptance evidence size limit is invalid")
    try:
        source = path.resolve()
        before = source.stat()
        if not 0 < before.st_size <= maximum_bytes:
            raise ValueError(f"acceptance evidence exceeds size limit: {path}")
        content = source.read_bytes()
        after = source.stat()
        if (
            len(content) != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ValueError(f"acceptance evidence changed while reading: {path}")
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"acceptance evidence is unreadable: {path}") from error
    if type(payload) is not dict:
        raise ValueError(f"acceptance evidence is not a JSON object: {path}")
    return payload


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _ground_heldout_cases(
    binding: NativeTokenizerBinding,
    database: Path,
    inferences: tuple[dict[str, object], ...],
) -> dict[str, HeldoutInferenceCase]:
    requested: dict[str, str] = {}
    for payload in inferences:
        task = payload.get("task")
        example_id = payload.get("heldout_example_id")
        if (
            type(task) is not str
            or task not in STRUCTURED_TASKS
            or type(example_id) is not str
            or len(example_id) != 64
            or any(character not in "0123456789abcdef" for character in example_id)
            or example_id in requested
        ):
            raise ValueError("inference held-out selection is invalid")
        requested[example_id] = task

    cases: dict[str, HeldoutInferenceCase] = {}
    dataset = ResumableTaskDataset(binding, database)
    for example in dataset.iter_examples_by_id(tuple(requested), split="test"):
        example_id = example.get("example_id")
        if type(example_id) is not str or example_id not in requested:
            continue
        task = requested[example_id]
        family = example.get("family")
        input_ids = example.get("input_ids")
        loss_start = example.get("loss_start_index")
        if (
            family != task
            or task in cases
            or type(input_ids) is not list
            or len(input_ids) < 2
            or any(
                type(token_id) is not int or not 0 <= token_id < binding.vocab_size
                for token_id in input_ids
            )
            or type(loss_start) is not int
            or not 0 < loss_start < len(input_ids)
        ):
            raise ValueError(f"{task} held-out database row is invalid")
        cases[task] = HeldoutInferenceCase(
            example_id=example_id,
            family=task,
            prompt_ids=tuple(input_ids[:loss_start]),
            expected_ids=tuple(input_ids[loss_start:]),
        )
    if set(cases) != set(STRUCTURED_TASKS):
        raise ValueError("task database does not contain every claimed held-out case")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-database", type=Path)
    parser.add_argument("--validation-database", type=Path)
    parser.add_argument("--test-database", type=Path)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--validation-evaluation", type=Path, required=True)
    parser.add_argument("--test-evaluation", type=Path, required=True)
    parser.add_argument("--code-to-semantic-inference", type=Path, required=True)
    parser.add_argument("--semantic-continuation-inference", type=Path, required=True)
    parser.add_argument("--semantic-to-code-inference", type=Path, required=True)
    for category in ACCEPTANCE_CATEGORIES:
        parser.add_argument(
            f"--{category.replace('_', '-')}-evidence",
            dest=f"{category}_evidence",
            type=Path,
            required=True,
        )
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
    inferences = (
        _read_payload(arguments.code_to_semantic_inference),
        _read_payload(arguments.semantic_continuation_inference),
        _read_payload(arguments.semantic_to_code_inference),
    )
    database_identity: str | None = None
    database_identities: dict[str, str] | None = None
    if arguments.task_database is not None:
        test_database = arguments.task_database
        database_identity = audit_task_database(
            binding, test_database
        ).database_identity_sha256
    else:
        if arguments.validation_database is None or arguments.test_database is None:
            raise RuntimeError("validated split databases are missing")
        test_database = arguments.test_database
        database_identities = {
            "validation": audit_task_database(
                binding, arguments.validation_database
            ).database_identity_sha256,
            "test": audit_task_database(
                binding, arguments.test_database
            ).database_identity_sha256,
        }
    heldout_cases = _ground_heldout_cases(
        binding,
        test_database,
        inferences,
    )
    result = verify_model_acceptance(
        (
            _read_payload(arguments.validation_evaluation),
            _read_payload(arguments.test_evaluation),
        ),
        inferences,
        tuple(
            _read_payload(
                getattr(arguments, f"{category}_evidence"),
                maximum_bytes=MAX_MODEL_CATEGORY_EVIDENCE_BYTES,
            )
            for category in ACCEPTANCE_CATEGORIES
        ),
        binding=binding,
        heldout_manifest=_read_payload(arguments.heldout_manifest),
        heldout_cases=heldout_cases,
        task_database_identity_sha256=database_identity,
        task_database_identities_sha256=database_identities,
    )
    payload = asdict(result)
    payload["acceptance_manifest_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    destination = arguments.output.resolve()
    publish_new_files(
        ((destination, rendered.encode("utf-8")),),
        artifact_label="model acceptance",
    )
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
