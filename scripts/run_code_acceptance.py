"""Generate and execute frozen code-acceptance cases for one checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.category_case_inventory import load_category_case_bundle
from bosscrafts_model.code_acceptance import (
    CODE_ACCEPTANCE_CATEGORIES,
    MAX_GENERATED_TOKENS,
    build_code_acceptance_receipt,
    publish_code_acceptance_receipt,
    run_code_acceptance_cases,
)
from bosscrafts_model.inference_protocol import task_end_token_id
from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.model_acceptance import verify_heldout_suite_manifest
from bosscrafts_model.resumable_task_packing import (
    ResumableTaskDataset,
    audit_task_database,
)
from bosscrafts_model.scientific_sandbox import DockerScientificSandbox
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config
from bosscrafts_model.training_checkpoint import load_model_checkpoint_for_inference

MAX_HELDOUT_MANIFEST_BYTES = 16 * 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate held-out manifest key: {key}")
        payload[key] = value
    return payload


def _read_heldout_manifest(path: Path) -> dict[str, object]:
    try:
        content = path.resolve().read_bytes()
        if not content or len(content) > MAX_HELDOUT_MANIFEST_BYTES:
            raise ValueError("held-out manifest size is invalid")
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("held-out manifest is unreadable") from error
    if type(payload) is not dict:
        raise ValueError("held-out manifest must be a JSON object")
    return payload


def _generate_candidates(
    *,
    model: EngineeringModel,
    binding: NativeTokenizerBinding,
    category: str,
    case_bundle_root: Path,
    test_task_database: Path,
    device: torch.device,
    maximum_new_tokens: int,
) -> dict[str, list[int]]:
    bundle = load_category_case_bundle(case_bundle_root)
    references = bundle.inventory.packed_task_references[category]
    if not references:
        raise ValueError("code acceptance category has no frozen cases")
    dataset = ResumableTaskDataset(binding, test_task_database)
    task_ids = tuple(reference.task_example_id for reference in references)
    tasks = {
        task["example_id"]: task
        for task in dataset.iter_examples_by_id(task_ids, split="test")
    }
    generated_by_example_id: dict[str, list[int]] = {}
    end_token = task_end_token_id(binding, "semantic_to_code")
    for reference in references:
        task = tasks.get(reference.task_example_id)
        if (
            task is None
            or task["family"] != "semantic_to_code"
            or task["language"] != reference.language
        ):
            raise ValueError("code acceptance generation task lineage mismatch")
        input_ids = task["input_ids"]
        loss_start = task["loss_start_index"]
        if not 0 < loss_start < len(input_ids):
            raise ValueError("code acceptance generation prompt is invalid")
        prompt_ids = input_ids[:loss_start]
        per_case_limit = min(
            maximum_new_tokens,
            model.config.max_sequence - len(prompt_ids),
        )
        if per_case_limit < 1:
            raise ValueError("code acceptance generation has no response capacity")
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            prompt,
            max_new_tokens=per_case_limit,
            eos_token_id=end_token,
        )
        response_ids = generated[0, len(prompt_ids) :].tolist()
        if len(response_ids) > MAX_GENERATED_TOKENS:
            raise ValueError("code acceptance generation exceeded its receipt bound")
        generated_by_example_id[reference.category_example_id] = response_ids
    return generated_by_example_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category", choices=CODE_ACCEPTANCE_CATEGORIES, required=True
    )
    parser.add_argument("--case-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint-database", type=Path, required=True)
    parser.add_argument("--validation-database", type=Path, required=True)
    parser.add_argument("--test-database", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sandbox-manifest",
        type=Path,
        default=PROJECT_ROOT / "sandbox" / "images.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_GENERATED_TOKENS
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if not 1 <= arguments.max_new_tokens <= MAX_GENERATED_TOKENS:
        parser.error(
            f"max-new-tokens must be between 1 and {MAX_GENERATED_TOKENS}"
        )
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if (
        arguments.device == "cuda"
        and arguments.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        parser.error("CUDA bfloat16 was requested but is unsupported")

    binding = NativeTokenizerBinding.load()
    checkpoint_audit = audit_task_database(binding, arguments.checkpoint_database)
    validation_audit = audit_task_database(binding, arguments.validation_database)
    test_audit = audit_task_database(binding, arguments.test_database)
    if not all(
        audit.audit_passed
        for audit in (checkpoint_audit, validation_audit, test_audit)
    ):
        raise ValueError("code acceptance requires fully audited task databases")
    evaluation_identities = {
        "validation": validation_audit.database_identity_sha256,
        "test": test_audit.database_identity_sha256,
    }
    heldout_suite = verify_heldout_suite_manifest(
        _read_heldout_manifest(arguments.heldout_manifest),
        task_database_identities_sha256=evaluation_identities,
    )
    bundle = load_category_case_bundle(arguments.case_bundle)
    if bundle.inventory.test_task_database_identity_sha256 != (
        test_audit.database_identity_sha256
    ):
        raise ValueError("code acceptance bundle is not bound to the test database")

    config = bound_native_65k_config(binding)
    device = torch.device(arguments.device)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model = EngineeringModel(config).to(device=device, dtype=dtype).eval()
    state = load_model_checkpoint_for_inference(
        arguments.checkpoint,
        model,
        expected_task_database_identity_sha256=(
            checkpoint_audit.database_identity_sha256
        ),
        expected_binding_sha256=binding.binding_sha256,
        expected_model_fingerprint=config.fingerprint(),
        expected_evaluation_task_database_identities_sha256=evaluation_identities,
        expected_heldout_suite_manifest_sha256=heldout_suite.manifest_sha256,
    )
    if not (
        state.epoch_index >= state.target_epochs
        and state.cursor_after_example_id is None
    ):
        raise ValueError("code acceptance requires a completed checkpoint")

    generated_ids = _generate_candidates(
        model=model,
        binding=binding,
        category=arguments.category,
        case_bundle_root=arguments.case_bundle,
        test_task_database=arguments.test_database,
        device=device,
        maximum_new_tokens=arguments.max_new_tokens,
    )
    sandbox = DockerScientificSandbox(arguments.sandbox_manifest)
    result = run_code_acceptance_cases(
        binding,
        sandbox,
        category=arguments.category,
        case_bundle_root=arguments.case_bundle,
        test_task_database=arguments.test_database,
        generated_ids_by_example_id=generated_ids,
    )
    receipt = build_code_acceptance_receipt(
        result,
        checkpoint_sha256=state.checkpoint_sha256,
        checkpoint_step=state.step,
    )
    destination = publish_code_acceptance_receipt(arguments.output, receipt)
    print(
        json.dumps(
            {
                "category": arguments.category,
                "checkpoint_sha256": state.checkpoint_sha256,
                "checkpoint_step": state.step,
                "case_count": len(result.records),
                "receipt_sha256": receipt["receipt_sha256"],
                "output": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
