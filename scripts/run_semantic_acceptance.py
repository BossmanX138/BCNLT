"""Generate frozen code-to-semantic cases and publish their receipt."""

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
from bosscrafts_model.checkpoint_case_generation import (
    MAX_FROZEN_GENERATED_TOKENS,
    generate_frozen_category_candidates,
)
from bosscrafts_model.model import EngineeringModel
from bosscrafts_model.model_acceptance import verify_heldout_suite_manifest
from bosscrafts_model.resumable_task_packing import audit_task_database
from bosscrafts_model.semantic_acceptance import (
    build_semantic_acceptance_receipt,
    publish_semantic_acceptance_receipt,
    run_semantic_acceptance_cases,
)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint-database", type=Path, required=True)
    parser.add_argument("--validation-database", type=Path, required=True)
    parser.add_argument("--test-database", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_FROZEN_GENERATED_TOKENS
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if not 1 <= arguments.max_new_tokens <= MAX_FROZEN_GENERATED_TOKENS:
        parser.error(
            "max-new-tokens must be between 1 and "
            f"{MAX_FROZEN_GENERATED_TOKENS}"
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
        raise ValueError("semantic acceptance requires fully audited task databases")
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
        raise ValueError("semantic acceptance bundle is not bound to the test database")

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
        raise ValueError("semantic acceptance requires a completed checkpoint")

    generated_ids = generate_frozen_category_candidates(
        model=model,
        binding=binding,
        category="semantic_fidelity",
        family="code_to_semantic",
        case_bundle_root=arguments.case_bundle,
        test_task_database=arguments.test_database,
        maximum_new_tokens=arguments.max_new_tokens,
    )
    result = run_semantic_acceptance_cases(
        binding,
        case_bundle_root=arguments.case_bundle,
        test_task_database=arguments.test_database,
        generated_ids_by_example_id=generated_ids,
    )
    receipt = build_semantic_acceptance_receipt(
        result,
        checkpoint_sha256=state.checkpoint_sha256,
        checkpoint_step=state.step,
    )
    destination = publish_semantic_acceptance_receipt(arguments.output, receipt)
    print(
        json.dumps(
            {
                "category": "semantic_fidelity",
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
