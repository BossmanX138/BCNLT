"""Deterministically generate frozen category cases from one loaded checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from .category_case_inventory import load_category_case_bundle
from .inference_protocol import INFERENCE_TASKS, task_end_token_id
from .model import EngineeringModel
from .model_acceptance import ACCEPTANCE_CATEGORIES
from .resumable_task_packing import ResumableTaskDataset, audit_task_database
from .tokenizer import NativeTokenizerBinding

MAX_FROZEN_GENERATED_TOKENS = 2_048


def generate_frozen_category_candidates(
    *,
    model: EngineeringModel,
    binding: NativeTokenizerBinding,
    category: str,
    family: str,
    case_bundle_root: str | Path,
    test_task_database: str | Path,
    maximum_new_tokens: int = MAX_FROZEN_GENERATED_TOKENS,
) -> dict[str, list[int]]:
    """Greedily generate every exact packed reference for one frozen category."""
    if type(model) is not EngineeringModel or type(binding) is not NativeTokenizerBinding:
        raise TypeError("frozen generation requires the production model and binding")
    if category not in ACCEPTANCE_CATEGORIES:
        raise ValueError("frozen generation category is invalid")
    if family not in INFERENCE_TASKS or family == "raw":
        raise ValueError("frozen generation family is invalid")
    if (
        type(maximum_new_tokens) is not int
        or not 1 <= maximum_new_tokens <= MAX_FROZEN_GENERATED_TOKENS
    ):
        raise ValueError("frozen generation token limit is invalid")
    if (
        model.config.vocab_size != binding.vocab_size
        or model.config.tokenizer_manifest != binding.binding_sha256
    ):
        raise ValueError("frozen generation model binding mismatch")

    bundle = load_category_case_bundle(case_bundle_root)
    references = bundle.inventory.packed_task_references[category]
    if not references:
        raise ValueError("frozen generation category has no packed cases")
    audit_before = audit_task_database(binding, test_task_database)
    if (
        not audit_before.audit_passed
        or audit_before.database_identity_sha256
        != bundle.inventory.test_task_database_identity_sha256
    ):
        raise ValueError("frozen generation test database lineage mismatch")
    dataset = ResumableTaskDataset(binding, test_task_database)
    task_ids = tuple(reference.task_example_id for reference in references)
    tasks = {
        task["example_id"]: task
        for task in dataset.iter_examples_by_id(task_ids, split="test")
    }
    generated_by_example_id: dict[str, list[int]] = {}
    end_token = task_end_token_id(binding, family)
    device = model.embedding.weight.device
    for reference in references:
        task = tasks.get(reference.task_example_id)
        if (
            task is None
            or reference.family != family
            or task["family"] != family
            or task["language"] != reference.language
        ):
            raise ValueError("frozen generation task lineage mismatch")
        input_ids = task["input_ids"]
        loss_start = task["loss_start_index"]
        if not 0 < loss_start < len(input_ids):
            raise ValueError("frozen generation prompt is invalid")
        prompt_ids = input_ids[:loss_start]
        per_case_limit = min(
            maximum_new_tokens,
            model.config.max_sequence - len(prompt_ids),
        )
        if per_case_limit < 1:
            raise ValueError("frozen generation has no response capacity")
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            prompt,
            max_new_tokens=per_case_limit,
            eos_token_id=end_token,
        )
        response_ids = generated[0, len(prompt_ids) :].tolist()
        if len(response_ids) > MAX_FROZEN_GENERATED_TOKENS:
            raise ValueError("frozen generation exceeded its receipt bound")
        generated_by_example_id[reference.category_example_id] = response_ids

    audit_after = audit_task_database(binding, test_task_database)
    bundle_after = load_category_case_bundle(case_bundle_root)
    if (
        audit_after != audit_before
        or bundle_after.manifest_sha256 != bundle.manifest_sha256
        or bundle_after.inventory.inventory_sha256
        != bundle.inventory.inventory_sha256
    ):
        raise ValueError("frozen generation inputs changed during execution")
    return generated_by_example_id
