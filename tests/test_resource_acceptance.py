from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bosscrafts_model.category_case_inventory import CATEGORY_CASE_SCHEMA
from bosscrafts_model.inference_protocol import build_inference_prompt
from bosscrafts_model.model_acceptance import verify_category_metrics
from bosscrafts_model.resource_acceptance import (
    RESOURCE_BENCHMARK_MAX_NEW_TOKENS,
    RESOURCE_BENCHMARK_PROMPT,
    ResourceAcceptanceResult,
    build_resource_acceptance_receipt,
    peak_process_rss_bytes,
    publish_resource_acceptance_receipt,
    resource_acceptance_receipt_metrics,
    validate_resource_acceptance_receipt,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding, bound_native_65k_config

BINDING = NativeTokenizerBinding.load()
CHECKPOINT_ID = "e" * 64
DATABASE_ID = "a" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _case_record() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": CATEGORY_CASE_SCHEMA,
        "category": "resources",
        "case_kind": "resource_budget",
        "case": {
            "vram_limit_bytes": 12 * 1024**3,
            "ram_limit_bytes": 64 * 1024**3,
            "minimum_tokens_per_second": 1.0,
        },
        "provenance": {
            "source": "fixture://resource-budget",
            "revision": "b" * 40,
            "source_sha256": "c" * 64,
            "license": "BossCrafts-internal",
            "attribution": "BossCrafts resource acceptance fixture",
        },
    }
    return {
        **body,
        "example_id": hashlib.sha256(_canonical(body)).hexdigest(),
    }


def _iteration(
    phase: str,
    index: int,
    *,
    token_count: int = 10,
    elapsed_ns: int = 1_000_000_000,
    status: str = "completed",
) -> dict[str, object]:
    generated_ids = [100] * token_count if status == "completed" else []
    return {
        "phase": phase,
        "iteration": index,
        "status": status,
        "elapsed_ns": elapsed_ns,
        "generated_token_count": len(generated_ids),
        "generated_ids": generated_ids,
        "generated_ids_sha256": hashlib.sha256(
            _canonical(generated_ids)
        ).hexdigest(),
    }


def _result(*, oom: bool = False) -> ResourceAcceptanceResult:
    iterations = (
        _iteration("warmup", 0),
        _iteration("measured", 0),
        _iteration("measured", 1, status="oom" if oom else "completed"),
        _iteration("measured", 2),
    )
    return ResourceAcceptanceResult(
        case_bundle_manifest_sha256="1" * 64,
        case_inventory_sha256="2" * 64,
        test_task_database_identity_sha256=DATABASE_ID,
        binding_sha256=BINDING.binding_sha256,
        tokenizer_sha256=BINDING.tokenizer_sha256,
        model_fingerprint=bound_native_65k_config(BINDING).fingerprint(),
        device_type="cuda",
        dtype="bfloat16",
        case_record=_case_record(),
        prompt_ids=tuple(
            build_inference_prompt(
                BINDING,
                task="instruction_to_code",
                prompt=RESOURCE_BENCHMARK_PROMPT,
            )
        ),
        peak_vram_bytes=8 * 1024**3,
        peak_ram_bytes=16 * 1024**3,
        iterations=iterations,
    )


def _receipt(*, oom: bool = False) -> dict[str, object]:
    return build_resource_acceptance_receipt(
        _result(oom=oom),
        checkpoint_sha256=CHECKPOINT_ID,
        checkpoint_step=86_016,
    )


def test_resource_receipt_rederives_budget_usage_and_throughput() -> None:
    receipt = _receipt()
    example_id = str(_case_record()["example_id"])
    record = receipt["record"]
    assert type(record) is dict
    assert record["inference_tokens_per_second"] == 10.0
    assert record["oom_count"] == 0
    assert validate_resource_acceptance_receipt(
        receipt,
        expected_checkpoint_sha256=CHECKPOINT_ID,
        expected_checkpoint_step=86_016,
        expected_example_ids=(example_id,),
        expected_records=[record],
        expected_task_database_identity_sha256=DATABASE_ID,
    ) == receipt
    assert resource_acceptance_receipt_metrics(receipt) == {
        key: record[key]
        for key in (
            "peak_vram_bytes",
            "vram_limit_bytes",
            "peak_ram_bytes",
            "ram_limit_bytes",
            "oom_count",
            "inference_tokens_per_second",
            "minimum_tokens_per_second",
        )
    }


def test_resource_receipt_counts_oom_and_rejects_rehashed_metric_forgery() -> None:
    receipt = _receipt(oom=True)
    record = receipt["record"]
    assert type(record) is dict
    assert record["oom_count"] == 1

    forged = copy.deepcopy(receipt)
    forged_record = forged["record"]
    assert type(forged_record) is dict
    forged_record["inference_tokens_per_second"] = 999.0
    body = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(ValueError, match="measurements are inconsistent"):
        validate_resource_acceptance_receipt(forged)


def test_resource_receipt_rejects_oversized_generation_and_publishes_no_clobber(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    output = tmp_path / "resource.json"
    assert publish_resource_acceptance_receipt(output, receipt) == output.resolve()
    with pytest.raises(ValueError, match="already exists"):
        publish_resource_acceptance_receipt(output, receipt)

    forged = copy.deepcopy(receipt)
    iterations = forged["iterations"]
    assert type(iterations) is list
    generated_ids = [100] * (RESOURCE_BENCHMARK_MAX_NEW_TOKENS + 1)
    iterations[1]["generated_ids"] = generated_ids
    iterations[1]["generated_token_count"] = len(generated_ids)
    iterations[1]["generated_ids_sha256"] = hashlib.sha256(
        _canonical(generated_ids)
    ).hexdigest()
    forged["iterations_sha256"] = hashlib.sha256(
        _canonical(iterations)
    ).hexdigest()
    body = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(ValueError, match="iteration is inconsistent"):
        validate_resource_acceptance_receipt(forged)


def test_os_peak_process_memory_is_available() -> None:
    assert peak_process_rss_bytes() > 0


def test_resource_gate_enforces_the_frozen_minimum_throughput() -> None:
    metrics = resource_acceptance_receipt_metrics(_receipt())
    metrics["inference_tokens_per_second"] = 0.5
    with pytest.raises(ValueError, match="resources acceptance gate failed"):
        verify_category_metrics("resources", metrics)


def test_resource_acceptance_script_requires_checkpoint_lineage() -> None:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_resource_acceptance.py"),
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--case-bundle" in completed.stdout
    assert "--checkpoint-database" in completed.stdout
    assert "--heldout-manifest" in completed.stdout
    assert "--records" not in completed.stdout
