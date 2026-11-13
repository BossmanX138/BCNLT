"""Wait for V8 training, then evaluate and package it with the V9 runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V8_ROOT = PROJECT_ROOT.parent / ".codex_staging_bosscrafts_model_v8_candidate"
V7_ROOT = PROJECT_ROOT.parent / ".codex_staging_bosscrafts_model_v7_candidate"
DATABASE = (
    V7_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v7-65536-deterministic-clean-v1"
    / "tasks.sqlite"
)
V8_CHECKPOINT = (
    V8_ROOT / "artifacts" / "checkpoints" / "bosscrafts-engineering-v8" / "checkpoint.pt"
)
V8_TRAINING_STATUS = V8_CHECKPOINT.with_name("training-controller-status.json")
EVALUATION_ROOT = PROJECT_ROOT / "artifacts" / "evaluations" / "v9-post-training"
BUNDLE = PROJECT_ROOT / "artifacts" / "models" / "BossCrafts-Engineering-0.5B-v9"
DEFAULT_STATUS = EVALUATION_ROOT / "post-training-controller-status.json"
EXPECTED_DATABASE_IDENTITY = (
    "f66255dbee97826a604819f8db2f71ec971c521a8d232887e02703fc9e626c52"
)
EXPECTED_MODEL_FINGERPRINT = (
    "383bc3acb9bea01ed9a40fb6293a5ac81ee7ffb27ffbc4fdcfeafd96358b3bca"
)
EXPECTED_OPTIMIZER_STEPS = 10_752
EXPECTED_SPLIT_COUNT = 5_376
EXPECTED_RUNTIME_IDENTITY = (
    "2394d962d00e464bbb8d6d3a64fbc69fc324e965150db9c05b711ebffed1327e"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _wait_for_training(poll_seconds: int) -> dict[str, object]:
    while True:
        if V8_TRAINING_STATUS.is_file():
            payload = _load_json(V8_TRAINING_STATUS, "V8 training status")
            if payload.get("training_complete") is True:
                return payload
        print(
            json.dumps(
                {
                    "stage": "waiting_for_v8_training",
                    "status": str(V8_TRAINING_STATUS),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def _verify_training(payload: dict[str, object]) -> tuple[str, str]:
    training = payload.get("training")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if not (
        payload.get("schema_version")
        == "bosscrafts-v8-training-controller-status-v1"
        and payload.get("training_complete") is True
        and payload.get("database_identity_sha256") == EXPECTED_DATABASE_IDENTITY
        and payload.get("model_fingerprint") == EXPECTED_MODEL_FINGERPRINT
        and type(checkpoint_sha256) is str
        and len(checkpoint_sha256) == 64
        and type(training) is dict
        and training.get("training_complete") is True
        and training.get("total_optimizer_steps") == EXPECTED_OPTIMIZER_STEPS
        and training.get("optimizer_steps_remaining") == 0
        and training.get("epoch_index") == 1
        and training.get("target_epochs") == 1
        and training.get("gradients_finite") is True
    ):
        raise ValueError("completed V8 training status failed the V9 contract")
    if not V8_CHECKPOINT.is_file() or _sha256(V8_CHECKPOINT) != checkpoint_sha256:
        raise ValueError("V8 checkpoint identity mismatch")
    plan_sha256 = training.get("training_plan_sha256")
    if type(plan_sha256) is not str or len(plan_sha256) != 64:
        raise ValueError("V8 training plan identity is invalid")
    return checkpoint_sha256, plan_sha256


def _verify_runtime_identity() -> str:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_v9_runtime_code_identity.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    observed = completed.stdout.strip()
    if completed.returncode != 0 or observed != EXPECTED_RUNTIME_IDENTITY:
        detail = completed.stderr.strip() or observed or "no verifier output"
        raise ValueError("V9 runtime identity verification failed: " + detail)
    return observed


def _evaluation_command(split: str, output: Path) -> list[str]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_checkpoint.py"),
        "--database",
        str(DATABASE),
        "--checkpoint-database",
        str(DATABASE),
        "--checkpoint",
        str(V8_CHECKPOINT),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--full-model",
        "--split",
        split,
        "--max-examples",
        str(EXPECTED_SPLIT_COUNT),
        "--output",
        str(output),
    ]


def _inference_command(family: str, output: Path) -> list[str]:
    if family not in {
        "code_to_semantic",
        "semantic_continuation",
        "semantic_to_code",
    }:
        raise ValueError("inference family is invalid")
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_checkpoint_inference.py"),
        "--database",
        str(DATABASE),
        "--checkpoint-database",
        str(DATABASE),
        "--checkpoint",
        str(V8_CHECKPOINT),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--full-model",
        "--heldout-family",
        family,
        "--heldout-split",
        "test",
        "--max-new-tokens",
        "256",
        "--output",
        str(output),
    ]


def _is_finite_number(value: object) -> bool:
    return (
        (type(value) is int or type(value) is float)
        and math.isfinite(value)
    )


def _verify_evaluation(payload: dict[str, object], split: str) -> None:
    numeric = (
        payload.get("mean_loss"),
        payload.get("baseline_mean_loss"),
        payload.get("relative_loss_improvement"),
        payload.get("target_token_accuracy"),
    )
    if not (
        payload.get("split") == split
        and payload.get("evaluation_complete") is True
        and payload.get("checkpoint_training_complete") is True
        and payload.get("checkpoint_lineage_verified") is True
        and payload.get("full_model") is True
        and payload.get("device") == "cuda"
        and payload.get("dtype") == "bfloat16"
        and payload.get("example_count") == EXPECTED_SPLIT_COUNT
        and payload.get("split_example_count") == EXPECTED_SPLIT_COUNT
        and payload.get("task_database_identity_sha256")
        == EXPECTED_DATABASE_IDENTITY
        and payload.get("model_fingerprint") == EXPECTED_MODEL_FINGERPRINT
        and all(_is_finite_number(value) for value in numeric)
    ):
        raise ValueError(f"{split} evaluation failed its frozen contract")


def _run(command: list[str], label: str) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def _publish_status(
    path: Path,
    *,
    stage: str,
    checkpoint_sha256: str,
    training_plan_sha256: str,
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "bosscrafts-v9-post-training-status-v1",
        "stage": stage,
        "checkpoint": str(V8_CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha256,
        "training_plan_sha256": training_plan_sha256,
        "database": str(DATABASE),
        "database_identity_sha256": EXPECTED_DATABASE_IDENTITY,
        "model_fingerprint": EXPECTED_MODEL_FINGERPRINT,
    }
    if extra:
        payload.update(extra)
    _write_json_atomic(path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--status-only", action="store_true")
    arguments = parser.parse_args()
    if not 5 <= arguments.poll_seconds <= 600:
        parser.error("poll-seconds must be between 5 and 600")
    if arguments.status_only:
        payload = {
            "v8_training_status_exists": V8_TRAINING_STATUS.is_file(),
            "controller_status_exists": arguments.status.is_file(),
            "validation_exists": (EVALUATION_ROOT / "validation.json").is_file(),
            "test_exists": (EVALUATION_ROOT / "test.json").is_file(),
            "generation_report_exists": (
                EVALUATION_ROOT / "strict-generation" / "report.json"
            ).is_file(),
            "bundle_exists": BUNDLE.is_dir(),
        }
        if arguments.status.is_file():
            payload["controller"] = _load_json(arguments.status, "controller status")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    training = _wait_for_training(arguments.poll_seconds)
    runtime_identity = _verify_runtime_identity()
    checkpoint_sha256, training_plan_sha256 = _verify_training(training)
    _publish_status(
        arguments.status,
        stage="evaluating",
        checkpoint_sha256=checkpoint_sha256,
        training_plan_sha256=training_plan_sha256,
        extra={"runtime_code_identity_sha256": runtime_identity},
    )

    evaluations: dict[str, dict[str, object]] = {}
    for split in ("validation", "test"):
        output = EVALUATION_ROOT / f"{split}.json"
        if not output.is_file():
            _run(_evaluation_command(split, output), f"{split} evaluation")
        evaluation = _load_json(output, f"{split} evaluation")
        _verify_evaluation(evaluation, split)
        evaluations[split] = evaluation

    loss_gates_passed = all(
        evaluation.get("promotion_gate_passed") is True
        and evaluation.get("target_token_accuracy_improved") is True
        for evaluation in evaluations.values()
    )
    if not loss_gates_passed:
        _publish_status(
            arguments.status,
            stage="loss_evaluation_failed_promotion",
            checkpoint_sha256=checkpoint_sha256,
            training_plan_sha256=training_plan_sha256,
            extra={"promotion_ready": False, "evaluations": evaluations},
        )
        return 0

    generation_root = EVALUATION_ROOT / "strict-generation"
    generation_report = generation_root / "report.json"
    if not generation_report.is_file():
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_generation_gate.py"),
                "--database",
                str(DATABASE),
                "--checkpoint-database",
                str(DATABASE),
                "--checkpoint",
                str(V8_CHECKPOINT),
                "--output",
                str(generation_root),
                "--device",
                "cuda",
                "--dtype",
                "bfloat16",
                "--split",
                "test",
                "--limit-per-family",
                "16",
                "--max-new-tokens",
                "256",
            ],
            "strict generation evaluation",
        )
    generation = _load_json(generation_report, "strict generation report")
    if not (
        generation.get("generation_complete") is True
        and generation.get("checkpoint_training_complete") is True
        and generation.get("checkpoint_sha256") == checkpoint_sha256
        and generation.get("evaluation_task_database_identity_sha256")
        == EXPECTED_DATABASE_IDENTITY
    ):
        raise ValueError("strict generation report failed its frozen contract")
    if generation.get("strict_gate_passed") is not True:
        _publish_status(
            arguments.status,
            stage="generation_failed_promotion",
            checkpoint_sha256=checkpoint_sha256,
            training_plan_sha256=training_plan_sha256,
            extra={
                "promotion_ready": False,
                "evaluations": evaluations,
                "generation": generation,
            },
        )
        return 0

    inference_paths = {
        family: EVALUATION_ROOT / f"heldout-{family}.json"
        for family in (
            "code_to_semantic",
            "semantic_continuation",
            "semantic_to_code",
        )
    }
    for family, output in inference_paths.items():
        if not output.is_file():
            _run(_inference_command(family, output), f"{family} held-out inference")
    acceptance_path = EVALUATION_ROOT / "model-acceptance.json"
    if not acceptance_path.is_file():
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "verify_model_acceptance.py"),
                "--validation-evaluation",
                str(EVALUATION_ROOT / "validation.json"),
                "--test-evaluation",
                str(EVALUATION_ROOT / "test.json"),
                "--code-to-semantic-inference",
                str(inference_paths["code_to_semantic"]),
                "--semantic-continuation-inference",
                str(inference_paths["semantic_continuation"]),
                "--semantic-to-code-inference",
                str(inference_paths["semantic_to_code"]),
                "--output",
                str(acceptance_path),
            ],
            "formal model acceptance",
        )
    acceptance = _load_json(acceptance_path, "formal model acceptance")
    if not (
        acceptance.get("schema_version") == "bosscrafts-model-acceptance-v1"
        and acceptance.get("acceptance_passed") is True
        and acceptance.get("checkpoint_step") == EXPECTED_OPTIMIZER_STEPS
        and acceptance.get("task_database_identity_sha256")
        == EXPECTED_DATABASE_IDENTITY
        and acceptance.get("model_fingerprint") == EXPECTED_MODEL_FINGERPRINT
        and type(acceptance.get("acceptance_manifest_sha256")) is str
        and len(str(acceptance["acceptance_manifest_sha256"])) == 64
    ):
        raise ValueError("formal model acceptance failed its output contract")

    if not BUNDLE.exists():
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "export_inference_bundle.py"),
                "--database",
                str(DATABASE),
                "--checkpoint",
                str(V8_CHECKPOINT),
                "--output",
                str(BUNDLE),
                "--dtype",
                "bfloat16",
                "--full-model",
            ],
            "inference bundle export",
        )

    smoke = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_inference_bundle.py"),
            "--bundle",
            str(BUNDLE),
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--task",
            "code_to_semantic",
            "--prompt",
            "def add(a, b):\n    return a + b\n",
            "--max-new-tokens",
            "128",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if smoke.returncode != 0:
        raise RuntimeError("inference bundle smoke failed: " + smoke.stderr.strip())
    try:
        smoke_payload = json.loads(smoke.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("inference bundle smoke output is invalid") from error
    if not (
        type(smoke_payload) is dict
        and smoke_payload.get("bundle_lineage_verified") is True
        and smoke_payload.get("response_protocol_complete") is True
        and smoke_payload.get("task_database_identity_sha256")
        == EXPECTED_DATABASE_IDENTITY
    ):
        raise ValueError("inference bundle smoke failed its output contract")
    smoke_path = EVALUATION_ROOT / "bundle-smoke.json"
    _write_json_atomic(smoke_path, smoke_payload)
    _publish_status(
        arguments.status,
        stage="complete",
        checkpoint_sha256=checkpoint_sha256,
        training_plan_sha256=training_plan_sha256,
        extra={
            "promotion_ready": True,
            "evaluations": evaluations,
            "generation": generation,
            "formal_acceptance": acceptance,
            "bundle": str(BUNDLE),
            "bundle_manifest_sha256": _sha256(BUNDLE / "manifest.json"),
            "bundle_smoke": smoke_payload,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
