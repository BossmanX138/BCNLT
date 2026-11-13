from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_full_model_acceptance_categories import (
    _model_response_generation_receipt,
)

import bosscrafts_model.safety_judge_case_runner as runner
from bosscrafts_model.category_case_inventory import CATEGORY_CASE_SCHEMA
from bosscrafts_model.safety_judge_case_runner import (
    SafetyJudgeBatchResult,
    build_safety_content_receipt,
    parse_safety_judgment,
    publish_safety_content_receipt,
    run_safety_judge_cases,
    validate_safety_content_receipt,
)
from bosscrafts_model.safety_judge_runtime import (
    EXPECTED_SAFETY_JUDGE_RUNTIME_MANIFEST_SHA256,
    SafetyJudgeRuntime,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _case(index: int, label: str) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": CATEGORY_CASE_SCHEMA,
        "category": "safety",
        "case_kind": "aegis_prompt_moderation",
        "case": {
            "source_id": hashlib.md5(
                f"safety:{index}".encode(), usedforsecurity=False
            ).hexdigest(),
            "split": "test",
            "prompt": f"Safety prompt {index}",
            "prompt_label": label,
            "prompt_label_source": "human",
            "violated_categories": ["Violence"] if label == "unsafe" else [],
        },
        "provenance": {
            "source": "test-fixture",
            "revision": "a" * 40,
            "source_sha256": "b" * 64,
            "license": "CC-BY-4.0",
            "attribution": "Test fixture",
        },
    }
    return {**body, "example_id": hashlib.sha256(_canonical(body)).hexdigest()}


def test_parses_exact_documented_safety_judgment() -> None:
    judgment = parse_safety_judgment(
        "User Safety: unsafe\n"
        "Response Safety: safe\n"
        "Safety Categories: Violence, Criminal Planning/Confessions"
    )

    assert judgment.user_safety == "unsafe"
    assert judgment.response_safety == "safe"
    assert judgment.categories == ("Violence", "Criminal Planning/Confessions")
    assert parse_safety_judgment(
        "User Safety: safe\nResponse Safety: safe"
    ).categories == ()

    for invalid in (
        "User Safety: maybe\nResponse Safety: safe",
        "Response Safety: safe\nUser Safety: safe",
        "User Safety: safe\nResponse Safety: safe\nSafety Categories: Violence",
        "User Safety: unsafe\nResponse Safety: safe\nSafety Categories: Unknown",
        "User Safety: unsafe\nResponse Safety: safe\nSafety Categories: Violence, Violence",
    ):
        with pytest.raises(ValueError):
            parse_safety_judgment(invalid)


def test_runner_binds_human_prompt_labels_candidate_responses_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = sorted(
        [_case(0, "safe"), _case(1, "unsafe")],
        key=lambda item: str(item["example_id"]),
    )
    runtime = SafetyJudgeRuntime(
        root=tmp_path.resolve(),
        repository="fixture/judge",
        revision="a" * 40,
        manifest_sha256=EXPECTED_SAFETY_JUDGE_RUNTIME_MANIFEST_SHA256,
        total_bytes=1,
    )
    verifications: list[Path] = []

    def verify(root: str | Path) -> SafetyJudgeRuntime:
        verifications.append(Path(root).resolve())
        return runtime

    def invoke(
        executable: Path,
        root: Path,
        payload: dict[str, object],
        timeout: int,
    ) -> dict[str, object]:
        assert executable == Path(sys.executable).resolve()
        assert root == runtime.root
        assert timeout == 7_200
        results = []
        for item in payload["cases"]:
            assert type(item) is dict
            expected = next(
                case["case"]["prompt_label"]
                for case in cases
                if case["example_id"] == item["example_id"]
            )
            results.append(
                {
                    "example_id": item["example_id"],
                    "raw_output": (
                        f"User Safety: {expected}\nResponse Safety: safe"
                    ),
                }
            )
        return {
            "schema_version": "bosscrafts-safety-judge-worker-output-v1",
            "runtime_manifest_sha256": runtime.manifest_sha256,
            "results": results,
        }

    monkeypatch.setattr(runner, "verify_safety_judge_runtime", verify)
    monkeypatch.setattr(runner, "_invoke_worker", invoke)
    responses = {
        str(case["example_id"]): f"safe candidate {index}"
        for index, case in enumerate(cases)
    }

    result = run_safety_judge_cases(
        tmp_path, sys.executable, cases, responses
    )

    assert verifications == [runtime.root, runtime.root]
    assert all(record["prompt_moderation_correct"] for record in result.records)
    assert all(record["response_safe"] for record in result.records)
    assert all(record["response_nonempty"] for record in result.records)


def test_receipt_is_derived_self_hashed_and_rejects_forged_record(
    tmp_path: Path,
) -> None:
    example_ids = tuple(
        sorted(str(_case(index, "safe")["example_id"]) for index in range(2))
    )
    records = tuple(
        {
            "example_id": example_id,
            "prompt_moderation_correct": True,
            "response_safe": True,
            "response_nonempty": True,
        }
        for example_id in example_ids
    )
    details = tuple(
        {
            "example_id": example_id,
            "expected_prompt_label": "safe",
            "observed_prompt_label": "safe",
            "observed_response_label": "safe",
            "safety_categories": [],
            "raw_output": "User Safety: safe\nResponse Safety: safe",
            "raw_output_sha256": hashlib.sha256(
                b"User Safety: safe\nResponse Safety: safe"
            ).hexdigest(),
        }
        for example_id in example_ids
    )
    result = SafetyJudgeBatchResult(
        runtime_manifest_sha256=EXPECTED_SAFETY_JUDGE_RUNTIME_MANIFEST_SHA256,
        case_example_ids=example_ids,
        records=records,
        details=details,
    )

    receipt = build_safety_content_receipt(
        result,
        _model_response_generation_receipt("safety", example_ids, "candidate"),
    )
    assert validate_safety_content_receipt(
        receipt,
        expected_checkpoint_sha256="e" * 64,
        expected_checkpoint_step=86_016,
        expected_example_ids=example_ids,
        expected_records=records,
    ) == receipt
    output = tmp_path / "receipt.json"
    assert publish_safety_content_receipt(output, receipt) == output.resolve()
    with pytest.raises(ValueError, match="already exists"):
        publish_safety_content_receipt(output, receipt)

    forged = json.loads(json.dumps(receipt))
    forged["records"][0]["response_safe"] = False
    forged["records_sha256"] = hashlib.sha256(
        _canonical(forged["records"])
    ).hexdigest()
    body = {
        key: value for key, value in forged.items() if key != "receipt_sha256"
    }
    forged["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(ValueError, match="record is inconsistent"):
        validate_safety_content_receipt(forged)


def test_worker_environment_does_not_forward_hugging_face_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "must-not-cross-worker-boundary")

    environment = runner._worker_environment(tmp_path)

    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "HF_TOKEN" not in environment


def test_safety_acceptance_script_generates_responses_from_checkpoint() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "run_safety_content_acceptance.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--runtime-root" in completed.stdout
    assert "--judge-python" in completed.stdout
    assert "--case-bundle" in completed.stdout
    assert "--checkpoint-database" in completed.stdout
    assert "--heldout-manifest" in completed.stdout
    assert "--candidate-responses" not in completed.stdout
