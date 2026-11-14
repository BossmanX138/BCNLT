from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
from test_full_model_acceptance_categories import (
    _model_response_generation_receipt,
)

import bosscrafts_model.ifeval_case_runner as runner
from bosscrafts_model.category_case_inventory import CATEGORY_CASE_SCHEMA
from bosscrafts_model.ifeval_case_runner import (
    IfevalBatchResult,
    _evaluate_payload,
    _worker_environment,
    build_ifeval_batch_receipt,
    publish_ifeval_batch_receipt,
    run_ifeval_cases,
    validate_ifeval_batch_receipt,
)
from bosscrafts_model.ifeval_runtime import (
    EXPECTED_IFEVAL_RUNTIME_MANIFEST_SHA256,
    IfevalRuntime,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _case(index: int) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": CATEGORY_CASE_SCHEMA,
        "category": "ordinary_instruction",
        "case_kind": "ifeval_instruction",
        "case": {
            "source_key": 1000 + index,
            "prompt": f"Prompt {index}",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        },
        "provenance": {
            "source": "test-fixture",
            "revision": "a" * 40,
            "source_sha256": "b" * 64,
            "license": "Apache-2.0",
            "attribution": "Test fixture",
        },
    }
    return {**body, "example_id": hashlib.sha256(_canonical(body)).hexdigest()}


def _fake_evaluator(root: Path) -> None:
    package = root / "instruction_following_eval"
    package.mkdir(parents=True)
    (root / "nltk_data").mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "evaluation_lib.py").write_text(
        """
from dataclasses import dataclass

@dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, object]]

@dataclass
class OutputExample:
    follow_all_instructions: bool
    follow_instruction_list: list[bool]

def test_instruction_following_strict(inp, responses):
    passed = responses[inp.prompt].startswith("STRICT")
    return OutputExample(passed, [passed for _ in inp.instruction_id_list])

def test_instruction_following_loose(inp, responses):
    response = responses[inp.prompt]
    passed = response.startswith("STRICT") or response.startswith("LOOSE")
    return OutputExample(passed, [passed for _ in inp.instruction_id_list])
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_worker_scores_strict_loose_and_baseline_deterministically(
    tmp_path: Path,
) -> None:
    _fake_evaluator(tmp_path)
    case = _case(0)
    payload = {
        "schema_version": "bosscrafts-ifeval-worker-input-v1",
        "runtime_manifest_sha256": "c" * 64,
        "cases": [
            {
                "example_id": case["example_id"],
                **case["case"],
                "candidate_response": "LOOSE candidate",
                "baseline_response": "STRICT baseline",
            }
        ],
    }

    original_random_state = random.getstate()
    original_dont_write_bytecode = sys.dont_write_bytecode

    output = _evaluate_payload(tmp_path, payload)

    assert output["runtime_manifest_sha256"] == "c" * 64
    assert output["results"] == [
        {
            "example_id": case["example_id"],
            "candidate_strict": False,
            "candidate_loose": True,
            "baseline_strict": True,
            "candidate_strict_instruction_results": [False],
            "candidate_loose_instruction_results": [True],
            "baseline_strict_instruction_results": [True],
        }
    ]
    assert random.getstate() == original_random_state
    assert sys.dont_write_bytecode is original_dont_write_bytecode


def test_runner_binds_exact_cases_responses_and_verified_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = sorted([_case(0), _case(1)], key=lambda item: str(item["example_id"]))
    runtime = IfevalRuntime(
        root=tmp_path.resolve(),
        manifest_sha256="c" * 64,
        evaluator_revision="d" * 40,
        nltk_revision="e" * 40,
    )
    verifications: list[Path] = []

    def verify(root: str | Path) -> IfevalRuntime:
        verifications.append(Path(root).resolve())
        return runtime

    def invoke(root: Path, payload: dict[str, object], timeout: int):
        assert root == runtime.root
        assert timeout == 300
        results = []
        for item in payload["cases"]:
            assert type(item) is dict
            results.append(
                {
                    "example_id": item["example_id"],
                    "candidate_strict": True,
                    "candidate_loose": True,
                    "baseline_strict": False,
                    "candidate_strict_instruction_results": [True],
                    "candidate_loose_instruction_results": [True],
                    "baseline_strict_instruction_results": [False],
                }
            )
        return {
            "schema_version": "bosscrafts-ifeval-worker-output-v1",
            "runtime_manifest_sha256": runtime.manifest_sha256,
            "results": results,
        }

    monkeypatch.setattr(runner, "verify_ifeval_runtime", verify)
    monkeypatch.setattr(runner, "_invoke_worker", invoke)
    candidate = {str(case["example_id"]): "candidate" for case in cases}
    baseline = {str(case["example_id"]): "" for case in cases}

    result = run_ifeval_cases(tmp_path, cases, candidate, baseline)

    assert verifications == [runtime.root, runtime.root]
    assert result.runtime_manifest_sha256 == runtime.manifest_sha256
    assert result.records == tuple(
        {
            "example_id": case["example_id"],
            "strict_instruction_following": True,
            "loose_instruction_following": True,
            "baseline_strict_instruction_following": False,
            "response_nonempty": True,
        }
        for case in cases
    )
    assert len(result.details) == len(cases)


def test_runner_rejects_missing_response_and_substituted_worker_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(0)
    runtime = IfevalRuntime(
        root=tmp_path.resolve(),
        manifest_sha256="c" * 64,
        evaluator_revision="d" * 40,
        nltk_revision="e" * 40,
    )
    monkeypatch.setattr(runner, "verify_ifeval_runtime", lambda root: runtime)
    with pytest.raises(ValueError, match="response inventory"):
        run_ifeval_cases(tmp_path, [case], {}, {str(case["example_id"]): "base"})

    monkeypatch.setattr(
        runner,
        "_invoke_worker",
        lambda root, payload, timeout: {
            "schema_version": "bosscrafts-ifeval-worker-output-v1",
            "runtime_manifest_sha256": runtime.manifest_sha256,
            "results": [
                {
                    "example_id": "f" * 64,
                    "candidate_strict": True,
                    "candidate_loose": True,
                    "baseline_strict": False,
                    "candidate_strict_instruction_results": [True],
                    "candidate_loose_instruction_results": [True],
                    "baseline_strict_instruction_results": [False],
                }
            ],
        },
    )
    responses = {str(case["example_id"]): "response"}
    with pytest.raises(ValueError, match="inventory"):
        run_ifeval_cases(tmp_path, [case], responses, responses)


def test_worker_environment_pins_nltk_data_without_forwarding_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "must-not-cross-worker-boundary")

    environment = _worker_environment(tmp_path)

    assert environment["NLTK_DATA"] == str(tmp_path.resolve() / "nltk_data")
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "HF_TOKEN" not in environment


def test_batch_receipt_is_derived_self_hashed_and_published_no_clobber(
    tmp_path: Path,
) -> None:
    ids = tuple(sorted(str(_case(index)["example_id"]) for index in range(2)))
    details = tuple(
        {
            "example_id": example_id,
            "candidate_strict": True,
            "candidate_loose": True,
            "baseline_strict": False,
            "candidate_strict_instruction_results": [True],
            "candidate_loose_instruction_results": [True],
            "baseline_strict_instruction_results": [False],
        }
        for example_id in ids
    )
    records = tuple(
        {
            "example_id": example_id,
            "strict_instruction_following": True,
            "loose_instruction_following": True,
            "baseline_strict_instruction_following": False,
            "response_nonempty": True,
        }
        for example_id in ids
    )
    result = IfevalBatchResult(
        runtime_manifest_sha256=EXPECTED_IFEVAL_RUNTIME_MANIFEST_SHA256,
        case_example_ids=ids,
        records=records,
        details=details,
    )

    receipt = build_ifeval_batch_receipt(
        result,
        _model_response_generation_receipt(
            "ordinary_instruction", ids, "candidate"
        ),
        _model_response_generation_receipt(
            "ordinary_instruction",
            ids,
            "",
            source_type="deterministic_random_initialization",
        ),
    )
    validated = validate_ifeval_batch_receipt(
        receipt,
        expected_checkpoint_sha256="e" * 64,
        expected_checkpoint_step=86_016,
        expected_example_ids=ids,
        expected_records=records,
    )

    assert validated == receipt
    assert receipt["records"] == list(records)
    assert receipt["case_example_ids_sha256"] == hashlib.sha256(
        _canonical(list(ids))
    ).hexdigest()
    output = tmp_path / "receipt.json"
    assert publish_ifeval_batch_receipt(output, receipt) == output.resolve()
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    with pytest.raises(ValueError, match="already exists"):
        publish_ifeval_batch_receipt(output, receipt)

    forged = json.loads(json.dumps(receipt))
    forged["details"][0]["candidate_strict"] = False
    forged["details"][0]["candidate_strict_instruction_results"] = [False]
    forged["details_sha256"] = hashlib.sha256(
        _canonical(forged["details"])
    ).hexdigest()
    forged_body = {
        key: value for key, value in forged.items() if key != "receipt_sha256"
    }
    forged["receipt_sha256"] = hashlib.sha256(_canonical(forged_body)).hexdigest()
    with pytest.raises(ValueError, match="record is inconsistent"):
        validate_ifeval_batch_receipt(forged)


def test_ifeval_acceptance_script_generates_responses_from_checkpoint() -> None:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_ifeval_acceptance.py"),
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
    assert "--candidate-responses" not in completed.stdout
    assert "--baseline-responses" not in completed.stdout
