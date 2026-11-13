from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Thread

import pytest


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _feature():
    if importlib.util.find_spec("bosscrafts_model.local_semantic_review") is None:
        pytest.fail("local semantic review feature is not implemented")
    return importlib.import_module("bosscrafts_model.local_semantic_review")


def _queue(root: Path, *, item_count: int = 1) -> None:
    item = {
        "schema_version": "bosscrafts-heldout-semantic-review-item-v2",
        "review_id": "1" * 64,
        "example_id": None,
        "document_id": "2" * 64,
        "record_index": 0,
        "language": "python",
        "split": "test",
        "family": "semantic_record_review",
        "source_excerpt": (
            "[source context lines 1-1; target span 1:0-1:9]\n"
            ">000001| value = 1\n"
        ),
        "canonical_record": {
            "event": "DEFINE",
            "ir": {"kind": "variable", "name": "value", "value": "1", "children": [], "metadata": {}},
            "scope": "module",
            "span": [1, 0, 1, 9],
            "uncertainty_band": "low",
        },
        "source_sha256": "3" * 64,
        "semantic_spec_sha256": "4" * 64,
        "parser_runtime_identity_sha256": "5" * 64,
        "binding_sha256": "6" * 64,
        "tokenizer_sha256": "7" * 64,
        "review_status": "unreviewed",
        "rubric": {
            "event_matches_source": None,
            "ir_kind_plausible": None,
            "scope_plausible": None,
            "span_matches_source": None,
            "uncertainty_calibrated": None,
        },
        "review_notes": None,
    }
    items = []
    for index in range(item_count):
        clone = json.loads(json.dumps(item))
        clone["review_id"] = str(index + 1) * 64
        clone["document_id"] = str(index + 2) * 64
        items.append(clone)
    queue_text = "".join(f"{_canonical(row)}\n" for row in items)
    unsigned = {
        "schema_version": "bosscrafts-heldout-semantic-review-manifest-v2",
        "source_admission_manifest_sha256": "8" * 64,
        "source_packed_manifest_sha256": None,
        "binding_sha256": "6" * 64,
        "tokenizer_sha256": "7" * 64,
        "parser_runtime_identity_sha256": "5" * 64,
        "selection_policy": "fixture",
        "requested_items_per_language": 1,
        "item_count": item_count,
        "represented_languages": ["python"],
        "missing_languages": [],
        "queue_file": "review_queue.jsonl",
        "queue_sha256": _sha256_text(queue_text),
        "review_complete": False,
        "semantic_review_passed": False,
        "training_ready": False,
        "training_blockers": ["fixture"],
    }
    manifest = {**unsigned, "manifest_sha256": _sha256_text(_canonical(unsigned))}
    (root / "review_queue.jsonl").write_text(queue_text, encoding="utf-8", newline="\n")
    (root / "manifest.json").write_text(f"{_canonical(manifest)}\n", encoding="utf-8", newline="\n")


def _decision(choice: str) -> dict[str, object]:
    rubric = {
        "event_matches_source": True,
        "ir_kind_plausible": True,
        "scope_plausible": True,
        "span_matches_source": True,
        "uncertainty_calibrated": True,
    }
    if choice == "reject":
        rubric["span_matches_source"] = False
    return {
        "decision": choice,
        "notes": f"fixture {choice} decision with explicit evidence",
        "rubric": rubric,
    }


@contextmanager
def _review_server(decisions: dict[str, dict[str, object]], *, response_models=None):
    state = {"calls": []}
    response_models = response_models or {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            model = request["model"]
            state["calls"].append(model)
            state.setdefault("requests", []).append(request)
            configured = decisions[model]
            if isinstance(configured, list):
                configured = configured[
                    sum(1 for called in state["calls"] if called == model) - 1
                ]
            payload = {
                "id": f"fixture-{len(state['calls'])}",
                "object": "chat.completion",
                "created": 1,
                "model": response_models.get(model, model),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": _canonical(configured),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
            body = _canonical(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _reviewers(feature, base_url: str):
    return (
        feature.LocalReviewerSpec("reviewer-a", base_url, "model-a", timeout_seconds=5),
        feature.LocalReviewerSpec("reviewer-b", base_url, "model-b", timeout_seconds=5),
    )


def test_reviewer_configuration_rejects_non_loopback_endpoint():
    feature = _feature()
    with pytest.raises(ValueError, match="loopback"):
        feature.LocalReviewerSpec(
            "external-reviewer", "https://api.example.com", "remote-model"
        )


def test_dual_review_is_resumable_and_never_auto_promotes():
    feature = _feature()
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")}
        ) as (base_url, state):
            runner = feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            )
            first = runner.run()
            second = runner.run()
            verified = runner.verify()

        assert state["calls"] == ["model-a", "model-b"]
        assert first.completed_reviews == 2
        assert second.completed_reviews == 2
        assert verified.completed_reviews == 2
        manifest = json.loads((Path(output_dir) / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["review_complete"] is True
        assert manifest["consensus_counts"] == {
            "agree_keep": 1,
            "agree_reject": 0,
            "manual_review": 0,
            "pending": 0,
        }
        assert manifest["semantic_review_passed"] is False
        assert manifest["training_ready"] is False
        assert manifest["requires_final_adjudication"] is True
        system_prompt = state["requests"][0]["messages"][0]["content"]
        assert "UNKNOWN_SEMANTIC can be correct" in system_prompt
        assert "uncertainty_calibrated" in system_prompt
        assert "faithfully represents" in system_prompt
        assert "zero-based and end-exclusive" in system_prompt
        assert "not a band name" in system_prompt
        response_format = state["requests"][0]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"]["additionalProperties"] is False
        rubric_schema = response_format["json_schema"]["schema"]["properties"]["rubric"]
        assert rubric_schema["properties"]["uncertainty_calibrated"]["type"] == [
            "boolean",
            "null",
        ]


def test_disagreement_is_written_to_manual_queue():
    feature = _feature()
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("reject")}
        ) as (base_url, _state):
            feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            ).run()

        manual = [
            json.loads(line)
            for line in (Path(output_dir) / "manual_review_queue.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(manual) == 1
        assert manual[0]["review_id"] == "1" * 64
        assert manual[0]["reason"] == "reviewer_disagreement"


def test_reviews_are_grouped_by_model_to_avoid_reloads():
    feature = _feature()
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir), item_count=2)
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")}
        ) as (base_url, state):
            result = feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            ).run()
        assert result.completed_reviews == 4
        assert state["calls"] == ["model-a", "model-a", "model-b", "model-b"]


def test_per_reviewer_limit_builds_balanced_canary():
    feature = _feature()
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir), item_count=2)
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")}
        ) as (base_url, state):
            result = feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            ).run(limit_per_reviewer=1)
        assert result.completed_reviews == 2
        assert state["calls"] == ["model-a", "model-b"]
        assert result.consensus_counts == {
            "agree_keep": 1,
            "agree_reject": 0,
            "manual_review": 0,
            "pending": 1,
        }


def test_explicit_review_ids_run_only_the_selected_controls():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _queue(Path(queue_dir), item_count=3)
        selected = ("2" * 64, "3" * 64)
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")}
        ) as (base_url, state):
            result = feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            ).run(review_ids=selected)

        decisions = [
            json.loads(line)
            for line in (Path(output_dir) / "reviewer_decisions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert result.completed_reviews == 4
        assert {row["review_id"] for row in decisions} == set(selected)
        assert state["calls"] == ["model-a", "model-a", "model-b", "model-b"]
        assert result.consensus_counts == {
            "agree_keep": 2,
            "agree_reject": 0,
            "manual_review": 0,
            "pending": 1,
        }


def test_span_evidence_protocol_supplies_machine_extracted_source_surface():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")}
        ) as (base_url, state):
            feature.DualLocalSemanticReviewer(
                queue_dir,
                output_dir,
                _reviewers(feature, base_url),
                review_protocol="span-evidence-v2",
            ).run(limit=1)

        request = state["requests"][0]
        review_input = json.loads(request["messages"][1]["content"])
        manifest = json.loads(
            (Path(output_dir) / "manifest.json").read_text(encoding="utf-8")
        )
        assert review_input["span_evidence"] == {
            "column_coordinates": "zero_based_utf8_bytes_end_exclusive",
            "extracted_surface": "value = 1",
            "line_coordinates": "one_based",
        }
        assert manifest["review_protocol"] == "span-evidence-v2"


def test_contract_protocol_supplies_common_core_and_scope_boundary_invariants():
    feature = _feature()
    with (
        tempfile.TemporaryDirectory() as queue_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")}
        ) as (base_url, state):
            feature.DualLocalSemanticReviewer(
                queue_dir,
                output_dir,
                _reviewers(feature, base_url),
                review_protocol="contract-aware-v3",
            ).run(limit=1)

        request = state["requests"][0]
        review_input = json.loads(request["messages"][1]["content"])
        contract = review_input["semantic_contract"]
        system_prompt = request["messages"][0]["content"]
        assert contract["event_ir_kinds"]["REFERENCE"] == ["variable"]
        assert contract["event_ir_kinds"]["SCOPE_EXIT"] == ["extension"]
        assert contract["zero_width_boundary_events"] == [
            "SCOPE_ENTER",
            "SCOPE_EXIT",
        ]
        assert contract["scope_root"] == "module"
        assert len(contract["contract_sha256"]) == 64
        assert "canonical_record.surface is the claim" in system_prompt
        assert "zero-width span is intentional" in system_prompt


def test_tampered_response_artifact_fails_closed_on_resume():
    feature = _feature()
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")}
        ) as (base_url, _state):
            runner = feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            )
            runner.run()
            response_path = next((Path(output_dir) / "responses").glob("*.json"))
            response_path.write_text("{}\n", encoding="utf-8")
            with pytest.raises(ValueError, match="response artifact SHA-256"):
                runner.run()


def test_response_model_mismatch_is_not_recorded():
    feature = _feature()
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")},
            response_models={"model-a": "unexpected-model"},
        ) as (base_url, _state):
            runner = feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            )
            with pytest.raises(ValueError, match="model identity"):
                runner.run()

        ledger = Path(output_dir) / "reviewer_decisions.jsonl"
        assert not ledger.exists() or ledger.read_text(encoding="utf-8") == ""


def test_malformed_response_is_preserved_in_hashed_failure_ledger():
    feature = _feature()
    malformed = _decision("keep")
    malformed["rubric"]["scope_plausible"] = "true"
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": malformed, "model-b": _decision("keep")}
        ) as (base_url, _state):
            with pytest.raises(ValueError, match="rubric is invalid"):
                feature.DualLocalSemanticReviewer(
                    queue_dir, output_dir, _reviewers(feature, base_url)
                ).run(limit=1)

        failure = json.loads(
            (Path(output_dir) / "reviewer_failures.jsonl").read_text(encoding="utf-8")
        )
        assert failure["failure_type"] == "invalid_response"
        assert failure["review_id"] == "1" * 64
        assert failure["reviewer_id"] == "reviewer-a"
        assert failure["failure_sha256"]
        response = Path(output_dir) / failure["response_file"]
        assert hashlib.sha256(response.read_bytes()).hexdigest() == failure["response_sha256"]
        decisions = Path(output_dir) / "reviewer_decisions.jsonl"
        assert not decisions.exists() or decisions.read_text(encoding="utf-8") == ""


def test_transport_failure_is_preserved_without_fabricated_response_artifact():
    feature = _feature()
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        reviewers = (
            feature.LocalReviewerSpec(
                "reviewer-a", "http://127.0.0.1:1", "model-a", timeout_seconds=1
            ),
            feature.LocalReviewerSpec(
                "reviewer-b", "http://127.0.0.1:2", "model-b", timeout_seconds=1
            ),
        )
        result = feature.DualLocalSemanticReviewer(
            queue_dir, output_dir, reviewers
        ).run(limit=1, continue_on_failure=True)

        failure = json.loads(
            (Path(output_dir) / "reviewer_failures.jsonl").read_text(
                encoding="utf-8"
            )
        )
        assert result.failed_attempts == 1
        assert failure["failure_type"] == "request_failed"
        assert failure["request_kind"] == "initial"
        assert failure["response_sha256"] is None
        assert failure["response_file"] is None
        assert failure["response_model"] is None
        assert not (Path(output_dir) / "responses").exists()


def test_continue_on_failure_runs_independent_reviewer_and_reports_attempt():
    feature = _feature()
    malformed = _decision("keep")
    malformed["rubric"]["uncertainty_calibrated"] = "medium"
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": malformed, "model-b": _decision("keep")}
        ) as (base_url, state):
            result = feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            ).run(continue_on_failure=True)

        assert state["calls"] == ["model-a", "model-b"]
        assert result.completed_reviews == 1
        assert result.failed_attempts == 1
        manifest = json.loads((Path(output_dir) / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["failed_attempts"] == 1
        assert manifest["failure_ledger_sha256"] == hashlib.sha256(
            (Path(output_dir) / "reviewer_failures.jsonl").read_bytes()
        ).hexdigest()
        assert manifest["review_complete"] is False


def test_failed_pair_gets_one_hashed_repair_request_on_resume():
    feature = _feature()
    contradictory = _decision("reject")
    contradictory["decision"] = "keep"
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {
                "model-a": [contradictory, _decision("reject")],
                "model-b": _decision("keep"),
            }
        ) as (base_url, state):
            runner = feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            )
            first = runner.run(limit=1, continue_on_failure=True)
            second = runner.run(limit=1, continue_on_failure=True)
            verified = runner.verify()

        assert first.completed_reviews == 0
        assert second.completed_reviews == 1
        assert verified.completed_reviews == 1
        assert state["calls"] == ["model-a", "model-a"]
        assert len(state["requests"][0]["messages"]) == 2
        assert len(state["requests"][1]["messages"]) == 3
        repair_prompt = state["requests"][1]["messages"][-1]["content"]
        assert "failed deterministic validation" in repair_prompt
        assert "keep decision conflicts with rubric" in repair_prompt
        entry = json.loads(
            (Path(output_dir) / "reviewer_decisions.jsonl").read_text(encoding="utf-8")
        )
        assert entry["request_kind"] == "repair"
        assert entry["prior_failure_sha256"]


def test_exhausted_repair_becomes_manual_terminal_outcome():
    feature = _feature()
    contradictory = _decision("reject")
    contradictory["decision"] = "keep"
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {
                "model-a": [contradictory, contradictory],
                "model-b": _decision("keep"),
            }
        ) as (base_url, state):
            runner = feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            )
            runner.run(limit=1, continue_on_failure=True)
            runner.run(limit=1, continue_on_failure=True)
            terminal = runner.run(continue_on_failure=True)
            verified = runner.verify()

        assert state["calls"] == ["model-a", "model-a", "model-b"]
        assert terminal.completed_reviews == 1
        assert terminal.review_complete is True
        assert verified.review_complete is True
        assert terminal.consensus_counts == {
            "agree_keep": 0,
            "agree_reject": 0,
            "manual_review": 1,
            "pending": 0,
        }
        manifest = json.loads(
            (Path(output_dir) / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["terminal_reviewer_outcomes"] == 2
        assert manifest["exhausted_repair_failures"] == 1
        manual = json.loads(
            (Path(output_dir) / "manual_review_queue.jsonl").read_text(
                encoding="utf-8"
            )
        )
        assert manual["reason"] == "reviewer_invalid_after_repair"
        assert len(manual["reviewer_failures"]) == 1


def test_uncertain_requires_null_rubric_field_and_rejects_false_field():
    feature = _feature()
    uncertain = _decision("keep")
    uncertain["decision"] = "uncertain"
    uncertain["notes"] = "The supplied context cannot establish lexical ownership."
    uncertain["rubric"]["scope_plausible"] = None
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": uncertain, "model-b": _decision("keep")}
        ) as (base_url, _state):
            result = feature.DualLocalSemanticReviewer(
                queue_dir, output_dir, _reviewers(feature, base_url)
            ).run(limit=1)
        assert result.completed_reviews == 1
        entry = json.loads(
            (Path(output_dir) / "reviewer_decisions.jsonl").read_text(encoding="utf-8")
        )
        assert entry["decision"] == "uncertain"
        assert entry["rubric"]["scope_plausible"] is None

    contradictory = _decision("reject")
    contradictory["decision"] = "uncertain"
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": contradictory, "model-b": _decision("keep")}
        ) as (base_url, _state):
            with pytest.raises(ValueError, match="uncertain decision conflicts"):
                feature.DualLocalSemanticReviewer(
                    queue_dir, output_dir, _reviewers(feature, base_url)
                ).run(limit=1)


def test_cli_runs_bounded_dual_review_and_reports_resume_state():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "run_local_semantic_review.py"
    with tempfile.TemporaryDirectory() as queue_dir, tempfile.TemporaryDirectory() as output_dir:
        _queue(Path(queue_dir))
        with _review_server(
            {"model-a": _decision("keep"), "model-b": _decision("keep")}
        ) as (base_url, state):
            command = [
                sys.executable,
                str(script),
                "--queue",
                queue_dir,
                "--output",
                output_dir,
                "--reviewer-a-id",
                "reviewer-a",
                "--reviewer-a-url",
                base_url,
                "--reviewer-a-model",
                "model-a",
                "--reviewer-b-id",
                "reviewer-b",
                "--reviewer-b-url",
                base_url,
                "--reviewer-b-model",
                "model-b",
                "--timeout-seconds",
                "5",
                "--limit",
                "1",
            ]
            first = subprocess.run(
                command,
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            second = subprocess.run(
                command,
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )

        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert state["calls"] == ["model-a", "model-b"]
        first_report = json.loads(first.stdout)
        second_report = json.loads(second.stdout)
        assert first_report["completed_reviews"] == 1
        assert first_report["review_complete"] is False
        assert second_report["completed_reviews"] == 2
        assert second_report["review_complete"] is True
