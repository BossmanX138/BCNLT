import json
import subprocess
import sys
from pathlib import Path

from bcnlt.model_eval import (
    OllamaChatModel,
    OpenAICompatibleChatModel,
    OracleDefinitionLineModel,
    _answer_matches_line,
    evaluate_model_source,
)


SOURCE = """
def helper(value):
    return value + 1


def caller(items):
    total = 0
    for value in items:
        total = helper(total + value)
    return total
"""


def test_model_eval_answers_definition_questions_from_contexts():
    report = evaluate_model_source(
        SOURCE,
        path="model_eval.py",
        model=OracleDefinitionLineModel(),
        max_probes=10,
    )

    assert report.probe_count >= 3
    assert report.bcnlt_correct_count >= report.standard_correct_count
    assert report.bcnlt_accuracy >= report.standard_accuracy
    assert report.bcnlt_accuracy == 1.0
    assert any(result.standard_correct is False for result in report.results)


def test_cli_writes_model_eval_report(tmp_path: Path):
    sample = tmp_path / "model_eval.py"
    sample.write_text(SOURCE, encoding="utf-8")
    report_path = tmp_path / "model-report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bcnlt.cli",
            str(sample),
            "--model-eval-report",
            str(report_path),
            "--model-eval-max-probes",
            "10",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "BCNLT model-context eval report" in result.stdout
    assert payload["model_name"] == "oracle-definition-line"
    assert payload["bcnlt_accuracy"] >= payload["standard_accuracy"]


def test_cli_rejects_openai_provider_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sample = tmp_path / "model_eval.py"
    sample.write_text(SOURCE, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bcnlt.cli",
            str(sample),
            "--model-eval-report",
            str(tmp_path / "model-report.json"),
            "--model-eval-provider",
            "openai",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "OPENAI_API_KEY" in result.stderr


def test_cli_accepts_ollama_provider_flag(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bcnlt.cli",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ollama" in result.stdout
    assert "--model-eval-ollama-url" in result.stdout


def test_ollama_model_uses_local_chat_transport():
    calls = []

    def fake_transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return {"message": {"content": "line 7"}}

    model = OllamaChatModel(
        model="qwen2.5-coder:latest",
        base_url="http://127.0.0.1:11434",
        transport=fake_transport,
    )

    answer = model.answer("What line defines `thing`?", "L7: DEFINE thing scope=module")

    assert answer == "line 7"
    assert model.name == "ollama:qwen2.5-coder:latest"
    assert calls[0][0] == "http://127.0.0.1:11434/api/chat"
    assert calls[0][1]["Content-Type"] == "application/json"
    assert calls[0][2]["model"] == "qwen2.5-coder:latest"
    assert calls[0][2]["stream"] is False
    assert calls[0][2]["options"]["num_predict"] == 64
    assert "L7: DEFINE thing" in calls[0][2]["messages"][-1]["content"]


def test_openai_compatible_model_uses_chat_completion_transport():
    calls = []

    def fake_transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return {
            "choices": [
                {
                    "message": {
                        "content": "line 42",
                    }
                }
            ]
        }

    model = OpenAICompatibleChatModel(
        api_key="test-key",
        model="gpt-test",
        base_url="https://example.test/v1",
        transport=fake_transport,
    )

    answer = model.answer("What line defines `thing`?", "L42: DEFINE thing scope=module")

    assert answer == "line 42"
    assert model.name == "openai-compatible:gpt-test"
    assert calls[0][0] == "https://example.test/v1/chat/completions"
    assert calls[0][1]["Authorization"] == "Bearer test-key"
    assert calls[0][2]["model"] == "gpt-test"
    assert calls[0][2]["temperature"] == 0
    assert "L42: DEFINE thing" in calls[0][2]["messages"][-1]["content"]


def test_line_grader_rejects_ambiguous_multi_line_answers():
    assert _answer_matches_line("line 7", 7)
    assert _answer_matches_line("L7: DEFINE thing", 7)
    assert _answer_matches_line("7", 7)
    assert not _answer_matches_line("L7: value\nL8: DEFINE thing\nL9: value", 8)
    assert not _answer_matches_line("the value is 7", 7)
