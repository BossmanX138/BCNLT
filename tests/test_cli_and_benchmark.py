import json
import subprocess
import sys
from pathlib import Path

from bcnlt.benchmark import benchmark_project, benchmark_source, trace_comparison


def test_cli_outputs_jsonl_and_trace(tmp_path: Path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import os\n\ndef main(name):\n    value = os.path.join('x', name)\n    return value\n",
        encoding="utf-8",
    )

    jsonl = tmp_path / "units.jsonl"
    trace = tmp_path / "trace.txt"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bcnlt.cli",
            str(sample),
            "--jsonl",
            str(jsonl),
            "--trace",
            str(trace),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "BCNLT emitted" in result.stdout
    first = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert {"surface", "event", "scope", "span", "recent_history"}.issubset(first)
    assert "IMPORT os" in trace.read_text(encoding="utf-8")


def test_cli_appends_wildcard_memory(tmp_path: Path):
    sample = tmp_path / "ambiguous.py"
    sample.write_text(
        "payload = {'kind': 'demo'}\n"
        "match payload:\n"
        "    case {'kind': kind}:\n"
        "        outcome = kind\n",
        encoding="utf-8",
    )
    jsonl = tmp_path / "units.jsonl"
    memory = tmp_path / "wildcards.jsonl"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "bcnlt.cli",
            str(sample),
            "--jsonl",
            str(jsonl),
            "--wildcard-memory",
            str(memory),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    records = [json.loads(line) for line in memory.read_text(encoding="utf-8").splitlines()]
    assert records
    assert records[0]["category"] == "pattern_matching"
    assert records[0]["seen_count"] == 1


def test_benchmark_reports_lexical_tokens_events_and_latency():
    report = benchmark_source("x = 1\nprint(x)\n", path="bench.py")

    assert report.standard_token_count >= 4
    assert report.standard_tokenizer_name
    assert report.lexical_token_count >= 4
    assert report.event_count >= 4
    assert report.event_to_standard_ratio > 0
    assert report.elapsed_ms >= 0
    assert report.path == "bench.py"


def test_trace_comparison_shows_standard_tokens_and_semantic_events():
    comparison = trace_comparison("def f(x):\n    return x + 1\n", path="trace.py")

    assert "Standard tokens" in comparison
    assert "BCNLT semantic events" in comparison
    assert "DEFINE f" in comparison
    assert "RETURN return" in comparison


def test_project_benchmark_aggregates_python_files_and_hotspots(tmp_path: Path):
    (tmp_path / "a.py").write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    (tmp_path / "ambiguous.py").write_text(
        "def run(plugin, payload):\n"
        "    handler = getattr(plugin, payload['name'])\n"
        "    return handler(payload)\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("not python", encoding="utf-8")

    report = benchmark_project(tmp_path, workers=1)

    assert report.file_count == 2
    assert report.total_standard_token_count > 0
    assert report.total_event_count > 0
    assert report.total_uncertain_events > 0
    assert report.total_uyr_activations > 0
    assert report.files[0].path.endswith(".py")


def test_cli_writes_comparison_report_for_directory(tmp_path: Path):
    (tmp_path / "sample.py").write_text("x = 1\nprint(x)\n", encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bcnlt.cli",
            str(tmp_path),
            "--compare-report",
            str(report_path),
            "--workers",
            "1",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "BCNLT comparison report" in result.stdout
    assert payload["file_count"] == 1
    assert payload["total_standard_token_count"] > 0
    assert payload["total_event_count"] > 0
