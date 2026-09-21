import json
import subprocess
import sys
from pathlib import Path

from bcnlt.meaning_eval import evaluate_meaning_source


SOURCE = """
def helper(value):
    return value + 1


def caller(items):
    total = 0
    for value in items:
        total = helper(total + value)
    return total
"""


def test_meaning_eval_compares_standard_text_and_bcnlt_semantic_retrieval():
    report = evaluate_meaning_source(SOURCE, path="meaning.py", max_probes=10)

    assert report.path == "meaning.py"
    assert report.probe_count >= 3
    assert report.bcnlt_hit_count >= report.standard_hit_count
    assert report.bcnlt_accuracy >= report.standard_accuracy
    assert any(result.standard_hit is False for result in report.results)
    assert all(result.bcnlt_hit for result in report.results)


def test_cli_writes_meaning_eval_report(tmp_path: Path):
    sample = tmp_path / "meaning.py"
    sample.write_text(SOURCE, encoding="utf-8")
    report_path = tmp_path / "meaning-report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bcnlt.cli",
            str(sample),
            "--meaning-report",
            str(report_path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "BCNLT meaning report" in result.stdout
    assert payload["probe_count"] >= 3
    assert payload["bcnlt_accuracy"] >= payload["standard_accuracy"]
