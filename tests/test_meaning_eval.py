import json
import subprocess
import sys
from pathlib import Path

from bcnlt.core import ContextualUnit
from bcnlt.meaning_eval import (
    _evaluate_probe,
    _generate_definition_probes,
    evaluate_meaning_source,
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


def test_meaning_eval_compares_standard_text_and_bcnlt_semantic_retrieval():
    report = evaluate_meaning_source(SOURCE, path="meaning.py", max_probes=10)

    assert report.path == "meaning.py"
    assert report.probe_count >= 3
    assert report.bcnlt_hit_count >= report.standard_hit_count
    assert report.bcnlt_accuracy >= report.standard_accuracy
    assert any(result.standard_hit is False for result in report.results)
    assert all(result.bcnlt_hit for result in report.results)
    assert report.language == "python"
    assert len(report.semantic_spec_sha256) == 64


def test_meaning_eval_discloses_extractor_self_consistency_not_semantic_correctness():
    source = 'print("hello")\n'
    fabricated_units = [
        ContextualUnit(
            surface="fabricated_answer",
            event="DEFINE",
            scope="module",
            span=(1, 0, 1, 14),
            symbol_id="module.fabricated_answer",
        )
    ]
    probe = _generate_definition_probes(fabricated_units, max_probes=1)[0]

    result = _evaluate_probe(
        source,
        fabricated_units,
        probe,
        standard_context_lines=1,
        bcnlt_context_units=0,
    )
    payload = evaluate_meaning_source(SOURCE).to_dict()

    assert result.standard_hit is False
    assert result.bcnlt_hit is True
    assert payload["evaluation_kind"] == "extractor_self_consistency_retrieval"
    assert payload["target_origin"] == "same_extractor_output"
    assert payload["independent_semantic_oracle"] is False
    assert payload["promotion_eligible"] is False
    assert payload["bcnlt_retrieval_hit_rate"] == payload["bcnlt_accuracy"]


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
            "--language",
            "python",
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
    assert payload["language"] == "python"
