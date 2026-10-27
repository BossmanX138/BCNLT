from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bcnlt.model_eval import OllamaChatModel, _answer_matches_line, evaluate_model_file


@dataclass
class TimedModel:
    model: OllamaChatModel
    latencies: list[float] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.model.name

    def answer(self, question: str, context: str) -> str:
        started = time.perf_counter()
        try:
            return self.model.answer(question, context)
        finally:
            self.latencies.append(time.perf_counter() - started)


def discover_files(root: Path, max_files: int) -> list[Path]:
    candidates = sorted(
        path
        for path in root.rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    )
    return candidates[:max_files]


def readiness(model: OllamaChatModel) -> tuple[str, bool]:
    answer = model.answer(
        "What line defines `thing`?",
        "L7: DEFINE thing scope=module symbol=module.thing",
    ).strip()
    if not answer:
        raise RuntimeError(f"{model.name} returned an empty readiness response")
    return answer, _answer_matches_line(answer, 7)


def unload(base_url: str, model: str) -> None:
    payload = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()


def evaluate_one(
    model_id: str,
    files: list[Path],
    base_url: str,
    max_probes: int,
) -> dict[str, Any]:
    client = OllamaChatModel(
        model=model_id,
        base_url=base_url,
        timeout=180,
        max_output_tokens=32,
    )
    ready_answer, ready_correct = readiness(client)
    timed = TimedModel(client)
    reports = []
    for index, path in enumerate(files, start=1):
        report = evaluate_model_file(path, max_probes=max_probes, model=timed)
        reports.append(report)
        print(
            f"[{model_id}] {index}/{len(files)} {path.name}: "
            f"raw={report.standard_correct_count}/{report.probe_count}, "
            f"bcnlt={report.bcnlt_correct_count}/{report.probe_count}",
            flush=True,
        )
    probe_count = sum(report.probe_count for report in reports)
    standard_correct = sum(report.standard_correct_count for report in reports)
    bcnlt_correct = sum(report.bcnlt_correct_count for report in reports)
    return {
        "model_id": model_id,
        "readiness_answer": ready_answer,
        "readiness_correct": ready_correct,
        "probe_count": probe_count,
        "request_count": len(timed.latencies),
        "standard_correct": standard_correct,
        "standard_accuracy": standard_correct / probe_count if probe_count else 0.0,
        "bcnlt_correct": bcnlt_correct,
        "bcnlt_accuracy": bcnlt_correct / probe_count if probe_count else 0.0,
        "bcnlt_gain": (bcnlt_correct - standard_correct) / probe_count if probe_count else 0.0,
        "latency_seconds": {
            "total": sum(timed.latencies),
            "mean": statistics.mean(timed.latencies) if timed.latencies else 0.0,
            "median": statistics.median(timed.latencies) if timed.latencies else 0.0,
        },
        "files": [report.to_dict() for report in reports],
    }


def paired_outcomes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    counts = {"both_correct": 0, "wizard_only": 0, "runeforge_only": 0, "both_wrong": 0}
    left_results = [result for report in left["files"] for result in report["results"]]
    right_results = [result for report in right["files"] for result in report["results"]]
    if len(left_results) != len(right_results):
        raise RuntimeError("Head-to-head reports have different probe counts")
    for left_result, right_result in zip(left_results, right_results, strict=True):
        left_correct = bool(left_result["bcnlt_correct"])
        right_correct = bool(right_result["bcnlt_correct"])
        if left_correct and right_correct:
            counts["both_correct"] += 1
        elif left_correct:
            counts["wizard_only"] += 1
        elif right_correct:
            counts["runeforge_only"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def render_markdown(payload: dict[str, Any]) -> str:
    wizard = payload["models"]["wizard_phi_v02"]
    runeforge = payload["models"]["runeforge_mk2"]
    paired = payload["paired_bcnlt_outcomes"]
    return "\n".join(
        [
            "# BCNLT head-to-head: Wizard-Phi v0.2 vs Runeforge Mk2 7B",
            "",
            f"Files: {payload['file_count']}  ",
            f"Definition probes: {payload['probe_count']}  ",
            f"Winner by BCNLT-context accuracy: **{payload['winner']}**",
            (
                "Readiness probe: "
                f"Wizard-Phi **{'pass' if wizard['readiness_correct'] else 'fail'}**, "
                f"Runeforge **{'pass' if runeforge['readiness_correct'] else 'fail'}**"
            ),
            "",
            "| Model | Raw context | BCNLT context | BCNLT gain | Mean request latency |",
            "|---|---:|---:|---:|---:|",
            (
                f"| Wizard-Phi v0.2 | {wizard['standard_accuracy']:.1%} | "
                f"{wizard['bcnlt_accuracy']:.1%} | {wizard['bcnlt_gain']:+.1%} | "
                f"{wizard['latency_seconds']['mean']:.3f}s |"
            ),
            (
                f"| Runeforge Mk2 7B | {runeforge['standard_accuracy']:.1%} | "
                f"{runeforge['bcnlt_accuracy']:.1%} | {runeforge['bcnlt_gain']:+.1%} | "
                f"{runeforge['latency_seconds']['mean']:.3f}s |"
            ),
            "",
            "## Paired BCNLT outcomes",
            "",
            f"- Both correct: {paired['both_correct']}",
            f"- Wizard-Phi only: {paired['wizard_only']}",
            f"- Runeforge only: {paired['runeforge_only']}",
            f"- Both wrong: {paired['both_wrong']}",
            "",
            "This benchmark measures definition-line extraction from raw and BCNLT semantic contexts. "
            "It does not establish general coding quality.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("bcnlt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/head-to-head-wizard-v02-vs-runeforge-mk2"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--wizard-model", default="wizard-phi-3.5:v0.2")
    parser.add_argument("--runeforge-model", default="runeforge-mk2:7b")
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-probes-per-file", type=int, default=6)
    args = parser.parse_args()

    files = discover_files(args.input.resolve(), args.max_files)
    if not files:
        raise SystemExit(f"No Python files found under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    wizard = evaluate_one(args.wizard_model, files, args.base_url, args.max_probes_per_file)
    unload(args.base_url, args.wizard_model)
    runeforge = evaluate_one(args.runeforge_model, files, args.base_url, args.max_probes_per_file)
    unload(args.base_url, args.runeforge_model)
    paired = paired_outcomes(wizard, runeforge)
    if wizard["bcnlt_accuracy"] > runeforge["bcnlt_accuracy"]:
        winner = "Wizard-Phi v0.2"
    elif runeforge["bcnlt_accuracy"] > wizard["bcnlt_accuracy"]:
        winner = "Runeforge Mk2 7B"
    else:
        winner = "Tie"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "input": str(args.input.resolve()),
        "files": [str(path) for path in files],
        "file_count": len(files),
        "max_probes_per_file": args.max_probes_per_file,
        "probe_count": wizard["probe_count"],
        "models": {"wizard_phi_v02": wizard, "runeforge_mk2": runeforge},
        "paired_bcnlt_outcomes": paired,
        "winner": winner,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = render_markdown(payload)
    (args.output / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

