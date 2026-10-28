from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapters.python import PythonAdapter
from .benchmark import benchmark_project, benchmark_source, trace_comparison
from .meaning_eval import evaluate_meaning_file
from .model_eval import OllamaChatModel, OpenAICompatibleChatModel, evaluate_model_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bcnlt", description="BCNLT Python v0.1 prototype")
    parser.add_argument("file", help="Python file or directory to analyze")
    parser.add_argument("--jsonl", help="Path to write contextual unit JSONL")
    parser.add_argument("--trace", help="Path to write a human-readable trace")
    parser.add_argument("--benchmark", action="store_true", help="Print benchmark summary")
    parser.add_argument("--compare-report", help="Path to write BCNLT vs standard JSON report")
    parser.add_argument("--meaning-report", help="Path to write deterministic meaning retrieval report")
    parser.add_argument("--meaning-max-probes", type=int, default=50)
    parser.add_argument("--model-eval-report", help="Path to write context-answer model eval report")
    parser.add_argument("--model-eval-max-probes", type=int, default=50)
    parser.add_argument(
        "--model-eval-provider",
        choices=["oracle", "openai", "ollama"],
        default="oracle",
        help="Model provider for context-answer eval",
    )
    parser.add_argument("--model-eval-model", default="gpt-4.1-mini")
    parser.add_argument("--model-eval-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model-eval-ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--trace-comparison", help="Path to write side-by-side tokenizer trace")
    parser.add_argument(
        "--wildcard-memory",
        help="Append UNKNOWN_SEMANTIC resolutions to this JSONL memory and reuse prior sightings",
    )
    parser.add_argument("--workers", type=int, default=1, help="Workers for project comparison")
    parser.add_argument("--confidence-threshold", type=float, default=0.72)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.file)

    if args.compare_report:
        if path.is_dir():
            report = benchmark_project(
                path,
                workers=args.workers,
                confidence_threshold=args.confidence_threshold,
            )
            payload = report.to_dict()
        else:
            source = path.read_text(encoding="utf-8")
            payload = benchmark_source(
                source,
                str(path),
                confidence_threshold=args.confidence_threshold,
            ).to_dict()
        Path(args.compare_report).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"BCNLT comparison report written to {args.compare_report}")
        if path.is_dir() and not (args.jsonl or args.trace or args.trace_comparison or args.benchmark):
            return 0

    if path.is_dir():
        if args.benchmark:
            report = benchmark_project(
                path,
                workers=args.workers,
                confidence_threshold=args.confidence_threshold,
            )
            print(json.dumps(report.to_dict(), sort_keys=True))
            return 0
        raise SystemExit("Directory input requires --compare-report or --benchmark")

    if args.meaning_report:
        report = evaluate_meaning_file(path, max_probes=args.meaning_max_probes)
        Path(args.meaning_report).write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"BCNLT meaning report written to {args.meaning_report}")

    if args.model_eval_report:
        model = None
        if args.model_eval_provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise SystemExit("OPENAI_API_KEY is required for --model-eval-provider openai")
            model = OpenAICompatibleChatModel(
                api_key=api_key,
                model=args.model_eval_model,
                base_url=args.model_eval_base_url,
            )
        elif args.model_eval_provider == "ollama":
            model = OllamaChatModel(
                model=args.model_eval_model,
                base_url=args.model_eval_ollama_url,
            )
        report = evaluate_model_file(path, max_probes=args.model_eval_max_probes, model=model)
        Path(args.model_eval_report).write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"BCNLT model-context eval report written to {args.model_eval_report}")

    source = path.read_text(encoding="utf-8")
    adapter = PythonAdapter(wildcard_memory_path=args.wildcard_memory)
    document = adapter.parse(source, path=str(path))
    extraction = adapter.extract(document)
    units = list(extraction.units)
    assessments = list(extraction.assessments)

    if args.jsonl:
        with Path(args.jsonl).open("w", encoding="utf-8") as handle:
            for unit, assessment in zip(units, assessments):
                score = assessment.final_score
                payload = unit.to_dict()
                payload["uncertainty"] = score.uncertainty
                payload["confidence"] = score.confidence
                payload["uncertainty_signals"] = list(score.signals)
                if assessment.initial_score.confidence < args.confidence_threshold:
                    payload["uyr"] = _uyr_payload(assessment)
                if assessment.wildcard_category is not None:
                    payload["wildcard"] = {
                        "category": assessment.wildcard_category,
                        "state": assessment.state.value,
                        "evidence": list(assessment.evidence),
                    }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    if args.trace:
        Path(args.trace).write_text(_trace(units), encoding="utf-8")

    if args.trace_comparison:
        Path(args.trace_comparison).write_text(
            trace_comparison(source, str(path)),
            encoding="utf-8",
        )

    print(f"BCNLT emitted {len(units)} contextual units from {path}")
    if args.benchmark:
        print(
            json.dumps(
                benchmark_source(
                    source,
                    str(path),
                    confidence_threshold=args.confidence_threshold,
                ).to_dict(),
                sort_keys=True,
            )
        )
    return 0


def _uyr_payload(result) -> dict[str, object]:
    return {
        "initial_confidence": result.initial_score.confidence,
        "final_confidence": result.final_score.confidence,
        "rounds": result.rounds,
        "retrievals": list(result.evidence),
    }


def _trace(units) -> str:
    lines = []
    for unit in units:
        link = f" -> {unit.defines}" if unit.defines else ""
        symbol = f" [{unit.symbol_id}]" if unit.symbol_id else ""
        span = f"{unit.span[0]}:{unit.span[1]}"
        lines.append(f"{span} {unit.event} {unit.surface}{symbol}{link} ({unit.scope})")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
