"""Compare the frozen model BCNLT runtime with a read-only BCNLM candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from bosscrafts_model.jsonl import split_jsonl_records

PROTOCOL = "bosscrafts-bcnlt-source-sidecar-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    *, python: Path, executable: Path, language: str, source: str, logical_path: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    request = {
        "protocol": PROTOCOL,
        "source": source,
        "language": language,
        "path": logical_path,
        "dialect": None,
    }
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(python), "-I", "-S", "-B", "-X", "utf8", str(executable)],
        input=json.dumps(request),
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=environment,
        cwd=executable.parent,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{executable} rejected {logical_path}: {completed.stderr.strip()[:1000]}"
        )
    response = json.loads(completed.stdout)
    canonical = response["canonical_jsonl"]
    records = [json.loads(line) for line in split_jsonl_records(canonical)]
    return response, records


def _definition(records: list[dict[str, object]], surface: str) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if record.get("event") == "DEFINE" and record.get("surface") == surface
    ]


def _references(records: list[dict[str, object]], surface: str) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if record.get("event") == "REFERENCE" and record.get("surface") == surface
    ]


def _plain_parameters(records: list[dict[str, object]]) -> bool:
    for name in ("left", "right"):
        definitions = _definition(records, name)
        references = _references(records, name)
        if len(definitions) != 1 or not references:
            return False
        if any(record.get("defines") != definitions[0].get("symbol_id") for record in references):
            return False
    return True


def _arrow_body(records: list[dict[str, object]]) -> bool:
    return bool(
        _definition(records, "double")
        and any(
            record.get("event") == "RETURN"
            and "return result" in str(record.get("surface"))
            and "double" in str(record.get("scope"))
            for record in records
        )
    )


def _rust_tail(records: list[dict[str, object]]) -> bool:
    return any(
        record.get("event") == "RETURN" and "x*2" in str(record.get("surface")).replace(" ", "")
        for record in records
    )


def _python_class_ownership(records: list[dict[str, object]]) -> bool:
    definitions = _definition(records, "value")
    return [record.get("scope") for record in definitions] == ["module.A", "module.B"]


def _c_shadowing(records: list[dict[str, object]]) -> bool:
    definitions = _definition(records, "y")
    references = _references(records, "y")
    return (
        len(definitions) == 2
        and definitions[0].get("scope") != definitions[1].get("scope")
        and [record.get("defines") for record in references]
        == [definitions[1].get("symbol_id"), definitions[0].get("symbol_id")]
    )


def _go_multi_binding(records: list[dict[str, object]]) -> bool:
    definitions = {
        record.get("surface"): record
        for record in records
        if record.get("event") == "DEFINE"
    }
    if not {"a", "b"}.issubset(definitions) or "a,b" in definitions:
        return False
    for name in ("a", "b"):
        references = _references(records, name)
        if not references or any(
            record.get("defines") != definitions[name].get("symbol_id")
            for record in references
        ):
            return False
    return True


CASES: tuple[tuple[str, str, str, Callable[[list[dict[str, object]]], bool]], ...] = (
    (
        "javascript_plain_parameters",
        "javascript",
        "function combine(left, right) { return left + right; }\n",
        _plain_parameters,
    ),
    (
        "ruby_plain_parameters",
        "ruby",
        "def combine(left, right)\n return left + right\nend\n",
        _plain_parameters,
    ),
    (
        "javascript_arrow_body_ownership",
        "javascript",
        "const double = x => {\n const result=x*2;\n return result;\n};\n",
        _arrow_body,
    ),
    (
        "rust_implicit_tail_return",
        "rust",
        "fn double(x:i32)->i32 {\n x*2\n}\n",
        _rust_tail,
    ),
    (
        "python_class_method_ownership",
        "python",
        "class A:\n    def value(self):\n        return 1\nclass B:\n    def value(self):\n        return 2\n",
        _python_class_ownership,
    ),
    (
        "c_block_shadowing",
        "c",
        "int f(int x) {\n int y=x;\n { int y=9; use(y); }\n return y;\n}\n",
        _c_shadowing,
    ),
    (
        "cpp_block_shadowing",
        "cpp",
        "int f(int x) {\n int y=x;\n { int y=9; use(y); }\n return y;\n}\n",
        _c_shadowing,
    ),
    (
        "go_multi_binding",
        "go",
        "package test\nfunc pair() int { a,b := 1,2; return a+b }\n",
        _go_multi_binding,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--model-sidecar", required=True, type=Path)
    parser.add_argument("--candidate-worker", required=True, type=Path)
    parser.add_argument("--model-runtime-manifest", required=True, type=Path)
    parser.add_argument("--candidate-runtime-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[dict[str, object]] = []
    runtime_specs: dict[str, dict[str, str]] = {"model": {}, "candidate": {}}
    for case_name, language, source, evaluator in CASES:
        row: dict[str, object] = {"case": case_name, "language": language}
        for label, executable in (
            ("model", args.model_sidecar),
            ("candidate", args.candidate_worker),
        ):
            response, records = _run(
                python=args.python,
                executable=executable,
                language=language,
                source=source,
                logical_path=f"{case_name}.{language}",
            )
            runtime_specs[label][language] = str(response["semantic_spec_sha256"])
            row[label] = {
                "passes": evaluator(records),
                "record_count": len(records),
                "semantic_spec_sha256": response["semantic_spec_sha256"],
                "relevant_records": [
                    {
                        key: record.get(key)
                        for key in ("event", "surface", "scope", "symbol_id", "defines", "ast_relation")
                    }
                    for record in records
                    if record.get("event") in {"DEFINE", "REFERENCE", "RETURN"}
                ],
            }
        results.append(row)
    model_passes = sum(bool(row["model"]["passes"]) for row in results)  # type: ignore[index]
    candidate_passes = sum(bool(row["candidate"]["passes"]) for row in results)  # type: ignore[index]
    report = {
        "schema_version": "bosscrafts-bcnlm-runtime-differential-v1",
        "read_only_comparison": True,
        "model_runtime": {
            "sidecar": str(args.model_sidecar.resolve()),
            "sidecar_sha256": _sha256(args.model_sidecar),
            "manifest": str(args.model_runtime_manifest.resolve()),
            "manifest_sha256": _sha256(args.model_runtime_manifest),
            "passed_cases": model_passes,
        },
        "candidate_runtime": {
            "worker": str(args.candidate_worker.resolve()),
            "worker_sha256": _sha256(args.candidate_worker),
            "manifest": str(args.candidate_runtime_manifest.resolve()),
            "manifest_sha256": _sha256(args.candidate_runtime_manifest),
            "passed_cases": candidate_passes,
        },
        "case_count": len(results),
        "results": results,
        "runtime_spec_hashes": runtime_specs,
        "candidate_improves_all_targeted_cases": (
            candidate_passes == len(results) and candidate_passes > model_passes
        ),
        "promotion_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "case_count": len(results),
                "model_passes": model_passes,
                "candidate_passes": candidate_passes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
