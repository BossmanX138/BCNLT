from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_code_acceptance import _execution

from bosscrafts_model.category_case_inventory import CATEGORY_CASE_SCHEMA
from bosscrafts_model.category_case_sources import (
    MCEVAL_LANGUAGE_FILES,
    _mceval_harness,
    freeze_aegis_test_cases,
    freeze_aegis_v1_parquet_test_cases,
    freeze_ifeval_cases,
    freeze_mceval_execution_cases,
)
from bosscrafts_model.model_acceptance import MINIMUM_CATEGORY_EXAMPLES

IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
AEGIS_REVISION = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ifeval_source(path: Path) -> None:
    rows = [
        {
            "key": 1000 + index,
            "prompt": f"Write item {index} in lowercase.",
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{}],
        }
        for index in range(MINIMUM_CATEGORY_EXAMPLES["ordinary_instruction"])
    ]
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _aegis_source(path: Path, *, include_redacted: bool = False) -> None:
    count = MINIMUM_CATEGORY_EXAMPLES["safety"] + int(include_redacted)
    rows: list[dict[str, object]] = []
    for index in range(count):
        redacted = include_redacted and index == count - 1
        unsafe = bool(index % 2)
        rows.append(
            {
                "id": hashlib.md5(
                    f"aegis:{index}".encode(), usedforsecurity=False
                ).hexdigest(),
                "reconstruction_id_if_redacted": "external-row" if redacted else None,
                "prompt": "REDACTED" if redacted else f"Safety prompt {index}",
                "response": None,
                "prompt_label": "unsafe" if unsafe else "safe",
                "response_label": None,
                "violated_categories": "Violence" if unsafe else "",
                "prompt_label_source": "human",
                "response_label_source": None,
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")


class _McEvalSandbox:
    def execute(
        self,
        *,
        language: str,
        source: str,
        stdin_text: str,
        mode: str = "program",
    ):
        assert stdin_text == ""
        assert mode == "test"
        return _execution(
            language=language,
            source=source,
            stdin_text="",
            stdout=b"canonical\n" if "CANONICAL" in source else b"broken\n",
            mode=mode,
        )


def _mceval_sources(root: Path, manifest_path: Path) -> None:
    root.mkdir()
    languages: dict[str, dict[str, str]] = {}
    per_language = 37
    for language, filename in MCEVAL_LANGUAGE_FILES.items():
        rows = []
        for index in range(per_language):
            rows.append(
                {
                    "task_id": f"{language}/{index}",
                    "prompt": f"PREFIX {language} {index}\n",
                    "canonical_solution": "CANONICAL\n",
                    "test": (
                        "static void Main() { TESTS; }\n}"
                        if language == "c_sharp"
                        else "TESTS\n"
                    ),
                    "entry_point": f"entry_{index}",
                    "signature": f"entry_{index}()",
                    "docstring": f"Fixture {index}",
                    "instruction": f"Complete fixture {index}",
                    "level": "easy",
                }
            )
        source = root / filename
        source.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        languages[language] = {"file": filename, "sha256": _sha256(source)}
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "bosscrafts-mceval-generation-sources-v1",
                "source": (
                    "hf://datasets/Multilingual-Multimodal-NLP/McEval"
                ),
                "revision": "c4767fec950de7decd487a40a4dab2795df4a094",
                "license": "CC-BY-SA-4.0",
                "attribution": "McEval; Chai et al., 2024",
                "languages": languages,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_freezes_exact_ifeval_revision_as_immutable_content_addressed_cases(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ifeval_input_data.jsonl"
    output = tmp_path / "ordinary_instruction.jsonl"
    _ifeval_source(source)

    result = freeze_ifeval_cases(
        source,
        output,
        revision=IFEVAL_REVISION,
        expected_source_sha256=_sha256(source),
    )

    assert result.record_count == MINIMUM_CATEGORY_EXAMPLES["ordinary_instruction"]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["example_id"] for record in records] == sorted(
        record["example_id"] for record in records
    )
    assert {record["schema_version"] for record in records} == {
        CATEGORY_CASE_SCHEMA
    }
    assert {record["case_kind"] for record in records} == {"ifeval_instruction"}
    assert {record["provenance"]["revision"] for record in records} == {
        IFEVAL_REVISION
    }
    assert {record["provenance"]["license"] for record in records} == {
        "Apache-2.0"
    }
    original = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        freeze_ifeval_cases(
            source,
            output,
            revision=IFEVAL_REVISION,
            expected_source_sha256=_sha256(source),
        )
    assert output.read_bytes() == original


def test_freezes_only_reconstructable_aegis_test_prompts_with_human_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "test.json"
    output = tmp_path / "safety.jsonl"
    _aegis_source(source, include_redacted=True)

    result = freeze_aegis_test_cases(
        source,
        output,
        revision=AEGIS_REVISION,
        expected_source_sha256=_sha256(source),
    )

    assert result.record_count == MINIMUM_CATEGORY_EXAMPLES["safety"]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(record["case"]["split"] == "test" for record in records)
    assert all(
        record["case"]["prompt_label_source"] == "human" for record in records
    )
    assert all(record["case"]["prompt"] != "REDACTED" for record in records)
    assert {record["provenance"]["revision"] for record in records} == {
        AEGIS_REVISION
    }
    assert {record["provenance"]["license"] for record in records} == {
        "CC-BY-4.0"
    }


def test_freezes_unanimous_prompt_only_aegis_v1_parquet_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "aegis-v1-test.parquet"
    rows: list[dict[str, object]] = []
    minimum = MINIMUM_CATEGORY_EXAMPLES["safety"]
    for index in range(minimum):
        label = "Safe" if index % 2 == 0 else "Violence"
        rows.append(
            {
                "num_annotations": 3,
                "id": hashlib.md5(
                    f"aegis-v1:{index}".encode(), usedforsecurity=False
                ).hexdigest(),
                "text": f"Prompt-only safety fixture {index}",
                "text_type": "user_message",
                "labels_0": label,
                "labels_1": label,
                "labels_2": label,
                "labels_3": None,
                "labels_4": None,
            }
        )
    rows.extend(
        (
            {
                "num_annotations": 3,
                "id": hashlib.md5(b"not-prompt", usedforsecurity=False).hexdigest(),
                "text": "Response-only fixture",
                "text_type": "llm_response",
                "labels_0": "Safe",
                "labels_1": "Safe",
                "labels_2": "Safe",
                "labels_3": None,
                "labels_4": None,
            },
            {
                "num_annotations": 3,
                "id": hashlib.md5(b"disagreement", usedforsecurity=False).hexdigest(),
                "text": "Mixed label fixture",
                "text_type": "user_message",
                "labels_0": "Safe",
                "labels_1": "Violence",
                "labels_2": "Safe",
                "labels_3": None,
                "labels_4": None,
            },
            {
                "num_annotations": 3,
                "id": hashlib.md5(b"no-category", usedforsecurity=False).hexdigest(),
                "text": "No category consensus fixture",
                "text_type": "user_message",
                "labels_0": "Violence",
                "labels_1": "Sexual",
                "labels_2": "Threat",
                "labels_3": None,
                "labels_4": None,
            },
        )
    )
    pq.write_table(pa.Table.from_pylist(rows), source)
    output = tmp_path / "safety.jsonl"
    audit = tmp_path / "safety.audit.json"

    result = freeze_aegis_v1_parquet_test_cases(
        source,
        output,
        audit,
        revision=AEGIS_REVISION,
        expected_source_sha256=_sha256(source),
    )

    assert result.record_count == minimum
    assert result.rejected_count == 3
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert {record["case"]["prompt_label"] for record in records} == {
        "safe",
        "unsafe",
    }
    assert all(record["case"]["prompt_label_source"] == "human" for record in records)
    assert all("Prompt-only" in record["case"]["prompt"] for record in records)
    audit_payload = json.loads(audit.read_text())
    assert audit_payload["accepted_label_counts"] == {
        "safe": minimum // 2,
        "unsafe": minimum // 2,
    }
    assert audit_payload["rejected_counts"] == {
        "binary_label_disagreement": 1,
        "no_consensus_violation_category": 1,
        "not_prompt_only": 1,
    }


def test_source_freezers_reject_wrong_digest_duplicates_and_invalid_revision(
    tmp_path: Path,
) -> None:
    ifeval = tmp_path / "ifeval.jsonl"
    _ifeval_source(ifeval)
    with pytest.raises(ValueError, match="SHA-256"):
        freeze_ifeval_cases(
            ifeval,
            tmp_path / "wrong.jsonl",
            revision=IFEVAL_REVISION,
            expected_source_sha256="0" * 64,
        )

    first = json.loads(ifeval.read_text().splitlines()[0])
    ifeval.write_text(
        ifeval.read_text() + json.dumps(first) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        freeze_ifeval_cases(
            ifeval,
            tmp_path / "duplicate.jsonl",
            revision=IFEVAL_REVISION,
            expected_source_sha256=_sha256(ifeval),
        )

    aegis = tmp_path / "aegis.json"
    _aegis_source(aegis)
    with pytest.raises(ValueError, match="revision"):
        freeze_aegis_test_cases(
            aegis,
            tmp_path / "invalid-revision.jsonl",
            revision="main",
            expected_source_sha256=_sha256(aegis),
        )


def test_freezes_only_canonical_pass_negative_fail_mceval_cases(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "generation"
    manifest = tmp_path / "sources.json"
    output = tmp_path / "execution_correctness.jsonl"
    audit = tmp_path / "execution_correctness.audit.json"
    _mceval_sources(source_root, manifest)

    result = freeze_mceval_execution_cases(
        source_root,
        manifest,
        output,
        audit,
        sandbox=_McEvalSandbox(),
        workers=4,
    )

    assert result.record_count == 37 * len(MCEVAL_LANGUAGE_FILES)
    assert result.rejected_count == 0
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == result.record_count
    assert {record["case"]["language"] for record in records} == set(
        MCEVAL_LANGUAGE_FILES
    )
    assert all(
        record["case"]["harness_prefix"]
        == (
            "using System.Linq;\n" + record["case"]["prompt"] + "\n"
            if record["case"]["language"] == "c_sharp"
            else record["case"]["prompt"] + "\n"
        )
        for record in records
    )
    assert all(
        record["case"]["expected_stdout"] == "canonical\n"
        for record in records
    )
    audit_payload = json.loads(audit.read_text())
    assert audit_payload["record_count"] == result.record_count
    assert audit_payload["rejected_count"] == 0
    assert audit_payload["worker_count"] == 4
    assert set(audit_payload["harness_adapters"]) == {"c_sharp", "kotlin"}
    assert len(audit_payload["accepted_evidence"]) == result.record_count
    with pytest.raises(ValueError, match="already exists"):
        freeze_mceval_execution_cases(
            source_root,
            manifest,
            output,
            audit,
            sandbox=_McEvalSandbox(),
        )


def test_mceval_kotlin_harness_is_valid_compiled_source_shape() -> None:
    prefix, suffix = _mceval_harness(
        "kotlin",
        "fun answer() = 42",
        "fun main() { check(answer() == 42) }\n\nmain()",
    )
    assert prefix == "fun answer() = 42\n"
    assert suffix == "\nfun main() { check(answer() == 42) }"

    _, wrapped = _mceval_harness(
        "kotlin",
        "fun answer() = 42",
        "check(answer() == 42)",
    )
    assert wrapped == "\nfun main() {\n    check(answer() == 42)\n}"


def test_mceval_c_sharp_harness_makes_debug_assertions_observable() -> None:
    prefix, suffix = _mceval_harness(
        "c_sharp",
        "using System;\nclass Program {",
        "static void Main() { Debug.Assert(false); }\n}",
    )
    assert prefix.startswith("using System.Linq;\n")
    assert "TextWriterTraceListener(System.Console.Error)" in suffix
    assert "Debug.AutoFlush = true" in suffix

def test_source_freezer_script_exposes_pinned_offline_imports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "freeze_category_case_source.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ifeval" in completed.stdout
    assert "aegis-test" in completed.stdout
    assert "aegis-v1-parquet-test" in completed.stdout
    assert "mceval-execution" in completed.stdout
    for subcommand in ("ifeval", "aegis-test"):
        help_result = subprocess.run(
            [sys.executable, str(script), subcommand, "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert help_result.returncode == 0, help_result.stderr
        assert "--source" in help_result.stdout
        assert "--revision" in help_result.stdout
        assert "--source-sha256" in help_result.stdout
        assert "--output" in help_result.stdout
    mceval_help = subprocess.run(
        [sys.executable, str(script), "mceval-execution", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert mceval_help.returncode == 0, mceval_help.stderr
    assert "--source-root" in mceval_help.stdout
    assert "--source-manifest" in mceval_help.stdout
    assert "--sandbox-manifest" in mceval_help.stdout
    assert "--audit-output" in mceval_help.stdout
    aegis_v1_help = subprocess.run(
        [
            sys.executable,
            str(script),
            "aegis-v1-parquet-test",
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert aegis_v1_help.returncode == 0, aegis_v1_help.stderr
    assert "--source" in aegis_v1_help.stdout
    assert "--audit-output" in aegis_v1_help.stdout
