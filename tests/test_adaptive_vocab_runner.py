from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bcnlt.adaptive_vocab.runner import (
    ExperimentInputs,
    ExperimentPhase,
    plan_experiment,
    run_experiment,
    run_phase,
)


ALL_COMMANDS = tuple(
    (
        name,
        (
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(r'{output}').mkdir()",
            "{config}",
        ),
    )
    for name in (
        "lineage", "preliminary_corpus_rebuild", "audit", "candidates",
        "inventories", "tokenizer_eval", "entropy", "paired_data",
        "paired_training", "model_eval",
    )
)


def _inputs(tmp_path: Path, **overrides) -> ExperimentInputs:
    config = tmp_path / "config with spaces.json"
    config.write_text('{"secret":"source text must not enter logs"}\n', encoding="utf-8")
    values = dict(
        output_root=tmp_path / "run output",
        protected_paths=(tmp_path / "preserved",),
        config_path=config,
        code_revision="a" * 40,
        phase_commands=ALL_COMMANDS,
        legacy_corpus=False,
        start_training=False,
    )
    values.update(overrides)
    return ExperimentInputs(**values)


def test_plan_orders_optional_rebuild_and_requires_explicit_training_flag(tmp_path: Path):
    tokenizer_only = plan_experiment(_inputs(tmp_path, legacy_corpus=True))
    assert tuple(item.name for item in tokenizer_only) == (
        "lineage", "preliminary_corpus_rebuild", "audit", "candidates",
        "inventories", "tokenizer_eval", "entropy",
    )
    assert all(not item.requires_training_gate for item in tokenizer_only)

    full = plan_experiment(_inputs(tmp_path, start_training=True))
    assert tuple(item.name for item in full) == (
        "lineage", "audit", "candidates", "inventories", "tokenizer_eval",
        "entropy", "paired_data", "paired_training", "model_eval",
    )
    assert all(item.requires_training_gate for item in full[-3:])


def test_plan_rejects_any_output_overlapping_a_preserved_artifact(tmp_path: Path):
    preserved = tmp_path / "preserved"
    with pytest.raises(ValueError, match="overlaps preserved"):
        plan_experiment(
            _inputs(tmp_path, output_root=preserved / "candidate", protected_paths=(preserved,))
        )


def test_phase_is_atomic_resumable_and_refuses_changed_inputs(tmp_path: Path):
    source = tmp_path / "input.txt"
    source.write_text("v1", encoding="utf-8")
    output = tmp_path / "phase output"
    code = (
        "from pathlib import Path; import sys; "
        "p=Path(sys.argv[1]); p.mkdir(); (p/'artifact.txt').write_text('ok')"
    )
    phase = ExperimentPhase(
        "audit", (sys.executable, "-c", code, "{output}"), (source,), output,
        "b" * 40,
    )

    first = run_phase(phase, resume=False)
    assert first.status == "completed"
    assert output.is_dir()
    assert first.input_sha256 and first.output_sha256
    resumed = run_phase(phase, resume=True)
    assert resumed == first

    source.write_text("v2", encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        run_phase(phase, resume=True)


def test_failed_or_interrupted_phase_never_renames_partial_output(tmp_path: Path):
    output = tmp_path / "result"
    code = (
        "from pathlib import Path; import sys; "
        "p=Path(sys.argv[1]); p.mkdir(); (p/'partial').write_text('x'); raise SystemExit(1)"
    )
    phase = ExperimentPhase("audit", (sys.executable, "-c", code, "{output}"), (), output, "c" * 40)
    result = run_phase(phase, resume=False)
    assert result.status == "failed"
    assert result.return_code == 1
    assert not output.exists()


def test_research_hard_stop_is_committed_as_a_valid_atomic_result(tmp_path: Path):
    output = tmp_path / "audit"
    code = (
        "from pathlib import Path; import sys; "
        "p=Path(sys.argv[1]); p.mkdir(); "
        "(p/'corpus-audit.json').write_text('{\"passed\":false}'); raise SystemExit(2)"
    )
    phase = ExperimentPhase("audit", (sys.executable, "-c", code, "{output}"), (), output, "d" * 40)
    result = run_phase(phase, resume=False)
    assert result.status == "hard_stop"
    assert result.return_code == 2
    assert (output / "corpus-audit.json").is_file()


def test_command_arguments_preserve_spaces_without_logging_file_contents(tmp_path: Path):
    planned = plan_experiment(_inputs(tmp_path))
    lineage = planned[0]
    assert str(_inputs(tmp_path).config_path) in lineage.command
    assert "--start-training" not in lineage.command
    serialized = json.dumps([item.to_dict() for item in planned])
    assert "source text must not enter logs" not in serialized


def test_training_phase_requires_flag_and_passing_tokenizer_selection(tmp_path: Path):
    output = tmp_path / "training"
    phase = ExperimentPhase(
        "paired_training",
        (sys.executable, "-c", "from pathlib import Path; Path(r'{output}').mkdir()"),
        (), output, "e" * 40, requires_training_gate=True,
    )
    selection = tmp_path / "selected.json"
    selection.write_text('{"training_eligible":false}\n', encoding="utf-8")
    stopped = run_experiment(
        (phase,), resume=False, start_training=True,
        tokenizer_selection_path=selection,
    )
    assert stopped.return_code == 2
    assert not output.exists()

    selection.write_text('{"training_eligible":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="explicit"):
        run_experiment(
            (phase,), resume=False, start_training=False,
            tokenizer_selection_path=selection,
        )
    completed = run_experiment(
        (phase,), resume=False, start_training=True,
        tokenizer_selection_path=selection,
    )
    assert completed.return_code == 0


def test_powershell_wrapper_forwards_an_argument_list_without_shell_fragments():
    source = Path("scripts/run_bcnlt_adaptive_vocab_v5.ps1").read_text(encoding="utf-8")
    assert "& wsl.exe @Arguments" in source
    assert "Invoke-Expression" not in source
    assert "--start-training" in source
    assert "if ($StartTraining)" in source

