from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from bcnlt.adapters.python import PythonAdapter, build_python_spec


FIXTURE_ROOT = Path(__file__).parent / "language_fixtures"


def load_golden(language: str) -> tuple[str, dict[str, object]]:
    directory = FIXTURE_ROOT / language
    sources = [path for path in directory.iterdir() if path.name != "core.expected.json"]
    assert len(sources) == 1
    expected = json.loads((directory / "core.expected.json").read_text(encoding="utf-8"))
    return sources[0].read_text(encoding="utf-8"), expected


def canonical_unit_bytes(source: str, path: str = "core.py") -> bytes:
    adapter = PythonAdapter()
    document = adapter.parse(source, path=path)
    payload = [unit.to_dict() for unit in adapter.to_units(document)]
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def test_python_golden_fixture_preserves_semantics_and_links():
    source, expected = load_golden("python")
    adapter = PythonAdapter()
    document = adapter.parse(source, path="core.py")
    units = list(adapter.to_units(document))

    events = {str(unit.event) for unit in units}
    assert document.language == expected["language"]
    assert set(expected["required_events"]) <= events
    assert set(expected["required_definitions"]) <= {
        unit.surface for unit in units if unit.event in {"DEFINE", "CLASS"}
    }
    for surface in expected["required_resolved_references"]:
        assert any(
            unit.surface == surface and unit.event == "REFERENCE" and unit.defines
            for unit in units
        )
    assert set(expected["unknown_nodes"]) == {
        unit.metadata["unknown_node"]
        for unit in units
        if unit.event == "UNKNOWN_SEMANTIC"
    }


def test_python_spec_is_stable_and_python_adapter_uses_a_cartridge_facade():
    first = build_python_spec()
    second = build_python_spec()
    adapter = PythonAdapter()

    assert first.canonical_hash == second.canonical_hash
    assert adapter.cartridge.definition.canonical_id == "python"
    assert adapter.cartridge.spec.canonical_hash == first.canonical_hash


@pytest.mark.skipif(os.name != "nt" or shutil.which("wsl.exe") is None, reason="Windows/WSL gate")
def test_python_golden_bytes_match_windows_and_wsl_after_lf_normalization():
    source, _ = load_golden("python")
    windows = canonical_unit_bytes(source).replace(b"\r\n", b"\n")
    repository = Path(__file__).parents[1]
    drive = repository.drive.rstrip(":").lower()
    converted = f"/mnt/{drive}/{repository.as_posix().split(':/', 1)[1]}"
    command = (
        f"cd {shlex.quote(converted)} && "
        "~/.venvs/wizard-phi-eval/bin/python -c "
        "'from tests.test_multilingual_golden import load_golden,canonical_unit_bytes; "
        "s,_=load_golden(\"python\"); import sys; sys.stdout.buffer.write(canonical_unit_bytes(s))'"
    )
    completed = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.stdout.replace(b"\r\n", b"\n") == windows
