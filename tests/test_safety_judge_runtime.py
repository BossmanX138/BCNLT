from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bosscrafts_model.safety_judge_runtime import (
    SafetyJudgeFilePin,
    _seal_pinned_safety_judge_runtime,
    _verify_pinned_safety_judge_runtime,
)


def _pin(content: bytes) -> SafetyJudgeFilePin:
    return SafetyJudgeFilePin(
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_seals_exact_inventory_and_rejects_tamper_or_extra_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "judge"
    root.mkdir()
    files = {
        "config.json": b'{"model_type":"fixture"}\n',
        "weights/model.safetensors": b"fixture-weights",
    }
    for relative, content in files.items():
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    pins = {path: _pin(content) for path, content in files.items()}

    sealed = _seal_pinned_safety_judge_runtime(
        root,
        repository="fixture/judge",
        revision="a" * 40,
        license_name="fixture-license",
        license_url="https://example.invalid/license",
        runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
        file_pins=pins,
    )
    verified = _verify_pinned_safety_judge_runtime(
        root,
        repository="fixture/judge",
        revision="a" * 40,
        license_name="fixture-license",
        license_url="https://example.invalid/license",
        runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
        file_pins=pins,
    )

    assert verified == sealed
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_sha256"] == sealed.manifest_sha256
    with pytest.raises(ValueError, match="already exists"):
        _seal_pinned_safety_judge_runtime(
            root,
            repository="fixture/judge",
            revision="a" * 40,
            license_name="fixture-license",
            license_url="https://example.invalid/license",
            runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
            file_pins=pins,
        )

    (root / "config.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        _verify_pinned_safety_judge_runtime(
            root,
            repository="fixture/judge",
            revision="a" * 40,
            license_name="fixture-license",
            license_url="https://example.invalid/license",
            runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
            file_pins=pins,
        )

    (root / "config.json").write_bytes(files["config.json"])
    (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        _verify_pinned_safety_judge_runtime(
            root,
            repository="fixture/judge",
            revision="a" * 40,
            license_name="fixture-license",
            license_url="https://example.invalid/license",
            runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
            file_pins=pins,
        )


def test_rejects_nested_manifest_file_that_is_not_pinned(tmp_path: Path) -> None:
    root = tmp_path / "judge"
    root.mkdir()
    content = b'{}\n'
    (root / "config.json").write_bytes(content)
    pins = {"config.json": _pin(content)}
    _seal_pinned_safety_judge_runtime(
        root,
        repository="fixture/judge",
        revision="a" * 40,
        license_name="fixture-license",
        license_url="https://example.invalid/license",
        runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
        file_pins=pins,
    )
    nested_manifest = root / "untrusted" / "manifest.json"
    nested_manifest.parent.mkdir()
    nested_manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory"):
        _verify_pinned_safety_judge_runtime(
            root,
            repository="fixture/judge",
            revision="a" * 40,
            license_name="fixture-license",
            license_url="https://example.invalid/license",
            runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
            file_pins=pins,
        )


def test_rejects_unexpected_empty_directory(tmp_path: Path) -> None:
    root = tmp_path / "judge"
    root.mkdir()
    content = b'{}\n'
    (root / "config.json").write_bytes(content)
    pins = {"config.json": _pin(content)}
    _seal_pinned_safety_judge_runtime(
        root,
        repository="fixture/judge",
        revision="a" * 40,
        license_name="fixture-license",
        license_url="https://example.invalid/license",
        runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
        file_pins=pins,
    )
    (root / "unexpected-empty-directory").mkdir()

    with pytest.raises(ValueError, match="inventory"):
        _verify_pinned_safety_judge_runtime(
            root,
            repository="fixture/judge",
            revision="a" * 40,
            license_name="fixture-license",
            license_url="https://example.invalid/license",
            runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
            file_pins=pins,
        )


def test_rejects_symbolic_link_runtime_root(tmp_path: Path) -> None:
    root = tmp_path / "judge"
    root.mkdir()
    content = b'{}\n'
    (root / "config.json").write_bytes(content)
    pins = {"config.json": _pin(content)}
    _seal_pinned_safety_judge_runtime(
        root,
        repository="fixture/judge",
        revision="a" * 40,
        license_name="fixture-license",
        license_url="https://example.invalid/license",
        runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
        file_pins=pins,
    )
    link = tmp_path / "judge-link"
    try:
        link.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("creating directory symlinks is unavailable")

    with pytest.raises(ValueError, match="root is invalid"):
        _verify_pinned_safety_judge_runtime(
            link,
            repository="fixture/judge",
            revision="a" * 40,
            license_name="fixture-license",
            license_url="https://example.invalid/license",
            runtime_versions={"torch": "2.8.0", "transformers": "4.57.1"},
            file_pins=pins,
        )


def test_safety_judge_runtime_script_exposes_seal_and_verify() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "seal_safety_judge_runtime.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "seal" in completed.stdout
    assert "verify" in completed.stdout
    for command in ("seal", "verify"):
        command_help = subprocess.run(
            [sys.executable, str(script), command, "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert command_help.returncode == 0, command_help.stderr
        assert "--runtime-root" in command_help.stdout
