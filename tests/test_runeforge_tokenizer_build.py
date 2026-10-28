from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.build_runeforge_bcnlt_tokenizer import build_tokenizer_artifact
from tests.tokenizer_fixtures import write_byte_level_tokenizer


ROOT = Path(__file__).parents[1]
PREFLIGHT = ROOT / "scripts" / "preflight_bcnlt_runeforge.ps1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object], Path]:
    base = write_byte_level_tokenizer(tmp_path / "base")
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    required_files = ("config.json", "tokenizer.json", "tokenizer_config.json")
    file_hashes = {name: _sha256(base / name) for name in required_files}
    config = {
        "schema_version": "runeforge-mk2-bcnlt-v1",
        "base": {
            "name": "fixture/runeforge",
            "revision": "fixture-merged-v1",
            "wsl_distribution": "Ubuntu",
            "wsl_path": "/tmp/fixture-runeforge",
            "windows_path": str(base),
            "required_file_sha256": file_hashes,
        },
        "preflight": {
            "python_executable": "/tmp/python",
            "min_gpu_memory_mib": 7000,
            "min_disk_free_bytes": 10_000,
            "required_packages": ["torch", "transformers", "peft"],
        },
        "hf_sources": [
            {"dataset": "bigcode/the-stack-dedup", "revision": "stack-pin"},
            {"dataset": "nvidia/OpenCodeInstruct", "revision": "nvidia-pin"},
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    probe = {
        "wsl_available": True,
        "wsl_distribution": "Ubuntu",
        "model_path_exists": True,
        "resolved_model_path": str(base),
        "gpu": {
            "name": "Fixture GPU",
            "memory_total_mib": 8192,
            "driver_version": "fixture",
            "cuda_available": True,
        },
        "disk_free_bytes": 1_000_000,
        "python_packages": {"torch": "1", "transformers": "1", "peft": "1"},
        "base_file_sha256": file_hashes,
        "hf_dataset_access": {
            "bigcode/the-stack-dedup": True,
            "nvidia/OpenCodeInstruct": True,
        },
    }
    evidence = tmp_path / "preflight.json"
    return base, config_path, probe, evidence


def _run_preflight(
    config_path: Path,
    probe: dict[str, object],
    evidence: Path,
) -> subprocess.CompletedProcess[str]:
    probe_path = evidence.with_name("probe.json")
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    return subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(PREFLIGHT),
            "-ConfigPath",
            str(config_path),
            "-EvidencePath",
            str(evidence),
            "-ProbeJsonPath",
            str(probe_path),
        ],
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda probe, config: probe.__setitem__("wsl_available", False), "WSL"),
        (
            lambda probe, config: probe["gpu"].__setitem__(  # type: ignore[union-attr]
                "memory_total_mib", config["preflight"]["min_gpu_memory_mib"] - 1  # type: ignore[index]
            ),
            "GPU memory",
        ),
        (
            lambda probe, config: probe.__setitem__(
                "disk_free_bytes", config["preflight"]["min_disk_free_bytes"] - 1  # type: ignore[index]
            ),
            "disk",
        ),
        (
            lambda probe, config: probe["hf_dataset_access"].__setitem__(  # type: ignore[union-attr]
                "bigcode/the-stack-dedup", False
            ),
            "Hugging Face dataset access",
        ),
    ],
)
def test_preflight_rejects_missing_runtime_requirements(tmp_path, mutation, message):
    _, config_path, probe, evidence = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mutation(probe, config)

    result = _run_preflight(config_path, probe, evidence)

    assert result.returncode != 0
    assert message.lower() in (result.stdout + result.stderr).lower()
    assert not evidence.exists()


def test_preflight_writes_complete_pass_evidence(tmp_path: Path):
    _, config_path, probe, evidence = _fixture(tmp_path)

    result = _run_preflight(config_path, probe, evidence)
    payload = json.loads(evidence.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stderr
    assert payload["status"] == "passed"
    assert payload["wsl_distribution"] == "Ubuntu"
    assert payload["gpu"]["memory_total_mib"] == 8192
    assert payload["disk_free_bytes"] == 1_000_000
    assert payload["python_packages"]["torch"] == "1"
    assert payload["base_file_sha256"] == probe["base_file_sha256"]
    assert all(payload["hf_dataset_access"].values())
    assert payload["config_sha256"] == _sha256(config_path)


@pytest.mark.parametrize("changed_file", ["config.json", "tokenizer.json"])
def test_builder_rejects_changed_base_checkpoint_files(tmp_path: Path, changed_file: str):
    base, config_path, probe, evidence = _fixture(tmp_path)
    assert _run_preflight(config_path, probe, evidence).returncode == 0
    (base / changed_file).write_bytes((base / changed_file).read_bytes() + b"changed")

    with pytest.raises(ValueError, match=changed_file):
        build_tokenizer_artifact(config_path, evidence, tmp_path / "artifact")


def test_builder_rejects_existing_output(tmp_path: Path):
    _, config_path, probe, evidence = _fixture(tmp_path)
    assert _run_preflight(config_path, probe, evidence).returncode == 0
    output = tmp_path / "artifact"
    output.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        build_tokenizer_artifact(config_path, evidence, output)


def test_builder_is_reproducible_and_records_lineage(tmp_path: Path):
    _, config_path, probe, evidence = _fixture(tmp_path)
    assert _run_preflight(config_path, probe, evidence).returncode == 0

    first = build_tokenizer_artifact(config_path, evidence, tmp_path / "first")
    second = build_tokenizer_artifact(config_path, evidence, tmp_path / "second")
    first_manifest = (tmp_path / "first" / "bcnlt_tokenizer_manifest.json").read_bytes()
    second_manifest = (tmp_path / "second" / "bcnlt_tokenizer_manifest.json").read_bytes()
    lineage = json.loads(
        (tmp_path / "first" / "tokenizer_lineage.json").read_text(encoding="utf-8")
    )

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first_manifest == second_manifest
    assert lineage["manifest_sha256"] == first["manifest_sha256"]
    assert lineage["preflight_sha256"] == _sha256(evidence)
    assert lineage["base_file_sha256"] == probe["base_file_sha256"]
