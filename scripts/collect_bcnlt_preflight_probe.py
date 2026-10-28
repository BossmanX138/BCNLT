from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.replace("\x00", "").strip()


def _wsl(distribution: str, *command: str) -> str:
    return _run(["wsl.exe", "-d", distribution, "--", *command])


def collect_probe(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base = config["base"]
    preflight = config["preflight"]
    distribution = str(base["wsl_distribution"])
    model_path = str(base["wsl_path"])
    python = str(preflight["python_executable"])

    distributions = {
        line.strip()
        for line in _run(["wsl.exe", "--list", "--quiet"]).splitlines()
        if line.strip()
    }
    if distribution not in distributions:
        return {"wsl_available": False, "wsl_distribution": distribution}

    model_exists = subprocess.run(
        ["wsl.exe", "-d", distribution, "--", "test", "-d", model_path],
        check=False,
    ).returncode == 0
    base_hashes: dict[str, str] = {}
    if model_exists:
        for filename in base["required_file_sha256"]:
            output = _wsl(distribution, "sha256sum", f"{model_path}/{filename}")
            base_hashes[filename] = output.split()[0].lower()

    gpu_parts = [
        part.strip()
        for part in _wsl(
            distribution,
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ).split(",")
    ]
    disk_lines = _wsl(distribution, "df", "-B1", "--output=avail", model_path).splitlines()
    disk_free = int(next(line.strip() for line in reversed(disk_lines) if line.strip().isdigit()))

    packages: dict[str, str] = {}
    for package in preflight["required_packages"]:
        packages[str(package)] = _wsl(
            distribution,
            python,
            "-c",
            f"import importlib.metadata as m; print(m.version({str(package)!r}))",
        ).splitlines()[-1]

    import truststore

    truststore.inject_into_ssl()
    from huggingface_hub import HfApi

    api = HfApi()
    identity = api.whoami()["name"]
    dataset_access: dict[str, bool] = {}
    for source in config["hf_sources"]:
        dataset = str(source["dataset"])
        revision = str(source["revision"])
        probe_path = str(source["probe_path"])
        info = api.dataset_info(repo_id=dataset, revision=revision)
        items = api.list_repo_tree(
            repo_id=dataset,
            repo_type="dataset",
            revision=info.sha,
            path_in_repo=probe_path,
            recursive=False,
        )
        dataset_access[dataset] = next(iter(items), None) is not None

    return {
        "wsl_available": True,
        "wsl_distribution": distribution,
        "model_path_exists": model_exists,
        "resolved_model_path": model_path,
        "gpu": {
            "name": gpu_parts[0],
            "memory_total_mib": int(gpu_parts[1]),
            "driver_version": gpu_parts[2],
            "cuda_available": True,
        },
        "disk_free_bytes": disk_free,
        "python_packages": packages,
        "base_file_sha256": base_hashes,
        "hf_dataset_access": dataset_access,
        "hf_probe_host": "windows",
        "hf_authenticated_identity": identity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect a mixed Windows-Hugging Face and WSL CUDA preflight probe"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    payload = collect_probe(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
