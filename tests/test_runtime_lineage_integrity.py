from __future__ import annotations

import json
import os
import py_compile
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from bosscrafts_model import bcnlt_source_pipeline as source_pipeline
from bosscrafts_model import raw_ast_witness_sidecar
from bosscrafts_model.tokenizer import NativeTokenizerBinding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate"


def test_python_source_identity_is_checkout_line_ending_invariant(tmp_path: Path) -> None:
    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    lf_root.mkdir()
    crlf_root.mkdir()
    lf_source = b"def answer():\n    return 42\n"
    (lf_root / "sample.py").write_bytes(lf_source)
    (crlf_root / "sample.py").write_bytes(lf_source.replace(b"\n", b"\r\n"))

    assert source_pipeline._aggregate_sha256(lf_root, "*.py") == (
        source_pipeline._aggregate_sha256(crlf_root, "*.py")
    )


def test_contained_runtime_lineage_accepts_current_runtime_bytes() -> None:
    pipeline = source_pipeline.BCNLTSourcePipeline(NativeTokenizerBinding.load())

    assert pipeline.parser_runtime_identity_sha256 == (
        source_pipeline.PARSER_RUNTIME_IDENTITY_SHA256
    )
    assert raw_ast_witness_sidecar.PARSER_RUNTIME_IDENTITY_SHA256 == (
        pipeline.parser_runtime_identity_sha256
    )


def test_contained_runtime_site_packages_are_immutable_shipped_files() -> None:
    site_packages = RUNTIME_ROOT / "site-packages"
    shipped = sorted(path for path in site_packages.rglob("*") if path.is_file())
    identity_files = [
        path
        for path in shipped
        if not source_pipeline._is_generated_python_bytecode(path, site_packages)
    ]

    assert len(identity_files) == 12
    assert not source_pipeline._unexpected_runtime_cache_entries(site_packages)


def test_generated_runtime_bytecode_is_ignored_but_cannot_be_loaded(
    tmp_path: Path,
) -> None:
    site_packages = RUNTIME_ROOT / "site-packages"
    source = site_packages / "tree_sitter" / "__init__.py"
    cache_directory = source.parent / "__pycache__"
    cache = cache_directory / "__init__.cpython-313.pyc"
    marker = tmp_path / "local-bytecode-executed.txt"
    malicious_source = tmp_path / "__init__.py"
    prefix = f"from pathlib import Path\nPath({str(marker)!r}).write_text('loaded')\n"
    encoded = prefix.encode("utf-8")
    assert len(encoded) < source.stat().st_size
    malicious_source.write_bytes(
        encoded + b"#" + b" " * (source.stat().st_size - len(encoded) - 1)
    )
    os.utime(malicious_source, (source.stat().st_atime, source.stat().st_mtime))
    previous = cache.read_bytes() if cache.exists() else None
    cache_directory.mkdir(exist_ok=True)
    try:
        py_compile.compile(
            str(malicious_source),
            cfile=str(cache),
            dfile=str(source),
            doraise=True,
        )
        expected = source_pipeline._EXPECTED_RUNTIME_IDENTITY["site_packages"]
        actual, count, _ = source_pipeline._aggregate_sha256(
            site_packages,
            "*",
            ignore_generated_python_bytecode=True,
        )
        assert actual == expected
        assert count == 12

        with source_pipeline.BCNLTSourcePipeline(
            NativeTokenizerBinding.load()
        ) as pipeline:
            prepared = pipeline.prepare_semantic(
                "value = 1\n", language="python", path="bytecode-isolation.py"
            )
        assert prepared.record_count > 0
        assert not marker.exists()
    finally:
        if previous is None:
            cache.unlink(missing_ok=True)
        else:
            cache.write_bytes(previous)
        if cache_directory.exists() and not any(cache_directory.iterdir()):
            cache_directory.rmdir()


def test_runtime_cache_rejects_non_bytecode_files(tmp_path: Path) -> None:
    cache_directory = tmp_path / "package" / "__pycache__"
    cache_directory.mkdir(parents=True)
    (cache_directory / "module.cpython-313.pyc").write_bytes(b"generated")
    assert source_pipeline._unexpected_runtime_cache_entries(tmp_path) == ()

    unexpected = cache_directory / "payload.py"
    unexpected.write_text("raise RuntimeError\n", encoding="utf-8")
    assert source_pipeline._unexpected_runtime_cache_entries(tmp_path) == (unexpected,)


def test_sidecar_refuses_launch_without_isolated_bytecode_policy() -> None:
    sidecar = RUNTIME_ROOT / "bcnlt_source_sidecar.py"
    completed = subprocess.run(
        [str(source_pipeline._DEFAULT_PYTHON_313), "-I", "-X", "utf8", str(sidecar)],
        input="{}",
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    assert completed.returncode != 0
    assert "requires an isolated bytecode-cache policy" in completed.stderr


def test_contained_runtime_identity_covers_executable_site_packages() -> None:
    aggregate = source_pipeline._aggregate_sha256

    def changed_site_packages(
        root: Path,
        pattern: str,
        *,
        ignore_generated_python_bytecode: bool = False,
    ) -> tuple[str, int, int]:
        identity, count, size = aggregate(
            root,
            pattern,
            ignore_generated_python_bytecode=ignore_generated_python_bytecode,
        )
        if root.name == "site-packages":
            return "0" * 64, count, size
        return identity, count, size

    with patch(
        "bosscrafts_model.bcnlt_source_pipeline._aggregate_sha256",
        side_effect=changed_site_packages,
    ):
        with pytest.raises(ValueError, match="runtime identity mismatch"):
            source_pipeline.BCNLTSourcePipeline(NativeTokenizerBinding.load())


def test_contained_runtime_manifest_bytes_are_pinned() -> None:
    sha256 = source_pipeline._sha256

    def changed_manifest(path: Path) -> str:
        if path == source_pipeline._RUNTIME_MANIFEST:
            return "0" * 64
        return sha256(path)

    with patch(
        "bosscrafts_model.bcnlt_source_pipeline._sha256",
        side_effect=changed_manifest,
    ):
        with pytest.raises(ValueError, match="runtime manifest identity mismatch"):
            source_pipeline.BCNLTSourcePipeline(NativeTokenizerBinding.load())


def test_calibration_runtime_accepts_current_contained_source() -> None:
    code = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})",
            "from bosscrafts_model import semantic_calibration_promotion as calibration",
            "print(calibration._EXPECTED_SOURCE_SHA256)",
        )
    )
    completed = subprocess.run(
        [str(source_pipeline._DEFAULT_PYTHON_313), "-I", "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == source_pipeline._EXPECTED_RUNTIME_IDENTITY["source"]


def test_semantic_spec_pins_match_current_contained_runtime() -> None:
    source_root = RUNTIME_ROOT / "source"
    code = "\n".join(
        (
            "import json, sys",
            f"sys.path.insert(0, {str(source_root)!r})",
            "from bcnlt.adapters.carousel import LanguageCarousel",
            "carousel = LanguageCarousel.default()",
            "print(json.dumps({item.definition.canonical_id: item.spec.canonical_hash for item in carousel.cartridges}, sort_keys=True))",
        )
    )
    completed = subprocess.run(
        [str(source_pipeline._DEFAULT_PYTHON_313), "-I", "-c", code],
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == source_pipeline.CURRENT_RUNTIME_SPEC_SHA256

