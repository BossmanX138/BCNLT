from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from bosscrafts_model.tokenizer import (
    EXPECTED_RUNTIME_BYTES,
    EXPECTED_RUNTIME_FILE_COUNT,
    EXPECTED_RUNTIME_PROVENANCE_SHA256,
    EXPECTED_RUNTIME_TREE_SHA256,
    _runtime_tree_identity,
    _sha256,
    _verify_owned_tokenizers_runtime,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "vendor" / "tokenizers_runtime_0_23"


def test_owned_tokenizers_runtime_tree_matches_frozen_identity() -> None:
    assert _sha256(RUNTIME_ROOT / "PROVENANCE.json") == (
        EXPECTED_RUNTIME_PROVENANCE_SHA256
    )
    assert _runtime_tree_identity(RUNTIME_ROOT) == (
        EXPECTED_RUNTIME_TREE_SHA256,
        EXPECTED_RUNTIME_FILE_COUNT,
        EXPECTED_RUNTIME_BYTES,
    )
    _verify_owned_tokenizers_runtime(RUNTIME_ROOT)


def test_runtime_verification_rejects_provenance_drift() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
        root = Path(temporary) / "runtime"
        shutil.copytree(RUNTIME_ROOT, root)
        provenance = root / "PROVENANCE.json"
        provenance.write_bytes(provenance.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="provenance identity mismatch"):
            _verify_owned_tokenizers_runtime(root)


def test_runtime_tree_identity_ignores_generated_bytecode() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
        root = Path(temporary) / "runtime"
        shutil.copytree(RUNTIME_ROOT, root)
        cache = root / "tokenizers" / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / "generated.pyc").write_bytes(b"not executable source")
        assert _runtime_tree_identity(root) == _runtime_tree_identity(RUNTIME_ROOT)


def test_runtime_tree_verification_rejects_python_wrapper_drift() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
        root = Path(temporary) / "runtime"
        shutil.copytree(RUNTIME_ROOT, root)
        wrapper = root / "tokenizers" / "__init__.py"
        wrapper.write_bytes(wrapper.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="runtime tree identity mismatch"):
            _verify_owned_tokenizers_runtime(root)


def test_owned_runtime_imports_without_public_tokenizers_package() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
assert "tokenizers" not in sys.modules
from bosscrafts_model.tokenizer import NativeTokenizerBinding
assert not any(
    name == "tokenizers" or name.startswith("tokenizers.")
    for name in sys.modules
)
binding = NativeTokenizerBinding.load()
assert binding.decode(binding.encode("clean import")) == "clean import"
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_owned_runtime_restores_preexisting_public_tokenizers_module() -> None:
    code = f"""
import sys
from types import ModuleType
sys.path.insert(0, {str(PROJECT_ROOT)!r})
existing = ModuleType("tokenizers")
sys.modules["tokenizers"] = existing
from bosscrafts_model.tokenizer import NativeTokenizerBinding
assert sys.modules["tokenizers"] is existing
binding = NativeTokenizerBinding.load()
assert binding.decode(binding.encode("preserved import")) == "preserved import"
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_owned_runtime_restores_preexisting_public_tokenizers_namespace() -> None:
    code = f"""
import sys
from types import ModuleType
sys.path.insert(0, {str(PROJECT_ROOT)!r})
root = ModuleType("tokenizers")
decoders = ModuleType("tokenizers.decoders")
sentinel = ModuleType("tokenizers.caller_only")
sys.modules["tokenizers"] = root
sys.modules["tokenizers.decoders"] = decoders
sys.modules["tokenizers.caller_only"] = sentinel
before = {{
    name: value
    for name, value in sys.modules.items()
    if name == "tokenizers" or name.startswith("tokenizers.")
}}
from bosscrafts_model.tokenizer import NativeTokenizerBinding
after = {{
    name: value
    for name, value in sys.modules.items()
    if name == "tokenizers" or name.startswith("tokenizers.")
}}
assert after == before
assert all(after[name] is value for name, value in before.items())
binding = NativeTokenizerBinding.load()
assert binding.decode(binding.encode("preserved namespace")) == "preserved namespace"
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_owned_runtime_restores_public_import_none_sentinel() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.modules["tokenizers"] = None
from bosscrafts_model.tokenizer import NativeTokenizerBinding
assert "tokenizers" in sys.modules
assert sys.modules["tokenizers"] is None
binding = NativeTokenizerBinding.load()
assert binding.decode(binding.encode("preserved sentinel")) == "preserved sentinel"
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
