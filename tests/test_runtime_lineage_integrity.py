from __future__ import annotations

import json
import subprocess
from pathlib import Path

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

