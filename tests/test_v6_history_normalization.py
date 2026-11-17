from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import pytest
from runtime_import import select_bcnlt_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate"
SIDECAR = RUNTIME / "bcnlt_source_sidecar.py"
PYTHON_313 = Path(sys.executable).resolve()
PROTOCOL = "bosscrafts-bcnlt-source-sidecar-v4"
select_bcnlt_runtime(RUNTIME / "source")

from bcnlt.adapters.python import PythonAdapter  # noqa: E402
from bcnlt.semantic_record import FrozenMap, freeze_json  # noqa: E402


@pytest.mark.parametrize(
    ("language", "source"),
    (
        (
            "python",
            "def f():\n    x = '{payload}'\n    return x\ny = f()\n",
        ),
        (
            "javascript",
            "function f() {{ const x = '{payload}'; return x; }} f();",
        ),
        (
            "php",
            "<?php function f() {{ $x = '{payload}'; return $x; }} f(); ?>",
        ),
    ),
)
def test_history_is_normalized_before_truncation(language: str, source: str) -> None:
    # U+0344 expands to two code points under NFC. Truncate-before-normalize
    # therefore violates the 512-character history bound.
    payload = "\u0344" * 520
    request = {
        "protocol": PROTOCOL,
        "source": source.format(payload=payload),
        "language": language,
        "path": f"history-normalization.{language}",
        "dialect": None,
        "learning_mode": "ephemeral",
        "learned_memory_root": None,
    }
    with tempfile.TemporaryDirectory(
        prefix="bosscrafts-history-normalization-pycache-"
    ) as bytecode_cache:
        completed = subprocess.run(
            [
                str(PYTHON_313),
                "-I",
                "-B",
                "-X",
                f"pycache_prefix={bytecode_cache}",
                "-X",
                "utf8",
                str(SIDECAR),
            ],
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=30,
            check=False,
            cwd=RUNTIME,
        )
    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    records = [json.loads(line) for line in response["canonical_jsonl"].splitlines()]
    histories = [item for record in records for item in record["recent_history"]]

    assert any(len(record["surface"]) > 512 for record in records)
    assert any(len(item) == 512 for item in histories)
    assert all(len(item) <= 512 for item in histories)
    assert all(unicodedata.normalize("NFC", item) == item for item in histories)


def test_direct_python_adapter_applies_the_same_history_bound() -> None:
    payload = "\u0344" * 520
    source = f"x = '{payload}'\ny = x\n"
    adapter = PythonAdapter()
    document = adapter.parse(source, path="direct-history.py")
    units = tuple(adapter.to_units(document))
    histories = [item for unit in units for item in unit.recent_history]

    assert any(len(unit.surface) > 512 for unit in units)
    assert any(len(item) == 512 for item in histories)
    assert all(len(item) <= 512 for item in histories)
    assert all(unicodedata.normalize("NFC", item) == item for item in histories)


def test_json_keys_that_collide_after_normalization_are_rejected() -> None:
    with pytest.raises(ValueError, match="collide after normalization"):
        freeze_json({"\u212b": 1, "\u00c5": 2})


def test_frozen_map_rejects_duplicate_or_noncanonical_keys() -> None:
    with pytest.raises(ValueError, match="unique"):
        FrozenMap((("a", 1), ("a", 2)))
    with pytest.raises(ValueError, match="normalized"):
        FrozenMap((("\u212b", 1),))
