from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from bcnlt.languages import CANONICAL_LANGUAGE_IDS


CONFIG_PATH = Path("configs/bcnlt_native_foundation_65536.json")
SCRIPT_PATH = Path("scripts/preflight_bcnlt_native_foundation.py")


def test_preflight_reuses_hash_bound_multilingual_and_wizard_sources(tmp_path: Path):
    multilingual_report = {
        "schema_version": "bcnlt-multilingual-packed-audit-v1",
        "passed": True,
        "language_reports": [
            {
                "language": language,
                "passed": True,
                "examples": 100,
                "training_records": 100,
            }
            for language in CANONICAL_LANGUAGE_IDS
        ],
        "training_records": 1_400,
        "training_tokens": 1_000_000,
    }
    multilingual_path = tmp_path / "per-language-validation.json"
    multilingual_bytes = (
        json.dumps(multilingual_report, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    multilingual_path.write_bytes(multilingual_bytes)
    dataset_manifest = {
        "schema_version": "bcnlt-packed-dataset-v2",
        "accepted_record_count": 1_400,
        "rejected_record_count": 0,
        "tokenizer_vocab_size": 32_073,
        "dataset_manifest_sha256": "d" * 64,
    }
    manifest_path = tmp_path / "dataset-manifest.json"
    manifest_bytes = (
        json.dumps(dataset_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    accepted_corpus_path = tmp_path / "accepted.jsonl"
    accepted_corpus_bytes = (
        b'{"split":"train","prompt":"hello","target":"world"}\n'
        b'{"split":"validation","prompt":"held out","target":"example"}\n'
    )
    accepted_corpus_path.write_bytes(accepted_corpus_bytes)
    english_path = tmp_path / "wizard-english.json"
    english_records = [
        {
            "instruction": f"Explain coding example {index}",
            "output": f"Example answer {index}",
            "review_status": "approved",
        }
        for index in range(100)
    ]
    english_bytes = (
        json.dumps(english_records, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    english_path.write_bytes(english_bytes)
    output_path = tmp_path / "preflight.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(CONFIG_PATH),
            "--multilingual-report",
            str(multilingual_path),
            "--dataset-manifest",
            str(manifest_path),
            "--accepted-corpus",
            str(accepted_corpus_path),
            "--english-dataset",
            str(english_path),
            "--output",
            str(output_path),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["primary_vocabulary_size"] == 65_536
    assert payload["control_vocabulary_size"] == 32_768
    assert payload["multilingual_languages"] == list(CANONICAL_LANGUAGE_IDS)
    assert payload["multilingual_training_records"] == 1_400
    assert payload["multilingual_training_tokens"] == 1_000_000
    assert payload["english_approved_records"] == 100
    assert payload["failure_codes"] == []
    assert payload["config_sha256"] == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert payload["multilingual_report_sha256"] == hashlib.sha256(multilingual_bytes).hexdigest()
    assert payload["dataset_manifest_file_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert payload["accepted_corpus_sha256"] == hashlib.sha256(accepted_corpus_bytes).hexdigest()
    assert payload["english_dataset_sha256"] == hashlib.sha256(english_bytes).hexdigest()
