from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import NativeFoundationConfig


def run_source_preflight(
    *,
    config_path: str | Path,
    multilingual_report_path: str | Path,
    dataset_manifest_path: str | Path,
    accepted_corpus_path: str | Path,
    english_dataset_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path)
    multilingual_file = Path(multilingual_report_path)
    manifest_file = Path(dataset_manifest_path)
    accepted_corpus_file = Path(accepted_corpus_path)
    english_file = Path(english_dataset_path)
    config = NativeFoundationConfig.load(config_file)
    multilingual = _load_object(multilingual_file, "multilingual report")
    manifest = _load_object(manifest_file, "dataset manifest")
    english = _load_list(english_file, "English dataset")

    failures: list[str] = []
    expected_languages = config.canonical_code_languages
    language_reports = multilingual.get("language_reports")
    reports_by_language: dict[str, dict[str, Any]] = {}
    if isinstance(language_reports, list):
        for report in language_reports:
            if isinstance(report, dict) and isinstance(report.get("language"), str):
                reports_by_language[report["language"]] = report
    if multilingual.get("passed") is not True:
        failures.append("multilingual_report_not_passed")
    if tuple(reports_by_language) != expected_languages:
        failures.append("multilingual_language_set_mismatch")
    else:
        for language in expected_languages:
            report = reports_by_language[language]
            if report.get("passed") is not True:
                failures.append(f"language_not_passed:{language}")
            if _nonnegative_int(report.get("examples")) < config.minimum_documents_per_domain:
                failures.append(f"insufficient_examples:{language}")
            if _nonnegative_int(report.get("training_records")) < config.minimum_documents_per_domain:
                failures.append(f"insufficient_training_records:{language}")

    if manifest.get("schema_version") != "bcnlt-packed-dataset-v2":
        failures.append("unsupported_dataset_manifest")
    if manifest.get("tokenizer_vocab_size") != 32_073:
        failures.append("unexpected_source_tokenizer_vocabulary")
    if _nonnegative_int(manifest.get("accepted_record_count")) == 0:
        failures.append("empty_packed_dataset")
    if not accepted_corpus_file.is_file() or accepted_corpus_file.stat().st_size == 0:
        failures.append("empty_accepted_corpus")

    approved_english = sum(
        1
        for record in english
        if isinstance(record, dict)
        and record.get("review_status") == "approved"
        and isinstance(record.get("instruction"), str)
        and bool(record["instruction"].strip())
        and isinstance(record.get("output"), str)
        and bool(record["output"].strip())
    )
    if approved_english < config.minimum_documents_per_domain:
        failures.append("insufficient_approved_english_records")

    payload: dict[str, Any] = {
        "schema_version": "bcnlt-native-foundation-preflight-v2",
        "passed": not failures,
        "primary_vocabulary_size": config.primary_vocabulary_size,
        "control_vocabulary_size": config.control_vocabulary_size,
        "multilingual_languages": list(expected_languages),
        "multilingual_training_records": _nonnegative_int(multilingual.get("training_records")),
        "multilingual_training_tokens": _nonnegative_int(multilingual.get("training_tokens")),
        "english_approved_records": approved_english,
        "source_tokenizer_vocab_size": manifest.get("tokenizer_vocab_size"),
        "packed_accepted_record_count": _nonnegative_int(manifest.get("accepted_record_count")),
        "failure_codes": failures,
        "config_sha256": _sha256(config_file),
        "multilingual_report_sha256": _sha256(multilingual_file),
        "dataset_manifest_file_sha256": _sha256(manifest_file),
        "accepted_corpus_sha256": _sha256(accepted_corpus_file),
        "english_dataset_sha256": _sha256(english_file),
    }
    _atomic_write_json(Path(output_path), payload)
    return payload


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_list(path: Path, label: str) -> list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    return value


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
