from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from bcnlt.languages import CANONICAL_LANGUAGE_IDS
from bcnlt.native_foundation.config import NativeFoundationConfig


CONFIG_PATH = Path("configs/bcnlt_native_foundation_65536.json")


def _evaluate(config, audit):
    try:
        module = importlib.import_module("bcnlt.native_foundation.readiness")
    except ModuleNotFoundError as exc:
        pytest.fail(f"native corpus readiness gate is not implemented: {exc}")
    return module.evaluate_corpus_readiness(config, audit)


def test_python_only_legacy_audit_cannot_authorize_native_tokenizer_induction():
    config = NativeFoundationConfig.load(CONFIG_PATH)
    audit = {
        "accepted_records_sha256": "a" * 64,
        "per_language": {
            "python": {
                "document_count": 557,
                "repository_count": 381,
                "payload_base_tokens": 7_295_136,
            }
        },
    }

    report = _evaluate(config, audit)

    assert report.passed is False
    assert report.ready_domains == ("python",)
    assert report.missing_domains == (
        "english",
        *tuple(language for language in CANONICAL_LANGUAGE_IDS if language != "python"),
    )
    assert report.below_threshold == {}
    assert report.failure_codes == ("missing_required_domains",)
