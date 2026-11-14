from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from bcnlt.languages import CANONICAL_LANGUAGE_IDS


CONFIG_PATH = Path("configs/bcnlt_native_foundation_65536.json")


def _config_type():
    try:
        module = importlib.import_module("bcnlt.native_foundation.config")
    except ModuleNotFoundError as exc:
        pytest.fail(f"native foundation configuration is not implemented: {exc}")
    return module.NativeFoundationConfig


def test_production_native_foundation_configuration_selects_the_65536_candidate():
    config = _config_type().load(CONFIG_PATH)

    assert config.schema_version == "bcnlt-native-foundation-tokenizer-v1"
    assert config.primary_vocabulary_size == 65_536
    assert config.control_vocabulary_size == 32_768
    assert config.tied_embeddings is True
    assert config.required_text_domains == ("english",)
    assert config.canonical_code_languages == CANONICAL_LANGUAGE_IDS
    assert config.byte_fallback is True
    assert config.preserve_legacy_artifacts is True
    assert config.minimum_documents_per_domain == 100
    assert config.minimum_sources_per_domain == 3
    assert config.minimum_payload_units_per_domain == 100_000
    assert config.default_output_root == "D:\\BCNLT-runs\\bcnlt-native-foundation-65536"
    assert CONFIG_PATH.read_bytes() == config.canonical_bytes()
