from __future__ import annotations

import pytest

from bcnlt.adaptive_vocab.config import AdaptiveVocabularyConfig
from bcnlt.adaptive_vocab.secrets import classify_sensitive_text


CONFIG = AdaptiveVocabularyConfig.load("configs/bcnlt_adaptive_vocab_v5.json")


@pytest.mark.parametrize(
    ("text", "pattern_name"),
    [
        ("-----BEGIN PRIVATE KEY-----", "private_key"),
        ("AKIAABCDEFGHIJKLMNOP", "aws_access_key"),
        ("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456", "github_token"),
        ("sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456", "openai_style_key"),
        ("eyJabcde.abcdefgh.ijklmnop", "jwt"),
        ("https://user:password@example.test/path", "uri_userinfo"),
        ("0123456789abcdefABCDEF012345", "hex_24"),
        ("QWxhZGRpbjpvcGVuIHNlc2FtZTEyMzQ1", "base64_24"),
    ],
)
def test_configured_secret_shapes_are_redacted(text: str, pattern_name: str):
    finding = classify_sensitive_text(text, CONFIG)

    assert finding is not None
    assert finding.reason_code == "credential_pattern"
    assert finding.category == pattern_name
    assert finding.evidence_sha256
    assert text not in repr(finding)


def test_high_entropy_whole_literal_is_rejected_but_low_entropy_literal_is_not():
    high = classify_sensitive_text("'aB3$dE5&gH7*jK9!'", CONFIG)
    low = classify_sensitive_text("'aaaaaaaaaaaaaaaa'", CONFIG)

    assert high is not None
    assert high.reason_code == "high_entropy_literal"
    assert low is None


@pytest.mark.parametrize(
    "text",
    ["src/private/project_module.py", "owner/private-repository"],
)
def test_path_or_repository_material_is_rejected(text: str):
    finding = classify_sensitive_text(
        text,
        CONFIG,
        repository="owner/private-repository",
    )

    assert finding is not None
    assert finding.reason_code == "path_or_repository_literal"


def test_unique_project_symbol_is_rejected_without_exposing_it():
    symbol = "BossCraftsOnlyInternalWidget"

    finding = classify_sensitive_text(
        symbol,
        CONFIG,
        document_count=1,
        repository_count=1,
    )

    assert finding is not None
    assert finding.reason_code == "unique_project_symbol"
    assert symbol not in repr(finding)
