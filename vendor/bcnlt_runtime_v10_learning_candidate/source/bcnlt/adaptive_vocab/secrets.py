from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from .config import AdaptiveVocabularyConfig


@dataclass(frozen=True, slots=True)
class SensitiveTextFinding:
    reason_code: str
    category: str
    evidence_sha256: str
    character_count: int


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _finding(reason: str, category: str, text: str) -> SensitiveTextFinding:
    return SensitiveTextFinding(
        reason_code=reason,
        category=category,
        evidence_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
    )


def classify_sensitive_text(
    text: str,
    config: AdaptiveVocabularyConfig,
    *,
    repository: str | None = None,
    document_count: int | None = None,
    repository_count: int | None = None,
) -> SensitiveTextFinding | None:
    for pattern in config.secret_patterns:
        match = re.search(pattern.expression, text)
        if match is None:
            continue
        matched = match.group(0)
        if pattern.name in {"hex_24", "base64_24"} and _entropy(matched) < config.minimum_secret_entropy:
            continue
        if pattern.name == "base64_24" and not (
            any(character.isupper() for character in matched)
            and any(character.islower() for character in matched)
            and any(character.isdigit() or character in "+/=" for character in matched)
        ):
            continue
        if match is not None:
            return _finding("credential_pattern", pattern.name, text)

    stripped = text.strip()
    if (
        len(stripped) >= 2
        and stripped[0] == stripped[-1]
        and stripped[0] in {"'", '"'}
    ):
        literal = stripped[1:-1]
        if len(literal) >= 16 and _entropy(literal) >= config.minimum_secret_entropy:
            return _finding("high_entropy_literal", "whole_literal", text)

    normalized_repository = (repository or "").strip("/\\")
    if normalized_repository and normalized_repository in text:
        return _finding("path_or_repository_literal", "repository", text)
    if re.search(
        r"(?:^|[/\\])(?:src|lib|include|home|Users|workspace)[/\\][^\s]+",
        text,
    ):
        return _finding("path_or_repository_literal", "path", text)

    if (
        document_count is not None
        and repository_count is not None
        and document_count <= 1
        and repository_count <= 1
        and len(text) >= 12
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text)
        and any(character.isupper() for character in text[1:])
    ):
        return _finding("unique_project_symbol", "identifier", text)
    return None
