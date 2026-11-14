from __future__ import annotations

import ast
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Protocol

from bcnlt.languages import LANGUAGE_BY_ID


APPROVED_SPDX = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "0BSD",
        "Unlicense",
        "CC0-1.0",
    }
)
MIN_DOCUMENT_BYTES = 512
MAX_DOCUMENT_BYTES = 200 * 1024
MAX_LINE_LENGTH = 2_000
_GENERATED_PARTS = {
    "vendor",
    "vendors",
    "third_party",
    "third-party",
    "node_modules",
    "site-packages",
    "dist",
    "build",
}
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
_KNOWN_SECRET = re.compile(
    r"(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})"
)
_ASSIGNED_SECRET = re.compile(
    r"(?im)^\s*[A-Za-z_][A-Za-z0-9_]*(?:key|token|secret|password|credential)[A-Za-z0-9_]*\s*=\s*['\"]([^'\"]{32,})['\"]"
)


@dataclass(frozen=True)
class FilterDecision:
    accepted: bool
    reason_code: str | None
    license_spdx: str | None
    content_sha256: str
    event_names: tuple[str, ...] = ()
    mapped_ratio: float | None = None
    unknown_ratio: float | None = None
    scopes_balanced: bool | None = None
    links_valid: bool | None = None
    unknown_node_types: tuple[str, ...] = ()


class _SemanticAdapter(Protocol):
    def parse(
        self,
        source: str,
        *,
        language: str,
        path: str,
        dialect: str | None = None,
    ) -> object: ...

    def extract(self, document: object) -> object: ...


def normalize_license(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        pieces = [
            item.strip()
            for item in re.split(r"\s+(?:AND|OR)\s+|[,;/]", stripped, flags=re.IGNORECASE)
            if item.strip()
        ]
    elif isinstance(value, (list, tuple, set)):
        pieces = [str(item).strip() for item in value if str(item).strip()]
    else:
        return None
    if len(pieces) != 1 or pieces[0] not in APPROVED_SPDX:
        return None
    return pieces[0]


def evaluate_python_document(
    *,
    relative_path: str,
    content: str,
    license_value: object,
    holdout_hashes: Iterable[str] = (),
    allow_owner_provided: bool = False,
) -> FilterDecision:
    decision = _evaluate_shared_document_filters(
        relative_path=relative_path,
        content=content,
        license_value=license_value,
        holdout_hashes=holdout_hashes,
        allow_owner_provided=allow_owner_provided,
    )
    if decision is not None:
        return decision
    payload = content.encode("utf-8", errors="surrogatepass")
    content_sha256 = hashlib.sha256(payload).hexdigest()
    license_spdx = (
        "owner_provided"
        if allow_owner_provided and license_value == "owner_provided"
        else normalize_license(license_value)
    )
    try:
        ast.parse(content, filename=relative_path)
    except (SyntaxError, ValueError, TypeError):
        return FilterDecision(False, "python_parse_failure", license_spdx, content_sha256)
    return FilterDecision(True, None, license_spdx, content_sha256)


def evaluate_source_document(
    *,
    relative_path: str,
    content: str,
    license_value: object,
    language: str,
    dialect: str | None = None,
    holdout_hashes: Iterable[str] = (),
    allow_owner_provided: bool = False,
    adapter: _SemanticAdapter | None = None,
) -> FilterDecision:
    """Apply provenance, syntax, and semantic-coverage gates to one source file.

    Language is always explicit. Extension checks happen before parser selection so
    ambiguous dialect forms cannot silently enter the shared curriculum.
    """

    try:
        definition = LANGUAGE_BY_ID[language]
    except KeyError as exc:
        raise ValueError(f"Use an exact registered canonical language id, got: {language}") from exc

    suffix = PurePosixPath(relative_path.replace("\\", "/")).suffix.lower()
    if suffix in definition.deferred_extensions:
        return _early_rejection(
            content,
            license_value,
            "ambiguous_or_deferred_extension",
            allow_owner_provided=allow_owner_provided,
        )
    if suffix not in definition.accepted_extensions:
        return _early_rejection(
            content,
            license_value,
            "extension_not_accepted",
            allow_owner_provided=allow_owner_provided,
        )

    # Preserve the established Python-only API for callers that have not yet
    # opted into the carousel. Multilingual ingestion always supplies one
    # shared adapter explicitly.
    if adapter is None and language == "python":
        return evaluate_python_document(
            relative_path=relative_path,
            content=content,
            license_value=license_value,
            holdout_hashes=holdout_hashes,
            allow_owner_provided=allow_owner_provided,
        )

    decision = _evaluate_shared_document_filters(
        relative_path=relative_path,
        content=content,
        license_value=license_value,
        holdout_hashes=holdout_hashes,
        allow_owner_provided=allow_owner_provided,
    )
    if decision is not None:
        return decision
    payload = content.encode("utf-8", errors="surrogatepass")
    content_sha256 = hashlib.sha256(payload).hexdigest()
    license_spdx = (
        "owner_provided"
        if allow_owner_provided and license_value == "owner_provided"
        else normalize_license(license_value)
    )
    if adapter is None:
        from bcnlt.adapters.carousel import BCNLTMultilingualAdapter
        from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine

        adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    try:
        document = adapter.parse(
            content,
            language=language,
            path=relative_path,
            dialect=dialect,
        )
        extraction = adapter.extract(document)
    except (ValueError, TypeError, RuntimeError, OSError):
        return FilterDecision(False, "semantic_parse_failure", license_spdx, content_sha256)
    coverage = extraction.coverage
    if not coverage.admitted:
        return FilterDecision(
            False,
            "semantic_coverage_failure",
            license_spdx,
            content_sha256,
            tuple(unit.event for unit in extraction.units),
            float(coverage.mapped_ratio),
            float(coverage.unknown_ratio),
            bool(coverage.scopes_balanced),
            bool(coverage.links_valid),
            tuple(
                str(getattr(unit, "metadata", {})["unknown_node"])
                for unit in extraction.units
                if "unknown_node" in getattr(unit, "metadata", {})
            ),
        )
    return FilterDecision(
        True,
        None,
        license_spdx,
        content_sha256,
        tuple(unit.event for unit in extraction.units),
        float(coverage.mapped_ratio),
        float(coverage.unknown_ratio),
        bool(coverage.scopes_balanced),
        bool(coverage.links_valid),
        tuple(
            str(getattr(unit, "metadata", {})["unknown_node"])
            for unit in extraction.units
            if "unknown_node" in getattr(unit, "metadata", {})
        ),
    )


def _early_rejection(
    content: str,
    license_value: object,
    reason_code: str,
    *,
    allow_owner_provided: bool,
) -> FilterDecision:
    payload = content.encode("utf-8", errors="surrogatepass")
    content_sha256 = hashlib.sha256(payload).hexdigest()
    license_spdx = (
        "owner_provided"
        if allow_owner_provided and license_value == "owner_provided"
        else normalize_license(license_value)
    )
    return FilterDecision(False, reason_code, license_spdx, content_sha256)


def _evaluate_shared_document_filters(
    *,
    relative_path: str,
    content: str,
    license_value: object,
    holdout_hashes: Iterable[str],
    allow_owner_provided: bool,
) -> FilterDecision | None:
    payload = content.encode("utf-8", errors="surrogatepass")
    content_sha256 = hashlib.sha256(payload).hexdigest()
    license_spdx = (
        "owner_provided"
        if allow_owner_provided and license_value == "owner_provided"
        else normalize_license(license_value)
    )
    if license_spdx is None:
        return FilterDecision(False, "license_not_approved", None, content_sha256)
    if content_sha256 in set(holdout_hashes):
        return FilterDecision(
            False, "evaluation_holdout_collision", license_spdx, content_sha256
        )
    normalized_path = PurePosixPath(relative_path.replace("\\", "/"))
    lowered_parts = {part.lower() for part in normalized_path.parts}
    lowered_name = normalized_path.name.lower()
    if lowered_parts.intersection(_GENERATED_PARTS) or "generated" in lowered_name:
        return FilterDecision(False, "generated_or_vendor_path", license_spdx, content_sha256)
    if not MIN_DOCUMENT_BYTES <= len(payload) <= MAX_DOCUMENT_BYTES:
        return FilterDecision(False, "size_out_of_range", license_spdx, content_sha256)
    if any(ord(character) < 32 and character not in "\n\r\t" for character in content):
        return FilterDecision(False, "control_character", license_spdx, content_sha256)
    if any(len(line) > MAX_LINE_LENGTH for line in content.splitlines()):
        return FilterDecision(False, "pathological_line_length", license_spdx, content_sha256)
    if _contains_secret(content):
        return FilterDecision(False, "secret_detected", license_spdx, content_sha256)
    return None


def _contains_secret(content: str) -> bool:
    if _PRIVATE_KEY.search(content) or _KNOWN_SECRET.search(content):
        return True
    for match in _ASSIGNED_SECRET.finditer(content):
        candidate = match.group(1)
        if _shannon_entropy(candidate) >= 4.0:
            return True
    return False


def _shannon_entropy(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())
