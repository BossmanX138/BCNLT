from __future__ import annotations

import ast
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable


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
    try:
        ast.parse(content, filename=relative_path)
    except (SyntaxError, ValueError, TypeError):
        return FilterDecision(False, "python_parse_failure", license_spdx, content_sha256)
    return FilterDecision(True, None, license_spdx, content_sha256)


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
