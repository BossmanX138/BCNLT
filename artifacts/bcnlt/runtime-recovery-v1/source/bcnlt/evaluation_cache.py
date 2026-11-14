from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BaselineCacheKey:
    base_model_sha256: str
    tokenizer_manifest_sha256: str
    packed_dataset_manifest_sha256: str
    suite_sha256: str
    selection_sha256: str
    seed: int

    def __post_init__(self) -> None:
        for name in (
            "base_model_sha256",
            "tokenizer_manifest_sha256",
            "packed_dataset_manifest_sha256",
            "suite_sha256",
            "selection_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_baseline_cache(
    path_value: str | Path,
    *,
    expected_key: BaselineCacheKey,
) -> dict[str, Any] | None:
    path = Path(path_value)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "bcnlt-baseline-cache-v1":
        raise ValueError("unsupported BCNLT baseline cache schema")
    if payload.get("key") != expected_key.to_dict():
        return None
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("BCNLT baseline cache has no baseline object")
    return baseline


def write_baseline_cache(
    path_value: str | Path,
    *,
    key: BaselineCacheKey,
    baseline: Mapping[str, Any],
) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "bcnlt-baseline-cache-v1",
        "key": key.to_dict(),
        "baseline": dict(baseline),
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def selection_sha256(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> str:
    metadata = getattr(records, "metadata", None)
    digest = hashlib.sha256()
    for position, index in enumerate(indices):
        row = metadata(index) if callable(metadata) else records[index]
        identity = {
            "position": position,
            "source_index": index,
            "record_id": str(row.get("record_id") or ""),
            "language": str(row.get("language") or ""),
            "family": str(row.get("family") or ""),
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()
