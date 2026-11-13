"""Prove the V6 held-out delta is limited to canonical recent history."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V5 = PROJECT_ROOT / "artifacts/corpora/bcnlt-v5-65536-heldout-admission-v1/admitted.jsonl"
V6 = PROJECT_ROOT / "artifacts/corpora/bcnlt-v6-65536-heldout-admission-v2/admitted.jsonl"
OUTPUT = PROJECT_ROOT / "artifacts/audits/v5-v6-heldout-history-delta-v2.json"


def _load(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    v5_rows = _load(V5)
    v6_rows = _load(V6)
    if [row["document_id"] for row in v5_rows] != [row["document_id"] for row in v6_rows]:
        raise ValueError("held-out document order or identity changed")

    changed_documents: list[dict[str, object]] = []
    changed_fields: Counter[str] = Counter()
    changed_records = 0
    non_history_changed_records = 0
    for v5_row, v6_row in zip(v5_rows, v6_rows, strict=True):
        v5_records = [json.loads(line) for line in str(v5_row["canonical_jsonl"]).splitlines()]
        v6_records = [json.loads(line) for line in str(v6_row["canonical_jsonl"]).splitlines()]
        if len(v5_records) != len(v6_records):
            raise ValueError(f"record count changed: {v5_row['relative_path']}")
        document_changed = 0
        for v5_record, v6_record in zip(v5_records, v6_records, strict=True):
            fields = sorted(
                key
                for key in set(v5_record) | set(v6_record)
                if v5_record.get(key) != v6_record.get(key)
            )
            if not fields:
                continue
            changed_records += 1
            document_changed += 1
            changed_fields.update(fields)
            if fields != ["recent_history"]:
                non_history_changed_records += 1
        if document_changed:
            changed_documents.append(
                {
                    "language": v5_row["language"],
                    "relative_path": v5_row["relative_path"],
                    "record_count": len(v5_records),
                    "changed_record_count": document_changed,
                }
            )

    unsigned = {
        "schema_version": "bosscrafts-v5-v6-heldout-history-delta-v2",
        "v5_records_sha256": _sha256(V5),
        "v6_records_sha256": _sha256(V6),
        "document_count": len(v5_rows),
        "changed_document_count": len(changed_documents),
        "changed_documents": changed_documents,
        "changed_record_count": changed_records,
        "changed_field_counts": dict(sorted(changed_fields.items())),
        "non_history_changed_record_count": non_history_changed_records,
        "expected_delta_only": non_history_changed_records == 0,
    }
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    payload = {
        **unsigned,
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if non_history_changed_records:
        raise ValueError("V6 changed fields outside recent_history")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
