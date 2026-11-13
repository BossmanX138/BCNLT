"""Carry V4 held-out decisions only after exact V5 semantic equivalence."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.bcnlt_source_pipeline import (  # noqa: E402
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.heldout_semantic_review import (  # noqa: E402
    HeldoutSemanticReviewAdjudicator,
)

BASELINE_IDENTITY = "2a8f439fb5e185240ee5b672c289fee6c1702bf69387b1d01d049814bcbda650"
BASELINE_ADMISSION = (
    PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-v4-65536-heldout-admission-v1"
)
CANDIDATE_ADMISSION = (
    PROJECT_ROOT / "artifacts" / "corpora" / "bcnlt-v5-65536-heldout-admission-v1"
)
BASELINE_ADJUDICATED = (
    PROJECT_ROOT
    / "artifacts"
    / "reviews"
    / "bcnlt-v4-65536-heldout-all-language-adjudicated-v1"
)
CANDIDATE_QUEUE = (
    PROJECT_ROOT / "artifacts" / "reviews" / "bcnlt-v5-65536-heldout-review-v1"
)
CANDIDATE_ADJUDICATED = (
    PROJECT_ROOT
    / "artifacts"
    / "reviews"
    / "bcnlt-v5-65536-heldout-all-language-adjudicated-v1"
)
MAX_JSONL_LINE_BYTES = 64 * 1024 * 1024


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError(f"manifest exceeds bounded size: {path}")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if type(value) is not dict:
        raise ValueError(f"manifest is not an object: {path}")
    return value


def _verify_signed_manifest(path: Path) -> tuple[dict[str, object], str]:
    manifest = _load_json(path)
    claimed = manifest.pop("manifest_sha256", None)
    if type(claimed) is not str or claimed != _sha256_text(_canonical_json(manifest)):
        raise ValueError(f"manifest SHA-256 mismatch: {path}")
    return manifest, claimed


def _iter_jsonl(path: Path):
    with path.open("rb") as stream:
        while True:
            raw = stream.readline(MAX_JSONL_LINE_BYTES + 1)
            if not raw:
                return
            if len(raw) > MAX_JSONL_LINE_BYTES or not raw.endswith(b"\n"):
                raise ValueError(f"invalid bounded JSONL framing: {path}")
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
            if type(value) is not dict:
                raise ValueError(f"JSONL row is not an object: {path}")
            yield value


def _verify_admission_equivalence() -> tuple[int, str, str, str]:
    baseline_manifest, baseline_manifest_sha = _verify_signed_manifest(
        BASELINE_ADMISSION / "manifest.json"
    )
    candidate_manifest, candidate_manifest_sha = _verify_signed_manifest(
        CANDIDATE_ADMISSION / "manifest.json"
    )
    baseline_file = BASELINE_ADMISSION / str(baseline_manifest["records_file"])
    candidate_file = CANDIDATE_ADMISSION / str(candidate_manifest["records_file"])
    if _sha256_file(baseline_file) != baseline_manifest.get("records_sha256"):
        raise ValueError("V4 admission records SHA-256 mismatch")
    if _sha256_file(candidate_file) != candidate_manifest.get("records_sha256"):
        raise ValueError("V5 admission records SHA-256 mismatch")
    semantic_digest = hashlib.sha256()
    count = 0
    for count, pair in enumerate(
        itertools.zip_longest(_iter_jsonl(baseline_file), _iter_jsonl(candidate_file)),
        1,
    ):
        baseline_row, candidate_row = pair
        if baseline_row is None or candidate_row is None:
            raise ValueError("V4/V5 admission row counts differ")
        if baseline_row.pop("parser_runtime_identity_sha256", None) != BASELINE_IDENTITY:
            raise ValueError("V4 admission parser identity mismatch")
        if (
            candidate_row.pop("parser_runtime_identity_sha256", None)
            != PARSER_RUNTIME_IDENTITY_SHA256
        ):
            raise ValueError("V5 admission parser identity mismatch")
        if baseline_row != candidate_row:
            raise ValueError(
                f"V4/V5 semantic admission differs at row {count}: "
                f"{baseline_row.get('document_id')}"
            )
        semantic_digest.update(_canonical_json(baseline_row).encode("utf-8"))
        semantic_digest.update(b"\n")
    if count != 168:
        raise ValueError(f"expected 168 equivalent held-out documents, found {count}")
    return (
        count,
        semantic_digest.hexdigest(),
        baseline_manifest_sha,
        candidate_manifest_sha,
    )


def _semantic_review_key(item: dict[str, object]) -> str:
    return _canonical_json(
        {
            "binding_sha256": item.get("binding_sha256"),
            "canonical_record": item.get("canonical_record"),
            "document_id": item.get("document_id"),
            "family": item.get("family"),
            "language": item.get("language"),
            "record_index": item.get("record_index"),
            "semantic_spec_sha256": item.get("semantic_spec_sha256"),
            "source_excerpt": item.get("source_excerpt"),
            "source_sha256": item.get("source_sha256"),
            "split": item.get("split"),
            "tokenizer_sha256": item.get("tokenizer_sha256"),
        }
    )


def main() -> int:
    document_count, semantic_sha, baseline_admission_sha, candidate_admission_sha = (
        _verify_admission_equivalence()
    )
    baseline_manifest, baseline_adjudication_sha = _verify_signed_manifest(
        BASELINE_ADJUDICATED / "manifest.json"
    )
    baseline_file = BASELINE_ADJUDICATED / str(
        baseline_manifest["adjudicated_file"]
    )
    if _sha256_file(baseline_file) != baseline_manifest.get("adjudicated_sha256"):
        raise ValueError("V4 adjudicated queue SHA-256 mismatch")
    source_by_key: dict[str, dict[str, object]] = {}
    for item in _iter_jsonl(baseline_file):
        if item.get("parser_runtime_identity_sha256") != BASELINE_IDENTITY:
            raise ValueError("V4 review item parser identity mismatch")
        key = _semantic_review_key(item)
        if key in source_by_key:
            raise ValueError("V4 adjudication contains duplicate semantic review items")
        source_by_key[key] = item

    candidate_queue_manifest, candidate_queue_manifest_sha = _verify_signed_manifest(
        CANDIDATE_QUEUE / "manifest.json"
    )
    candidate_queue_file = CANDIDATE_QUEUE / str(
        candidate_queue_manifest["queue_file"]
    )
    if _sha256_file(candidate_queue_file) != candidate_queue_manifest.get("queue_sha256"):
        raise ValueError("V5 review queue SHA-256 mismatch")
    decisions: list[dict[str, object]] = []
    matched: set[str] = set()
    for item in _iter_jsonl(candidate_queue_file):
        if item.get("parser_runtime_identity_sha256") != PARSER_RUNTIME_IDENTITY_SHA256:
            raise ValueError("V5 review item parser identity mismatch")
        key = _semantic_review_key(item)
        source = source_by_key.get(key)
        if source is None:
            raise ValueError("V5 review item has no exact V4 semantic match")
        matched.add(key)
        decisions.append(
            {
                "review_id": item["review_id"],
                "decision": source["review_status"],
                "reviewer": f"{source['reviewer']}+v5-exact-equivalence",
                "notes": (
                    f"{source['review_notes']} Carried forward only after exact "
                    "168-document V4/V5 semantic-content equivalence."
                ),
                "rubric": source["rubric"],
            }
        )
    if matched != set(source_by_key) or len(decisions) != 28:
        raise ValueError("V4/V5 review queues are not a complete 28-item bijection")

    result = HeldoutSemanticReviewAdjudicator().materialize(
        CANDIDATE_QUEUE,
        decisions,
        CANDIDATE_ADJUDICATED,
    )
    unsigned_equivalence = {
        "schema_version": "bosscrafts-heldout-runtime-equivalence-v1",
        "baseline_parser_runtime_identity_sha256": BASELINE_IDENTITY,
        "candidate_parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "baseline_admission_manifest_sha256": baseline_admission_sha,
        "candidate_admission_manifest_sha256": candidate_admission_sha,
        "baseline_adjudication_manifest_sha256": baseline_adjudication_sha,
        "candidate_review_manifest_sha256": candidate_queue_manifest_sha,
        "candidate_adjudication_manifest_sha256": result.manifest_sha256,
        "equivalent_document_count": document_count,
        "equivalent_review_item_count": len(decisions),
        "semantic_content_sha256": semantic_sha,
        "comparison_policy": "exact canonical JSON equality after removing only parser_runtime_identity_sha256 from each admission row",
        "review_transfer_policy": "bijective equality over source excerpt, canonical record, document, record index, language, split, family, source hash, semantic spec, tokenizer and binding",
        "semantic_review_passed": result.semantic_review_passed,
        "training_ready": False,
        "training_blockers": [
            "review evidence is bounded to 168 held-out documents",
            "full replay and model evaluation are not complete",
        ],
    }
    equivalence_sha = _sha256_text(_canonical_json(unsigned_equivalence))
    equivalence = {**unsigned_equivalence, "manifest_sha256": equivalence_sha}
    equivalence_path = CANDIDATE_ADJUDICATED / "equivalence-manifest.json"
    equivalence_path.write_text(
        f"{_canonical_json(equivalence)}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "adjudication_manifest_sha256": result.manifest_sha256,
                "decision_counts": result.decision_counts,
                "equivalence_manifest_sha256": equivalence_sha,
                "semantic_content_sha256": semantic_sha,
                "semantic_review_passed": result.semantic_review_passed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
