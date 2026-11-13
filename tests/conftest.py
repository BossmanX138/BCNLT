from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.current_corpus import CurrentCorpusAdmission, SourceCandidate
from bosscrafts_model.heldout_semantic_review import HeldoutSemanticReviewQueue
from bosscrafts_model.tokenizer import NativeTokenizerBinding

_CURRENT_RUNTIME_SOURCES = (
    (
        "current-runtime-js-00050",
        "fixture/add.js",
        "function add(a, b) {\n  return a + b;\n}\n",
    ),
    (
        "current-runtime-js-00111",
        "fixture/multiply.js",
        "function multiply(a, b) {\n  return a * b;\n}\n",
    ),
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@pytest.fixture(scope="session")
def current_runtime_semantic_evidence(tmp_path_factory: pytest.TempPathFactory):
    """Materialize genuine current-runtime admission and review evidence.

    The chosen document IDs deterministically select each function's explicit
    return record for held-out review while assigning both documents to the
    test split.  Nothing from the preserved historical evidence is rewritten.
    """

    root = tmp_path_factory.mktemp("current-runtime-semantic-evidence")
    admission_root = root / "admission"
    review_root = root / "review"
    binding = NativeTokenizerBinding.load()
    candidates = []
    for document_id, relative_path, source in _CURRENT_RUNTIME_SOURCES:
        candidates.append(
            SourceCandidate(
                document_id=document_id,
                language="javascript",
                source=source,
                relative_path=relative_path,
                content_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                license_spdx="MIT",
                attribution="BossCrafts current-runtime regression fixture",
                ownership="BossCrafts",
                source_provider="local-test-fixture",
                source_name="current-runtime-semantic-evidence",
                source_revision="v1",
                repository_identity="BossmanX138/BCNLT",
                upstream_url="https://github.com/BossmanX138/BCNLT",
            )
        )

    with BCNLTSourcePipeline(binding, learning_mode="ephemeral") as pipeline:
        admission = CurrentCorpusAdmission(pipeline).materialize(
            candidates,
            admission_root,
        )
    assert admission.admitted_count == 2
    assert admission.rejected_count == 0

    review = HeldoutSemanticReviewQueue(binding, per_language=2)
    review.materialize_admission_only(admission_root, review_root)
    rows = [
        json.loads(line)
        for line in (review_root / "review_queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 2
    assert all(row["canonical_record"]["event"] == "RETURN" for row in rows)
    return SimpleNamespace(
        admission_root=admission_root,
        review_root=review_root,
        selected_review_id=rows[0]["review_id"],
    )


@pytest.fixture(scope="session")
def stale_runtime_semantic_evidence(
    current_runtime_semantic_evidence,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Create valid, content-addressed review evidence with stale lineage."""

    root = tmp_path_factory.mktemp("stale-runtime-semantic-evidence")
    review_root = root / "review"
    review_root.mkdir()
    source_root = current_runtime_semantic_evidence.review_root
    rows = [
        json.loads(line)
        for line in (source_root / "review_queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    stale_identity = "0" * 64
    for row in rows:
        row["parser_runtime_identity_sha256"] = stale_identity
    queue_text = "".join(f"{_canonical(row)}\n" for row in rows)
    (review_root / "review_queue.jsonl").write_text(queue_text, encoding="utf-8")

    manifest = json.loads(
        (source_root / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["parser_runtime_identity_sha256"] = stale_identity
    manifest["queue_sha256"] = hashlib.sha256(queue_text.encode()).hexdigest()
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical(manifest).encode()
    ).hexdigest()
    (review_root / "manifest.json").write_text(
        f"{_canonical(manifest)}\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        admission_root=current_runtime_semantic_evidence.admission_root,
        review_root=review_root,
        selected_review_id=current_runtime_semantic_evidence.selected_review_id,
    )
