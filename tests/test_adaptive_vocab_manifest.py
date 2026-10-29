from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bcnlt.adaptive_vocab.candidates import (
    CandidateDiscoveryReport,
    CandidateRate,
    MacroCandidate,
    PayloadClassScore,
)
from bcnlt.adaptive_vocab.config import AdaptiveVocabularyConfig
from bcnlt.adaptive_vocab.manifest import (
    build_inventories,
    load_adaptive_manifest,
    write_adaptive_manifest,
)
from bcnlt.tokenizer_manifest import BaseTokenizerIdentity, build_manifest


class _BaseTokenizer:
    def __init__(self, size: int = 2048):
        self.vocab_size = size
        self._vocab = {f"token-{index}": index for index in range(size)}

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False):
        return []

    def decode(self, ids):
        return ""


def _v1_manifest():
    base = _BaseTokenizer()
    identity = BaseTokenizerIdentity.from_files(
        name="fixture/base",
        revision="a" * 40,
        files={"tokenizer.json": b"fixture"},
    )
    return build_manifest(base, identity)


def _candidate(index: int) -> MacroCandidate:
    return MacroCandidate(
        expansion=(index, index + 1),
        length=2,
        score=float(10_000 - index),
        languages=("javascript", "python", "typescript"),
        repositories=("repo/a", "repo/b", "repo/c"),
        documents=tuple(f"doc-{item}" for item in range(10)),
        raw_occurrences=10,
        capped_occurrences=10,
        repository_counts=(),
        document_counts=(),
        rates=(CandidateRate("python", "surface", 10, 100, 100_000.0),),
        payload_class_scores=(PayloadClassScore("surface", 100_000.0),),
        eligible=True,
        reason_codes=(),
        sensitive_evidence=(),
    )


def _report(count: int) -> CandidateDiscoveryReport:
    return CandidateDiscoveryReport(
        candidates=tuple(_candidate(index) for index in range(count)),
        observation_count=count,
        eligible_base_tokens=(("python", "surface", count * 2),),
        lineage_sha256="d" * 64,
    )


def _config():
    return AdaptiveVocabularyConfig.load("configs/bcnlt_adaptive_vocab_v5.json")


def test_inventory_budgets_materialize_only_when_fully_supported():
    v1 = _v1_manifest()
    with pytest.raises(ValueError, match="insufficient_eligible_candidates"):
        build_inventories(_report(255), v1, _config())

    assert [len(item.macros) for item in build_inventories(_report(300), v1, _config())] == [256]
    assert [len(item.macros) for item in build_inventories(_report(700), v1, _config())] == [256, 512]
    assert [len(item.macros) for item in build_inventories(_report(1024), v1, _config())] == [256, 512, 1024]


def test_macro_ids_are_contiguous_after_v1_in_candidate_rank_order():
    v1 = _v1_manifest()
    manifest = build_inventories(_report(300), v1, _config())[0]

    assert manifest.base_vocab_size == v1.base_vocab_size
    assert manifest.v1_vocab_size == v1.vocab_size
    assert [item.id for item in manifest.macros] == list(
        range(v1.vocab_size, v1.vocab_size + 256)
    )
    assert [item.expansion for item in manifest.macros[:3]] == [(0, 1), (1, 2), (2, 3)]
    assert manifest.vocab_size == v1.vocab_size + 256


def _rewrite(path: Path, mutate, *, canonical: bool = True, rehash: bool = True):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    if rehash:
        body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
        payload["manifest_sha256"] = hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    text = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if canonical
        else json.dumps(payload, indent=2)
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


@pytest.mark.parametrize(
    ("name", "mutate", "message", "rehash"),
    [
        ("digest", lambda p: p.__setitem__("config_sha256", "f" * 64), "digest", True),
        ("duplicate-id", lambda p: p["macros"][1].__setitem__("id", p["macros"][0]["id"]), "unique", True),
        ("duplicate-name", lambda p: p["macros"][1].__setitem__("name", p["macros"][0]["name"]), "unique", True),
        ("inside-v1", lambda p: p["macros"][0].__setitem__("id", p["v1_vocab_size"] - 1), "contiguous", True),
        ("noncontiguous", lambda p: p["macros"][-1].__setitem__("id", p["macros"][-1]["id"] + 1), "contiguous", True),
        ("one-token", lambda p: p["macros"][0].__setitem__("expansion", [1]), "two base tokens", True),
        ("structural", lambda p: p["macros"][0].__setitem__("expansion", [1, p["base_vocab_size"]]), "base-token", True),
        ("macro-reference", lambda p: p["macros"][0].__setitem__("expansion", [1, p["v1_vocab_size"]]), "base-token", True),
        ("unknown-field", lambda p: p["macros"][0].__setitem__("allowed_fields", ["mystery"]), "allowed field", True),
    ],
)
def test_corrupt_manifests_are_rejected(
    tmp_path: Path, name: str, mutate, message: str, rehash: bool
):
    v1 = _v1_manifest()
    path = tmp_path / f"{name}.json"
    original = build_inventories(_report(300), v1, _config())[0]
    write_adaptive_manifest(path, original)
    _rewrite(path, mutate, rehash=rehash)

    with pytest.raises(ValueError, match=message):
        load_adaptive_manifest(
            path,
            v1,
            expected_candidate_report_sha256=original.candidate_report_sha256,
            expected_config_sha256=original.config_sha256,
            expected_lineage_sha256=original.lineage_sha256,
        )


def test_noncanonical_manifest_json_is_rejected(tmp_path: Path):
    v1 = _v1_manifest()
    path = tmp_path / "manifest.json"
    write_adaptive_manifest(path, build_inventories(_report(300), v1, _config())[0])
    _rewrite(path, lambda payload: None, canonical=False)

    with pytest.raises(ValueError, match="canonical"):
        load_adaptive_manifest(path, v1)
