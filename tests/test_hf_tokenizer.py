from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcnlt.hf_tokenizer import (
    HuggingFaceBaseTokenizer,
    identity_from_directory,
    resolve_tokenizer_source,
)
from bcnlt.adapters.python import PythonAdapter
from bcnlt.ir import IRKind
from bcnlt.tokenizer import BCNLTTokenizer
from bcnlt.tokenizer_manifest import build_manifest
from bcnlt.wildcard import ProvisionalMeaning, WildcardMemory
from tests.tokenizer_fixtures import write_byte_level_tokenizer


def test_scientific_learning_publishes_extended_core_without_changing_ids(tmp_path: Path):
    from dataclasses import replace

    from bcnlt.adapters.python import build_python_spec
    from bcnlt.scientific_learning import (
        CodingExperimentFixture,
        HypothesisProjection,
    )
    from bcnlt.wildcard import DefinitionState

    source = "ready = True\nassert ready\n"
    learned_path = tmp_path / "scientific-learned.jsonl"
    spec_sha = build_python_spec().canonical_hash
    seed_adapter = PythonAdapter()
    seed_document = seed_adapter.parse(source, path="seed.py")
    seed = next(
        unit for unit in seed_adapter.to_units(seed_document)
        if unit.event == "UNKNOWN_SEMANTIC" and unit.ast_relation == "Assert"
    )
    memory = WildcardMemory(
        learned_path,
        language="python",
        semantic_spec_sha256=spec_sha,
    )
    units = [
        replace(
            seed,
            scope=f"module.case_{index}",
            span=(index * 10 + 2, 0, index * 10 + 2, 12),
            recent_history=(f"DEFINE:case_{index}",),
        )
        for index in (1, 2, 3)
    ]
    for index, unit in enumerate(units, start=1):
        memory.remember(
            unit,
            ProvisionalMeaning(
                category="error",
                hypotheses=("assert may raise an assertion error",),
                evidence=("language reference fixture",),
                confidence=0.64,
                target_event="RAISE",
                target_ir_kind=IRKind.ERROR.value,
            ),
            document_identity=f"assert-{index}.py",
        )

    tokenizer, base_path = _build_tokenizer_with_learning(tmp_path, learned_path)
    frozen_manifest = tokenizer.manifest.to_dict()
    provisional = tokenizer.decode_records(
        tokenizer.encode_source(source, path="runtime.py")
    )
    assert any(record.event == "UNKNOWN_SEMANTIC" for record in provisional)

    for index, unit in enumerate(units, start=1):
        record = memory.lookup(unit)
        assert record is not None
        control = HypothesisProjection(
            signature=record.signature,
            revision=record.hypothesis_revision,
            target_event="BRANCH",
            target_ir_kind=IRKind.BRANCH.value,
            language=record.language,
            semantic_spec_sha256=record.semantic_spec_sha256,
        )
        memory.run_scientific_experiment(
            unit,
            source=f"assert ready  # independent fixture {index}\n",
            fixture=CodingExperimentFixture(
                experiment_id=f"assert-exp-{index}",
                fixture_id=f"trusted/assert/{index}",
                fixture_version=1,
                domain="python.assert.raise-category",
                document_identity=f"assert-{index}.py",
                oracle_id="python-language-reference-assert",
                oracle_kind="language_spec",
                oracle_version="3.12",
                operation="raise_signal",
                inputs={
                    "category": "AssertionError",
                    "message": f"assertion-case-{index}",
                },
                expected_output={
                    "raised": "AssertionError",
                    "message": f"assertion-case-{index}",
                },
            ),
            negative_control=control,
        )

    assert memory.lookup(units[0]).definition_state is DefinitionState.VALIDATED
    tokenizer.reload_learned_state()
    learned_ids = tokenizer.encode_source(source, path="runtime.py")
    learned = tokenizer.decode_records(learned_ids)
    extended_record = next(
        record
        for record in learned
        if dict(record.metadata).get("extended_core_state") == "validated"
    )
    learned_metadata = dict(extended_record.metadata)
    library = tokenizer.extended_core_library()
    entry = library.by_signature(memory.signature_for(units[0]))

    assert entry is not None
    assert extended_record.event == "UNKNOWN_SEMANTIC"
    assert extended_record.ir is not None
    assert extended_record.ir.kind == IRKind.EXTENSION.value
    assert learned_metadata["extended_core_id"] == entry.extended_core_id
    assert learned_metadata["extended_core_state"] == "validated"
    assert entry.common_core_projection.event == "RAISE"
    assert entry.common_core_projection.ir_kind == IRKind.ERROR.value
    assert tokenizer.manifest.token_to_id["<|bcnlt:event:unknown_semantic|>"] in learned_ids
    assert tokenizer.manifest.token_to_id["<|bcnlt:event:raise|>"] not in learned_ids
    assert tokenizer.manifest.to_dict() == frozen_manifest

    artifact = tmp_path / "scientific-artifact"
    tokenizer.save_pretrained(artifact)
    frozen_tokenizer = BCNLTTokenizer.from_pretrained(artifact, base_path)
    frozen_records = frozen_tokenizer.decode_records(
        frozen_tokenizer.encode_source(source, path="runtime.py")
    )
    assert any(
        dict(record.metadata).get("extended_core_id") == entry.extended_core_id
        and record.event == "UNKNOWN_SEMANTIC"
        for record in frozen_records
    )
    assert tuple(frozen_tokenizer.extended_core_library()) == tuple(library)

    record = memory.lookup(units[0])
    assert record is not None
    memory.run_scientific_experiment(
        units[0],
        source="assert ready  # contradictory fixture\n",
        fixture=CodingExperimentFixture(
            experiment_id="assert-exp-contradiction",
            fixture_id="trusted/assert/contradiction",
            fixture_version=1,
            domain="python.assert.raise-category",
            document_identity="assert-1.py",
            oracle_id="python-language-reference-assert",
            oracle_kind="language_spec",
            oracle_version="3.12",
            operation="raise_signal",
            inputs={"category": "AssertionError"},
            expected_output={"raised": "ValueError"},
        ),
        negative_control=HypothesisProjection(
            signature=record.signature,
            revision=record.hypothesis_revision,
            target_event="BRANCH",
            target_ir_kind=IRKind.BRANCH.value,
            language=record.language,
            semantic_spec_sha256=record.semantic_spec_sha256,
        ),
    )
    tokenizer.reload_learned_state()
    revoked = tokenizer.decode_records(tokenizer.encode_source(source, path="runtime.py"))

    assert any(record.event == "UNKNOWN_SEMANTIC" for record in revoked)
    assert all("extended_core_state" not in dict(record.metadata) for record in revoked)
    assert len(tokenizer.extended_core_library()) == 0
    assert tokenizer.manifest.to_dict() == frozen_manifest
    still_frozen = frozen_tokenizer.decode_records(
        frozen_tokenizer.encode_source(source, path="runtime.py")
    )
    assert any(
        dict(record.metadata).get("extended_core_id") == entry.extended_core_id
        and record.event == "UNKNOWN_SEMANTIC"
        for record in still_frozen
    )


def _build_tokenizer(tmp_path: Path) -> tuple[BCNLTTokenizer, Path]:
    base_path = write_byte_level_tokenizer(tmp_path / "base")
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    identity = identity_from_directory(base_path, name="fixture/base", revision="local-v1")
    return BCNLTTokenizer(base, build_manifest(base, identity)), base_path


def _build_tokenizer_with_learning(
    tmp_path: Path, learned_path: Path
) -> tuple[BCNLTTokenizer, Path]:
    base_path = write_byte_level_tokenizer(tmp_path / "base-learning")
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    identity = identity_from_directory(
        base_path, name="fixture/base", revision="local-v1"
    )
    return (
        BCNLTTokenizer(
            base,
            build_manifest(base, identity),
            learned_memory_path=learned_path,
        ),
        base_path,
    )


def test_save_and_reload_preserve_ids_and_artifact_contract(tmp_path: Path):
    tokenizer, base_path = _build_tokenizer(tmp_path)
    artifact = tmp_path / "artifact"

    files = tokenizer.save_pretrained(artifact)
    reloaded = BCNLTTokenizer.from_pretrained(artifact, base_path)

    assert {Path(path).name for path in files} == {
        "bcnlt_tokenizer_manifest.json",
        "bcnlt-tokenizer-v1.tokens.json",
    }
    assert not list(artifact.glob("*.safetensors"))
    assert not list(artifact.glob("*.bin"))
    assert reloaded.manifest == tokenizer.manifest
    assert reloaded.encode_text("hello world") == tokenizer.encode_text("hello world")
    assert reloaded.decode(reloaded.encode_text("hello world")) == "hello world"


def test_save_and_reload_freeze_multilingual_learned_state(tmp_path: Path):
    learned_path = tmp_path / "learned.jsonl"
    learned_path.write_text(
        '{"language":"javascript","semantic_spec_sha256":"fixture","signature":"fixture"}\n',
        encoding="utf-8",
    )
    base_path = write_byte_level_tokenizer(tmp_path / "base-state")
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    identity = identity_from_directory(
        base_path, name="fixture/base", revision="local-v1"
    )
    tokenizer = BCNLTTokenizer(
        base,
        build_manifest(base, identity),
        learned_memory_path=learned_path,
    )
    artifact = tmp_path / "artifact-state"

    files = tokenizer.save_pretrained(artifact)
    reloaded = BCNLTTokenizer.from_pretrained(artifact, base_path)

    assert {Path(path).name for path in files} >= {
        "bcnlt_learned_definitions.jsonl",
        "bcnlt_learned_state.json",
    }
    assert (
        artifact / "bcnlt_learned_definitions.jsonl"
    ).read_bytes() == learned_path.read_bytes()
    assert reloaded.learned_memory_path == artifact / "bcnlt_learned_definitions.jsonl"


def test_loaded_artifact_does_not_mutate_frozen_learned_state(tmp_path: Path):
    learned_path = tmp_path / "empty-learned.jsonl"
    learned_path.write_text("", encoding="utf-8")
    tokenizer, base_path = _build_tokenizer_with_learning(tmp_path, learned_path)
    artifact = tmp_path / "read-only-artifact"
    tokenizer.save_pretrained(artifact)
    frozen_path = artifact / "bcnlt_learned_definitions.jsonl"
    frozen_bytes = frozen_path.read_bytes()

    reloaded = BCNLTTokenizer.from_pretrained(artifact, base_path)
    reloaded.encode_source("ready = True\nassert ready\n", path="runtime.py")

    assert frozen_path.read_bytes() == frozen_bytes
    BCNLTTokenizer.from_pretrained(artifact, base_path)


def test_sqlite_learned_state_is_consumed_and_frozen_as_sqlite(tmp_path: Path):
    source = "ready = True\nassert ready\n"
    learned_path = tmp_path / "learned.sqlite3"
    seed_adapter = PythonAdapter()
    document = seed_adapter.parse(source, path="sqlite-seed.py")
    unit = next(
        item
        for item in seed_adapter.to_units(document)
        if item.event == "UNKNOWN_SEMANTIC" and item.ast_relation == "Assert"
    )
    from bcnlt.adapters.python import build_python_spec

    memory = WildcardMemory(
        learned_path,
        language="python",
        semantic_spec_sha256=build_python_spec().canonical_hash,
        storage="sqlite",
    )
    memory.remember(
        unit,
        ProvisionalMeaning(
            category="error",
            hypotheses=("assert may raise an assertion error",),
            evidence=("sqlite parity fixture",),
            confidence=0.64,
            target_event="RAISE",
            target_ir_kind=IRKind.ERROR.value,
        ),
        document_identity="sqlite-seed.py",
    )
    memory.close()

    base_path = write_byte_level_tokenizer(tmp_path / "base-sqlite-state")
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    identity = identity_from_directory(
        base_path, name="fixture/base", revision="local-v1"
    )
    tokenizer = BCNLTTokenizer(
        base,
        build_manifest(base, identity),
        learned_memory_path=learned_path,
        learned_memory_storage="sqlite",
    )

    assert len(tokenizer.extended_core_library()) == 0
    tokenizer.reload_learned_state()
    records = tokenizer.decode_records(tokenizer.encode_source(source, path="runtime.py"))
    assert any(record.event == "UNKNOWN_SEMANTIC" for record in records)

    artifact = tmp_path / "artifact-sqlite-state"
    files = tokenizer.save_pretrained(artifact)
    reloaded = BCNLTTokenizer.from_pretrained(artifact, base_path)

    assert {Path(path).name for path in files} >= {
        "bcnlt_learned_definitions.sqlite3",
        "bcnlt_learned_state.json",
    }
    assert reloaded.learned_memory_storage == "sqlite"
    assert reloaded.learned_memory_path == artifact / "bcnlt_learned_definitions.sqlite3"
    assert len(reloaded.extended_core_library()) == 0
    frozen_path = artifact / "bcnlt_learned_definitions.sqlite3"
    frozen_bytes = frozen_path.read_bytes()
    reloaded.encode_source(source, path="frozen-runtime.py")
    assert frozen_path.read_bytes() == frozen_bytes
    assert not (artifact / "bcnlt_learned_definitions.sqlite3-wal").exists()
    assert not (artifact / "bcnlt_learned_definitions.sqlite3-shm").exists()
    BCNLTTokenizer.from_pretrained(artifact, base_path)
    state = json.loads((artifact / "bcnlt_learned_state.json").read_text(encoding="utf-8"))
    assert state["storage"] == "sqlite"
    assert state["filename"] == "bcnlt_learned_definitions.sqlite3"

    with (artifact / "bcnlt_learned_definitions.sqlite3").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="learned semantic state hash mismatch"):
        BCNLTTokenizer.from_pretrained(artifact, base_path)


def test_reload_rejects_tampered_multilingual_learned_state(tmp_path: Path):
    learned_path = tmp_path / "learned.jsonl"
    learned_path.write_text("{}\n", encoding="utf-8")
    base_path = write_byte_level_tokenizer(tmp_path / "base-state")
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    identity = identity_from_directory(
        base_path, name="fixture/base", revision="local-v1"
    )
    tokenizer = BCNLTTokenizer(
        base,
        build_manifest(base, identity),
        learned_memory_path=learned_path,
    )
    artifact = tmp_path / "artifact-state"
    tokenizer.save_pretrained(artifact)
    with (artifact / "bcnlt_learned_definitions.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="learned semantic state hash mismatch"):
        BCNLTTokenizer.from_pretrained(artifact, base_path)


def test_fast_base_tokenizer_returns_exact_unicode_character_offsets(tmp_path: Path):
    _, base_path = _build_tokenizer(tmp_path)
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    text = "Aé🧙Z"

    ids, offsets = base.encode_with_offsets(text)

    assert len(ids) == len(offsets)
    assert all(text[start:end] for start, end in offsets if end > start)
    unique_spans = sorted({(start, end) for start, end in offsets if end > start})
    assert "".join(text[start:end] for start, end in unique_spans) == text


def test_generic_tokenizers_backend_artifact_loads_without_remote_code(tmp_path: Path):
    base_path = write_byte_level_tokenizer(tmp_path / "generic-fast")
    config_path = base_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tokenizer_class"] = "TokenizersBackend"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    tokenizer = HuggingFaceBaseTokenizer.from_pretrained(base_path)

    encoded = tokenizer.encode("hello λ")
    assert tokenizer.decode(encoded) == "hello λ"


def test_reload_rejects_corrupted_artifact_manifest(tmp_path: Path):
    tokenizer, base_path = _build_tokenizer(tmp_path)
    artifact = tmp_path / "artifact"
    tokenizer.save_pretrained(artifact)
    manifest_path = artifact / "bcnlt_tokenizer_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["base_vocab_size"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        BCNLTTokenizer.from_pretrained(artifact, base_path)


def test_reload_rejects_a_different_base_tokenizer(tmp_path: Path):
    tokenizer, _ = _build_tokenizer(tmp_path)
    artifact = tmp_path / "artifact"
    tokenizer.save_pretrained(artifact)
    other_base = write_byte_level_tokenizer(
        tmp_path / "other-base",
        corpus="completely different tokens and training material",
    )

    with pytest.raises(ValueError, match="base tokenizer|base vocabulary"):
        BCNLTTokenizer.from_pretrained(artifact, other_base)


def test_save_refuses_to_mix_with_existing_files(tmp_path: Path):
    tokenizer, _ = _build_tokenizer(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        tokenizer.save_pretrained(artifact)

    assert (artifact / "unrelated.txt").read_text(encoding="utf-8") == "keep"


def test_hub_tokenizer_source_requires_an_explicit_revision():
    with pytest.raises(ValueError, match="revision"):
        resolve_tokenizer_source("organization/model", revision=None)


def test_hub_tokenizer_source_resolves_revision_to_immutable_commit(
    tmp_path: Path, monkeypatch
):
    import huggingface_hub

    snapshot = write_byte_level_tokenizer(tmp_path / "snapshot")
    calls: dict[str, object] = {}

    class FakeApi:
        def model_info(self, repo_id: str, revision: str):
            calls["model_info"] = (repo_id, revision)
            return type("ModelInfo", (), {"sha": "a" * 40})()

    def fake_snapshot_download(**kwargs):
        calls["snapshot_download"] = kwargs
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: FakeApi())
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    resolved, commit = resolve_tokenizer_source(
        "organization/model", revision="release-v1"
    )

    assert resolved == snapshot
    assert commit == "a" * 40
    assert calls["model_info"] == ("organization/model", "release-v1")
    assert calls["snapshot_download"]["revision"] == "a" * 40
    assert calls["snapshot_download"]["repo_id"] == "organization/model"
