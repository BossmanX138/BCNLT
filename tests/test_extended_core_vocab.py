from __future__ import annotations

from dataclasses import replace

import pytest

from bcnlt.extended_core import ExtendedCoreEntry, ExtendedCoreLibrary
from bcnlt.extended_core_tokenizer import ExtendedCoreBCNLTTokenizer
from bcnlt.extended_core_vocab import (
    build_extended_core_vocab,
    load_extended_core_vocab,
    validate_extended_core_vocab,
    write_extended_core_vocab,
)
from bcnlt.tokenizer_manifest import BaseTokenizerIdentity, build_manifest
from bcnlt.ir import IRKind
from bcnlt.semantic_record import CanonicalIR, ScoreBand, SemanticTokenRecord
from bcnlt.tokenizer import BCNLTTokenizer
from bcnlt.hf_tokenizer import HuggingFaceBaseTokenizer, identity_from_directory
from tests.tokenizer_fixtures import write_byte_level_tokenizer


class _ByteTokenizer:
    vocab_size = 256

    def get_vocab(self):
        return {f"<byte:{index:02x}>": index for index in range(256)}

    def encode(self, text: str, add_special_tokens: bool = False):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


def _v1_manifest():
    base = _ByteTokenizer()
    identity = BaseTokenizerIdentity.from_files(
        name="fixture/byte",
        revision="a" * 40,
        files={"tokenizer.json": b"fixture"},
    )
    return build_manifest(base, identity)


def _entry(name: str, *, revision: int = 1) -> ExtendedCoreEntry:
    return ExtendedCoreEntry(
        extended_core_id=f"bcnlt.extended.python.{name}.r{revision}",
        signature=f"python|spec|UNKNOWN_SEMANTIC|{name}|surface",
        revision=revision,
        language="python",
        semantic_spec_sha256="b" * 64,
        surface=f"{name} surface",
        relation=f"{name}.self",
        category="language_extension",
        hypotheses=(f"{name} has independently tested behavior",),
        evidence=(f"verified {name}",),
        confidence=0.93,
        validated_domain=f"python.{name}",
        common_core_projection=None,
        evidence_sha256=("c" * 64,),
        extension_name=f"python.{name}",
    )


def _record(entry: ExtendedCoreEntry) -> SemanticTokenRecord:
    return SemanticTokenRecord(
        event="UNKNOWN_SEMANTIC",
        surface=entry.surface,
        scope="module",
        span=(1, 0, 1, len(entry.surface)),
        symbol_id=None,
        defines=None,
        ast_relation=entry.relation,
        ir=CanonicalIR(
            kind=IRKind.EXTENSION.value,
            name=entry.extension_name,
            value=None,
            children=(),
            metadata=(("extended_core_id", entry.extended_core_id),),
        ),
        recent_history=(),
        metadata=tuple(
            sorted(
                (
                    ("extended_core_id", entry.extended_core_id),
                    ("extended_core_revision", entry.revision),
                    ("extended_core_signature", entry.signature),
                    ("extended_core_state", "validated"),
                    ("extended_core_name", entry.extension_name),
                )
            )
        ),
        confidence_band=ScoreBand.HIGH,
        uncertainty_band=ScoreBand.LOW,
    )


def test_extended_core_ids_append_and_revocation_never_reuses_or_renumbers_rows():
    v1 = _v1_manifest()
    first = build_extended_core_vocab(
        v1,
        ExtendedCoreLibrary((_entry("zeta"), _entry("alpha"))),
    )

    assert [token.extended_core_id for token in first.tokens] == [
        "bcnlt.extended.python.alpha.r1",
        "bcnlt.extended.python.zeta.r1",
    ]
    assert [token.id for token in first.tokens] == [v1.vocab_size, v1.vocab_size + 1]
    assert all(token.active for token in first.tokens)

    second = build_extended_core_vocab(
        v1,
        ExtendedCoreLibrary((_entry("zeta"), _entry("beta"))),
        previous=first,
    )

    by_meaning = {token.extended_core_id: token for token in second.tokens}
    assert by_meaning["bcnlt.extended.python.alpha.r1"].id == v1.vocab_size
    assert by_meaning["bcnlt.extended.python.alpha.r1"].active is False
    assert by_meaning["bcnlt.extended.python.zeta.r1"].id == v1.vocab_size + 1
    assert by_meaning["bcnlt.extended.python.zeta.r1"].active is True
    assert by_meaning["bcnlt.extended.python.beta.r1"].id == v1.vocab_size + 2
    assert by_meaning["bcnlt.extended.python.beta.r1"].active is True
    assert second.vocab_size == v1.vocab_size + 3
    assert second.parent_manifest_sha256 == first.manifest_sha256


def test_extended_core_manifest_round_trips_canonical_bytes(tmp_path):
    v1 = _v1_manifest()
    manifest = build_extended_core_vocab(
        v1,
        ExtendedCoreLibrary((_entry("alpha"),)),
    )
    path = tmp_path / "bcnlt_extended_core_manifest.json"

    write_extended_core_vocab(path, manifest, v1)
    reloaded = load_extended_core_vocab(path, v1)

    assert reloaded == manifest
    assert path.read_bytes().endswith(b"\n")


def test_native_extended_core_token_is_required_and_round_trips_the_record():
    base = _ByteTokenizer()
    v1_manifest = _v1_manifest()
    v1 = BCNLTTokenizer(base, v1_manifest)
    entry = _entry("alpha")
    library = ExtendedCoreLibrary((entry,))
    manifest = build_extended_core_vocab(v1_manifest, library)
    tokenizer = ExtendedCoreBCNLTTokenizer(v1, manifest, library=library)
    record = _record(entry)

    encoded = tokenizer.encode_records((record,))

    assert manifest.tokens[0].id in encoded
    assert manifest.tokens[0].id not in v1.encode_records((record,))
    assert tokenizer.decode_records(encoded) == (record,)

    without_native_identity = list(encoded)
    without_native_identity.remove(manifest.tokens[0].id)
    with pytest.raises(ValueError, match="missing native Extended Core token"):
        tokenizer.decode_records(without_native_identity)


def test_native_extended_core_artifact_reload_preserves_ids(tmp_path):
    base_path = write_byte_level_tokenizer(tmp_path / "base")
    base = HuggingFaceBaseTokenizer.from_pretrained(base_path)
    identity = identity_from_directory(
        base_path,
        name="fixture/base",
        revision="local-v1",
    )
    v1 = BCNLTTokenizer(base, build_manifest(base, identity))
    entry = _entry("alpha")
    library = ExtendedCoreLibrary((entry,))
    manifest = build_extended_core_vocab(v1.manifest, library)
    tokenizer = ExtendedCoreBCNLTTokenizer(v1, manifest, library=library)
    artifact = tmp_path / "native-artifact"

    tokenizer.save_pretrained(artifact)
    reloaded = ExtendedCoreBCNLTTokenizer.from_pretrained(
        artifact,
        base_path,
        library=library,
    )

    assert reloaded.manifest == manifest
    assert reloaded.encode_records((_record(entry),)) == tokenizer.encode_records(
        (_record(entry),)
    )


def test_manifest_rejects_non_boolean_activity_before_digest_validation():
    v1 = _v1_manifest()
    manifest = build_extended_core_vocab(
        v1,
        ExtendedCoreLibrary((_entry("alpha"),)),
    )
    corrupted = replace(
        manifest,
        tokens=(replace(manifest.tokens[0], active="yes"),),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="active.*boolean"):
        validate_extended_core_vocab(corrupted, v1)


def test_writer_refuses_tampered_manifest(tmp_path):
    v1 = _v1_manifest()
    manifest = build_extended_core_vocab(
        v1,
        ExtendedCoreLibrary((_entry("alpha"),)),
    )
    corrupted = replace(
        manifest,
        tokens=(replace(manifest.tokens[0], active="yes"),),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="active.*boolean"):
        write_extended_core_vocab(tmp_path / "bad.json", corrupted, v1)


def test_revoked_token_is_decode_only_and_cannot_be_emitted_again():
    base = _ByteTokenizer()
    v1_manifest = _v1_manifest()
    v1 = BCNLTTokenizer(base, v1_manifest)
    alpha = _entry("alpha")
    first_library = ExtendedCoreLibrary((alpha,))
    first_manifest = build_extended_core_vocab(v1_manifest, first_library)
    first = ExtendedCoreBCNLTTokenizer(
        v1,
        first_manifest,
        library=first_library,
    )
    historical = first.encode_records((_record(alpha),))

    beta = _entry("beta")
    current_library = ExtendedCoreLibrary((beta,))
    current_manifest = build_extended_core_vocab(
        v1_manifest,
        current_library,
        previous=first_manifest,
    )
    current = ExtendedCoreBCNLTTokenizer(
        v1,
        current_manifest,
        library=current_library,
    )

    assert current.decode_records(historical) == (_record(alpha),)
    with pytest.raises(ValueError, match="no active native token"):
        current.encode_records((_record(alpha),))


def test_tokenizer_rejects_library_identity_drift_even_when_id_is_unchanged():
    base = _ByteTokenizer()
    v1_manifest = _v1_manifest()
    v1 = BCNLTTokenizer(base, v1_manifest)
    entry = _entry("alpha")
    manifest = build_extended_core_vocab(
        v1_manifest,
        ExtendedCoreLibrary((entry,)),
    )
    drifted = replace(entry, signature="python|tampered|signature")

    with pytest.raises(ValueError, match="identity.*does not match"):
        ExtendedCoreBCNLTTokenizer(
            v1,
            manifest,
            library=ExtendedCoreLibrary((drifted,)),
        )


def test_manifest_rejects_invalid_scientific_evidence_digest():
    v1 = _v1_manifest()
    manifest = build_extended_core_vocab(
        v1,
        ExtendedCoreLibrary((_entry("alpha"),)),
    )
    corrupted = replace(
        manifest,
        tokens=(
            replace(manifest.tokens[0], evidence_sha256=("not-a-digest",)),
        ),
    )

    with pytest.raises(ValueError, match="evidence.*digest"):
        validate_extended_core_vocab(corrupted, v1)


def test_manifest_rejects_invalid_parent_lineage_digest():
    v1 = _v1_manifest()
    manifest = build_extended_core_vocab(
        v1,
        ExtendedCoreLibrary((_entry("alpha"),)),
    )

    with pytest.raises(ValueError, match="parent.*digest"):
        validate_extended_core_vocab(
            replace(manifest, parent_manifest_sha256="not-a-digest"),
            v1,
        )
