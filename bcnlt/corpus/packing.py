from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from bcnlt.canonical import record_from_canonical_bytes

from .curriculum import CurriculumRecord


@dataclass(frozen=True)
class PackedShardManifest:
    split: str
    shard_index: int
    binary_path: str
    index_path: str
    records_path: str
    binary_sha256: str
    index_sha256: str
    records_sha256: str
    first_record_id: str
    last_record_id: str
    record_count: int
    token_count: int
    start_offset: int = 0

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PackedDatasetManifest:
    schema_version: str
    tokenizer_manifest_sha256: str
    tokenizer_vocab_size: int
    maximum_sequence_tokens: int
    shard_token_limit: int
    input_record_sha256: str
    lineage_manifest_sha256: str
    accepted_record_count: int
    rejected_record_count: int
    shards: tuple[PackedShardManifest, ...]
    dataset_manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "maximum_sequence_tokens": self.maximum_sequence_tokens,
            "shard_token_limit": self.shard_token_limit,
            "input_record_sha256": self.input_record_sha256,
            "lineage_manifest_sha256": self.lineage_manifest_sha256,
            "accepted_record_count": self.accepted_record_count,
            "rejected_record_count": self.rejected_record_count,
            "shards": [shard.to_dict() for shard in self.shards],
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
        }


@dataclass(frozen=True)
class _EncodedRecord:
    record: CurriculumRecord
    input_ids: tuple[int, ...]
    prompt_length: int


def pack_training_shards(
    records: Iterable[CurriculumRecord],
    tokenizer: Any,
    output_path: str | Path,
    *,
    maximum_sequence_tokens: int = 8192,
    shard_token_limit: int = 8_000_000,
    resume: bool = False,
) -> PackedDatasetManifest:
    if maximum_sequence_tokens <= 0 or shard_token_limit <= 0:
        raise ValueError("packing token limits must be positive")
    manifest_digest, vocab_size = _tokenizer_identity(tokenizer)
    ordered = tuple(sorted(records, key=lambda item: (item.split, item.record_id)))
    input_digest = _input_digest(ordered)
    lineage_digest = _lineage_digest(ordered)
    output = Path(output_path)
    manifest_path = output / "dataset_manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"output already exists: {output}")
        if not manifest_path.is_file():
            raise ValueError("resume output has no dataset manifest")
        existing = _load_manifest(manifest_path)
        if existing.lineage_manifest_sha256 != lineage_digest:
            raise ValueError("resume parser/spec/registry lineage does not match")
        if existing.input_record_sha256 != input_digest:
            raise ValueError("resume input range does not match the existing manifest")
        if existing.tokenizer_manifest_sha256 != manifest_digest:
            raise ValueError("resume tokenizer manifest digest does not match")
        if (
            existing.maximum_sequence_tokens != maximum_sequence_tokens
            or existing.shard_token_limit != shard_token_limit
        ):
            raise ValueError("resume packing policy does not match the existing manifest")
        _verify_manifest_files(output, existing)
        return existing

    encoded: list[_EncodedRecord] = []
    rejections: list[dict[str, object]] = []
    for record in ordered:
        provenance_language = dict(record.provenance).get("canonical_language")
        if provenance_language != record.language:
            raise ValueError("curriculum record language does not match provenance")
        prefix, prompt_body = _split_language_prefix(record.prompt, record.language)
        prompt_ids = list(tokenizer.encode_text(prefix)) + _encode_mode(
            tokenizer, record.prompt_mode, prompt_body
        )
        target_ids = _encode_mode(tokenizer, record.target_mode, record.target)
        _validate_token_ids(prompt_ids, vocab_size)
        _validate_token_ids(target_ids, vocab_size)
        input_ids = tuple(prompt_ids + target_ids)
        if len(input_ids) > maximum_sequence_tokens:
            rejections.append(
                {
                    "record_id": record.record_id,
                    "reason_code": "sequence_too_long",
                    "encoded_tokens": len(input_ids),
                    "maximum_sequence_tokens": maximum_sequence_tokens,
                }
            )
            continue
        encoded.append(_EncodedRecord(record, input_ids, len(prompt_ids)))

    output.mkdir(parents=True, exist_ok=False)
    rejection_bytes = b"".join(
        (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for item in rejections
    )
    (output / "rejections.jsonl").write_bytes(rejection_bytes)
    shards: list[PackedShardManifest] = []
    by_split: dict[str, list[_EncodedRecord]] = {}
    for item in encoded:
        by_split.setdefault(item.record.split, []).append(item)
    for split in sorted(by_split):
        batches = _shard_batches(by_split[split], shard_token_limit)
        for shard_index, batch in enumerate(batches):
            shards.append(_write_shard(output, split, shard_index, batch))

    body = {
        "schema_version": "bcnlt-packed-dataset-v2",
        "tokenizer_manifest_sha256": manifest_digest,
        "tokenizer_vocab_size": vocab_size,
        "maximum_sequence_tokens": maximum_sequence_tokens,
        "shard_token_limit": shard_token_limit,
        "input_record_sha256": input_digest,
        "lineage_manifest_sha256": lineage_digest,
        "accepted_record_count": len(encoded),
        "rejected_record_count": len(rejections),
        "shards": [shard.to_dict() for shard in shards],
    }
    dataset_digest = _canonical_hash(body)
    manifest = PackedDatasetManifest(
        schema_version="bcnlt-packed-dataset-v2",
        tokenizer_manifest_sha256=manifest_digest,
        tokenizer_vocab_size=vocab_size,
        maximum_sequence_tokens=maximum_sequence_tokens,
        shard_token_limit=shard_token_limit,
        input_record_sha256=input_digest,
        lineage_manifest_sha256=lineage_digest,
        accepted_record_count=len(encoded),
        rejected_record_count=len(rejections),
        shards=tuple(shards),
        dataset_manifest_sha256=dataset_digest,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


class PackedBCNLTDataset:
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        expected_tokenizer_digest: str,
    ) -> None:
        self.root = Path(root)
        self.manifest = _load_manifest(self.root / "dataset_manifest.json")
        if self.manifest.tokenizer_manifest_sha256 != expected_tokenizer_digest:
            raise ValueError("tokenizer manifest digest does not match packed dataset")
        _verify_manifest_files(self.root, self.manifest)
        self._entries: list[tuple[PackedShardManifest, int, int, int, str, str, str]] = []
        for shard in self.manifest.shards:
            if shard.split != split:
                continue
            index_bytes = (self.root / shard.index_path).read_bytes()
            provenance_rows = [
                json.loads(line)
                for line in (self.root / shard.records_path).read_text(encoding="utf-8").splitlines()
            ]
            if len(index_bytes) != shard.record_count * 24:
                raise ValueError(f"index size mismatch: {shard.index_path}")
            if len(provenance_rows) != shard.record_count:
                raise ValueError(f"record provenance count mismatch: {shard.records_path}")
            for row_index, offset in enumerate(range(0, len(index_bytes), 24)):
                token_offset, length, prompt_length = struct.unpack_from(
                    "<QQQ", index_bytes, offset
                )
                if prompt_length > length:
                    raise ValueError(f"invalid prompt length: {shard.index_path}")
                metadata = provenance_rows[row_index]
                self._entries.append(
                    (
                        shard, token_offset, length, prompt_length,
                        str(metadata["record_id"]), str(metadata["target_mode"]),
                        str(metadata["language"]),
                    )
                )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        shard, offset, length, prompt_length, record_id, target_mode, language = self._entries[index]
        binary = self.root / shard.binary_path
        with binary.open("rb") as handle:
            handle.seek(offset * 4)
            payload = handle.read(length * 4)
        if len(payload) != length * 4:
            raise ValueError(f"truncated packed record: {shard.binary_path}")
        ids = list(struct.unpack(f"<{length}I", payload)) if length else []
        _validate_token_ids(ids, self.manifest.tokenizer_vocab_size)
        return {
            "record_id": record_id,
            "target_mode": target_mode,
            "language": language,
            "input_ids": ids,
            "labels": [-100] * prompt_length + ids[prompt_length:],
            "attention_mask": [1] * len(ids),
        }


def _tokenizer_identity(tokenizer: Any) -> tuple[str, int]:
    manifest = getattr(tokenizer, "manifest", None)
    digest = getattr(manifest, "manifest_sha256", None)
    vocab_size = getattr(manifest, "vocab_size", None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("tokenizer requires a 64-character manifest digest")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise ValueError("tokenizer manifest requires a positive vocabulary size")
    return digest, vocab_size


def _encode_mode(tokenizer: Any, mode: str, value: str) -> list[int]:
    if mode == "raw_code":
        return list(tokenizer.encode_raw_code(value))
    if mode == "semantic_code":
        records = _records_from_embedded_stream(value)
        if records:
            text = "".join(
                line
                for line in value.splitlines(keepends=True)
                if not line.lstrip().startswith("{")
            )
            return list(tokenizer.encode_text(text)) + list(tokenizer.encode_records(records))
    return list(tokenizer.encode_text(value))


def _records_from_embedded_stream(value: str) -> tuple[Any, ...]:
    records = []
    for line in value.splitlines():
        if not line.startswith("{"):
            continue
        try:
            records.append(record_from_canonical_bytes((line + "\n").encode()))
        except ValueError:
            continue
    return tuple(records)


def _validate_token_ids(ids: Sequence[int], vocab_size: int) -> None:
    for token_id in ids:
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id >= vocab_size
        ):
            raise ValueError(f"token ID outside tokenizer vocabulary: {token_id!r}")


def _shard_batches(
    records: Sequence[_EncodedRecord], token_limit: int
) -> list[list[_EncodedRecord]]:
    batches: list[list[_EncodedRecord]] = []
    current: list[_EncodedRecord] = []
    current_tokens = 0
    for record in records:
        if current and current_tokens + len(record.input_ids) > token_limit:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(record)
        current_tokens += len(record.input_ids)
    if current:
        batches.append(current)
    return batches


def _write_shard(
    output: Path,
    split: str,
    shard_index: int,
    records: Sequence[_EncodedRecord],
) -> PackedShardManifest:
    stem = f"{split}-{shard_index:05d}"
    binary_path = f"{stem}.bin"
    index_path = f"{stem}.idx"
    records_path = f"{stem}.records.jsonl"
    binary = bytearray()
    index = bytearray()
    provenance = bytearray()
    offset = 0
    for item in records:
        if item.input_ids:
            binary.extend(struct.pack(f"<{len(item.input_ids)}I", *item.input_ids))
        index.extend(struct.pack("<QQQ", offset, len(item.input_ids), item.prompt_length))
        metadata = {
            "record_id": item.record.record_id,
            "source_document_ids": list(item.record.source_document_ids),
            "family": item.record.family,
            "template_id": item.record.template_id,
            "target_mode": item.record.target_mode,
            "split": item.record.split,
            "language": item.record.language,
            "dialect": item.record.dialect,
            "offset": offset,
            "length": len(item.input_ids),
            "prompt_length": item.prompt_length,
            "provenance": dict(item.record.provenance),
        }
        provenance.extend(
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        offset += len(item.input_ids)
    (output / binary_path).write_bytes(binary)
    (output / index_path).write_bytes(index)
    (output / records_path).write_bytes(provenance)
    return PackedShardManifest(
        split=split,
        shard_index=shard_index,
        binary_path=binary_path,
        index_path=index_path,
        records_path=records_path,
        binary_sha256=hashlib.sha256(binary).hexdigest(),
        index_sha256=hashlib.sha256(index).hexdigest(),
        records_sha256=hashlib.sha256(provenance).hexdigest(),
        first_record_id=records[0].record.record_id,
        last_record_id=records[-1].record.record_id,
        record_count=len(records),
        token_count=offset,
    )


def _input_digest(records: Sequence[CurriculumRecord]) -> str:
    payload = [record.to_dict() for record in records]
    return _canonical_hash(payload)


def _lineage_digest(records: Sequence[CurriculumRecord]) -> str:
    keys = (
        "registry_entry_sha256",
        "parser_bundle_sha256",
        "grammar_revision",
        "semantic_spec_sha256",
        "extraction_rule_version",
    )
    payload = set()
    for record in records:
        provenance = dict(record.provenance)
        payload.add(
            json.dumps(
                {
                    "language": record.language,
                    **{key: provenance.get(key, "") for key in keys},
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return _canonical_hash(sorted(payload))


def _split_language_prefix(value: str, language: str) -> tuple[str, str]:
    prefix = f"Language: {language}\n"
    if not value.startswith(prefix):
        raise ValueError("curriculum prompt must begin with its canonical language field")
    return prefix, value[len(prefix) :]


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest_files(root: Path, manifest: PackedDatasetManifest) -> None:
    for shard in manifest.shards:
        for relative, expected in (
            (shard.binary_path, shard.binary_sha256),
            (shard.index_path, shard.index_sha256),
            (shard.records_path, shard.records_sha256),
        ):
            path = root / relative
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"packed shard hash mismatch: {relative}")


def _load_manifest(path: Path) -> PackedDatasetManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    supplied = raw.get("dataset_manifest_sha256")
    body = {key: value for key, value in raw.items() if key != "dataset_manifest_sha256"}
    if supplied != _canonical_hash(body):
        raise ValueError("dataset manifest hash mismatch")
    shards = tuple(PackedShardManifest(**item) for item in body.pop("shards"))
    return PackedDatasetManifest(
        **body, shards=shards, dataset_manifest_sha256=supplied
    )
