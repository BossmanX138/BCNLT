from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import tempfile
from contextlib import closing
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
    output = Path(output_path)
    manifest_path = output / "dataset_manifest.json"
    with tempfile.TemporaryDirectory(prefix="bcnlt-packing-") as state_root, closing(
        sqlite3.connect(Path(state_root) / "packing-state.sqlite3")
    ) as state:
        state.execute("PRAGMA journal_mode=WAL")
        state.execute("PRAGMA synchronous=NORMAL")
        state.execute("PRAGMA temp_store=FILE")
        state.execute(
            "CREATE TABLE records ("
            "row_id INTEGER PRIMARY KEY, split TEXT NOT NULL, "
            "record_id TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        state.execute("CREATE INDEX records_order ON records(split, record_id, row_id)")
        state.execute("BEGIN")
        lineage_rows: set[str] = set()
        input_count = 0
        for record in records:
            provenance = dict(record.provenance)
            if provenance.get("canonical_language") != record.language:
                state.close()
                raise ValueError("curriculum record language does not match provenance")
            payload = json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            state.execute(
                "INSERT INTO records(row_id, split, record_id, payload) VALUES (?, ?, ?, ?)",
                (input_count, record.split, record.record_id, payload),
            )
            lineage_rows.add(_lineage_row(record))
            input_count += 1
            if input_count % 10_000 == 0:
                state.commit()
                state.execute("BEGIN")
        state.commit()

        input_digest = _streaming_input_digest(state)
        lineage_digest = _canonical_hash(sorted(lineage_rows))
        if output.exists():
            if not resume:
                state.close()
                raise FileExistsError(f"output already exists: {output}")
            if not manifest_path.is_file():
                state.close()
                raise ValueError("resume output has no dataset manifest")
            existing = _load_manifest(manifest_path)
            if existing.lineage_manifest_sha256 != lineage_digest:
                state.close()
                raise ValueError("resume parser/spec/registry lineage does not match")
            if existing.input_record_sha256 != input_digest:
                state.close()
                raise ValueError("resume input range does not match the existing manifest")
            if existing.tokenizer_manifest_sha256 != manifest_digest:
                state.close()
                raise ValueError("resume tokenizer manifest digest does not match")
            if (
                existing.maximum_sequence_tokens != maximum_sequence_tokens
                or existing.shard_token_limit != shard_token_limit
            ):
                state.close()
                raise ValueError("resume packing policy does not match the existing manifest")
            state.close()
            _verify_manifest_files(output, existing)
            return existing

        output.mkdir(parents=True, exist_ok=False)
        shards: list[PackedShardManifest] = []
        accepted_count = 0
        rejected_count = 0
        current_split: str | None = None
        shard_index = 0
        batch: list[_EncodedRecord] = []
        batch_tokens = 0

        def flush_batch() -> None:
            nonlocal batch, batch_tokens, shard_index
            if not batch or current_split is None:
                return
            shards.append(_write_shard(output, current_split, shard_index, batch))
            shard_index += 1
            batch = []
            batch_tokens = 0

        with (output / "rejections.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as rejection_handle:
            rows = state.execute(
                "SELECT payload FROM records ORDER BY split, record_id, row_id"
            )
            for (payload,) in rows:
                record = _record_from_dict(json.loads(str(payload)))
                if current_split != record.split:
                    flush_batch()
                    current_split = record.split
                    shard_index = 0
                prefix, prompt_body = _split_language_prefix(
                    record.prompt, record.language
                )
                prompt_ids = list(tokenizer.encode_text(prefix)) + _encode_mode(
                    tokenizer, record.prompt_mode, prompt_body
                )
                target_ids = _encode_mode(tokenizer, record.target_mode, record.target)
                _validate_token_ids(prompt_ids, vocab_size)
                _validate_token_ids(target_ids, vocab_size)
                input_ids = tuple(prompt_ids + target_ids)
                if len(input_ids) > maximum_sequence_tokens:
                    rejection_handle.write(
                        json.dumps(
                            {
                                "record_id": record.record_id,
                                "reason_code": "sequence_too_long",
                                "encoded_tokens": len(input_ids),
                                "maximum_sequence_tokens": maximum_sequence_tokens,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    rejected_count += 1
                    continue
                encoded = _EncodedRecord(record, input_ids, len(prompt_ids))
                if batch and batch_tokens + len(input_ids) > shard_token_limit:
                    flush_batch()
                batch.append(encoded)
                batch_tokens += len(input_ids)
                accepted_count += 1
            flush_batch()
        state.close()

    body = {
        "schema_version": "bcnlt-packed-dataset-v2",
        "tokenizer_manifest_sha256": manifest_digest,
        "tokenizer_vocab_size": vocab_size,
        "maximum_sequence_tokens": maximum_sequence_tokens,
        "shard_token_limit": shard_token_limit,
        "input_record_sha256": input_digest,
        "lineage_manifest_sha256": lineage_digest,
        "accepted_record_count": accepted_count,
        "rejected_record_count": rejected_count,
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
        accepted_record_count=accepted_count,
        rejected_record_count=rejected_count,
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
        self._entries: list[
            tuple[PackedShardManifest, int, int, int, str, str, str, str]
        ] = []
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
                        str(metadata["language"]), str(metadata["family"]),
                    )
                )

    def __len__(self) -> int:
        return len(self._entries)

    def metadata(self, index: int) -> dict[str, str]:
        entry = self._entries[index]
        return {
            "record_id": entry[4],
            "target_mode": entry[5],
            "language": entry[6],
            "family": entry[7],
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        (
            shard, offset, length, prompt_length, record_id, target_mode,
            language, family,
        ) = self._entries[index]
        binary = self.root / shard.binary_path
        with binary.open("rb") as handle:
            handle.seek(offset * 4)
            payload = handle.read(length * 4)
        if len(payload) != length * 4:
            raise ValueError(f"truncated packed record: {shard.binary_path}")
        ids = list(struct.unpack(f"<{length}I", payload)) if length else []
        _validate_token_ids(ids, self.manifest.tokenizer_vocab_size)
        return {
            **self.metadata(index),
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


def _streaming_input_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    rows = connection.execute(
        "SELECT payload FROM records ORDER BY split, record_id, row_id"
    )
    for (payload,) in rows:
        if not first:
            digest.update(b",")
        digest.update(str(payload).encode("utf-8"))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _lineage_digest(records: Sequence[CurriculumRecord]) -> str:
    return _canonical_hash(sorted({_lineage_row(record) for record in records}))


def _lineage_row(record: CurriculumRecord) -> str:
    keys = (
        "registry_entry_sha256",
        "parser_bundle_sha256",
        "grammar_revision",
        "semantic_spec_sha256",
        "extraction_rule_version",
    )
    provenance = dict(record.provenance)
    return json.dumps(
        {
            "language": record.language,
            **{key: provenance.get(key, "") for key in keys},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _record_from_dict(raw: dict[str, object]) -> CurriculumRecord:
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("packed curriculum provenance must be an object")
    source_ids = raw.get("source_document_ids")
    if not isinstance(source_ids, list):
        raise ValueError("packed curriculum source_document_ids must be a list")
    return CurriculumRecord(
        record_id=str(raw["record_id"]),
        family=str(raw["family"]),
        template_id=str(raw["template_id"]),
        source_document_ids=tuple(str(item) for item in source_ids),
        prompt_mode=str(raw["prompt_mode"]),
        target_mode=str(raw["target_mode"]),
        prompt=str(raw["prompt"]),
        target=str(raw["target"]),
        provenance=tuple(
            sorted((str(key), str(value)) for key, value in provenance.items())
        ),
        split=str(raw["split"]),
        language=str(raw.get("language", provenance.get("canonical_language", "python"))),
        dialect=(str(raw["dialect"]) if raw.get("dialect") is not None else None),
    )


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
