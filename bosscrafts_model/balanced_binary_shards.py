"""Compact uint16 task shards with deterministic balanced sampling weights."""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TypedDict, cast

from .bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from .current_task_packing import (
    EXAMPLE_SCHEMA,
    FAMILIES,
    PACKED_MANIFEST_SCHEMA,
)
from .tokenizer import NativeTokenizerBinding

BINARY_MANIFEST_SCHEMA = "bosscrafts-balanced-binary-task-manifest-v1"
INDEX_STRUCT = struct.Struct("<QHHfBBBB32s32s32s32s")


class _PackedExample(TypedDict):
    example_id: str
    document_id: str
    language: str
    family: str
    split: str
    input_ids: list[int]
    loss_start_index: int
    semantic_spec_sha256: str
    source_sha256: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_u16_tokens(token_ids: list[int]) -> bytes:
    if any(type(token_id) is not int or not 0 <= token_id <= 65_535 for token_id in token_ids):
        raise ValueError("token IDs must fit uint16")
    if not token_ids:
        return b""
    return struct.pack(f"<{len(token_ids)}H", *token_ids)


def decode_u16_tokens(content: bytes) -> list[int]:
    if len(content) % 2:
        raise ValueError("uint16 token bytes have an odd length")
    if not content:
        return []
    return list(struct.unpack(f"<{len(content) // 2}H", content))


@dataclass(frozen=True)
class BalancedBinaryShardResult:
    output_root: Path
    example_count: int
    token_count: int
    shard_count: int
    shard_hashes: tuple[tuple[str, str], ...]
    manifest_sha256: str


def _load_manifest(root: Path, binding: NativeTokenizerBinding) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    examples_path = root / "examples.jsonl"
    rejections_path = root / "rejected.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("packed manifest is invalid") from error
    if type(manifest) is not dict:
        raise ValueError("packed manifest is invalid")
    if manifest.get("schema_version") != PACKED_MANIFEST_SCHEMA:
        raise ValueError("packed manifest schema mismatch")
    claimed_hash = manifest.get("manifest_sha256")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if claimed_hash != _sha256_text(_canonical_json(unsigned_manifest)):
        raise ValueError("packed manifest SHA-256 mismatch")
    if (
        manifest.get("examples_file") != "examples.jsonl"
        or manifest.get("rejections_file") != "rejected.jsonl"
    ):
        raise ValueError("packed manifest file inventory mismatch")
    if _sha256_file(examples_path) != manifest.get("examples_sha256"):
        raise ValueError("packed examples SHA-256 mismatch")
    if _sha256_file(rejections_path) != manifest.get("rejections_sha256"):
        raise ValueError("packed rejections SHA-256 mismatch")
    if (
        manifest.get("binding_sha256") != binding.binding_sha256
        or manifest.get("tokenizer_sha256") != binding.tokenizer_sha256
        or manifest.get("vocabulary_size") != binding.vocab_size
        or manifest.get("parser_runtime_identity_sha256")
        != PARSER_RUNTIME_IDENTITY_SHA256
    ):
        raise ValueError("packed tokenizer lineage mismatch")
    return dict(manifest)


def _parse_example(line: str, binding: NativeTokenizerBinding) -> _PackedExample:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError("packed example JSON is invalid") from error
    if type(row) is not dict or row.get("schema_version") != EXAMPLE_SCHEMA:
        raise ValueError("packed example schema mismatch")
    if (
        row.get("binding_sha256") != binding.binding_sha256
        or row.get("tokenizer_sha256") != binding.tokenizer_sha256
        or row.get("parser_runtime_identity_sha256")
        != PARSER_RUNTIME_IDENTITY_SHA256
    ):
        raise ValueError("packed example tokenizer lineage mismatch")
    input_ids = row.get("input_ids")
    if type(input_ids) is not list or any(
        type(token_id) is not int or not 0 <= token_id < binding.vocab_size
        for token_id in input_ids
    ):
        raise ValueError("packed input_ids exceed the frozen vocabulary")
    loss_start = row.get("loss_start_index")
    if type(loss_start) is not int or not 0 < loss_start < len(input_ids):
        raise ValueError("packed loss_start_index is invalid")
    for field in ("example_id", "semantic_spec_sha256", "source_sha256"):
        value = row.get(field)
        if type(value) is not str or len(value) != 64:
            raise ValueError(f"packed {field} is not a SHA-256 hex digest")
        try:
            bytes.fromhex(value)
        except ValueError as error:
            raise ValueError(f"packed {field} is not a SHA-256 hex digest") from error
    if type(row.get("document_id")) is not str or not row["document_id"]:
        raise ValueError("packed document_id is invalid")
    if type(row.get("record_index")) is not int or row["record_index"] < 0:
        raise ValueError("packed record_index is invalid")
    if row.get("split") not in {"train", "validation", "test"}:
        raise ValueError("packed split is invalid")
    if row.get("family") not in FAMILIES:
        raise ValueError("packed family is invalid")
    language = row.get("language")
    if (
        type(language) is not str
        or language not in CURRENT_RUNTIME_SPEC_SHA256
        or row.get("semantic_spec_sha256")
        != CURRENT_RUNTIME_SPEC_SHA256[language]
    ):
        raise ValueError("packed language is invalid")
    expected_example_id = _sha256_text(
        _canonical_json(
            {
                "document_id": row["document_id"],
                "family": row["family"],
                "record_index": row.get("record_index"),
                "input_ids_sha256": _sha256_text(_canonical_json(input_ids)),
            }
        )
    )
    if row.get("example_id") != expected_example_id:
        raise ValueError("packed example identity is invalid")
    if (
        input_ids[0]
        != binding.structural_token_ids[f"<|bcnlt:task:{row['family']}|>"]
        or input_ids[-1]
        != binding.structural_token_ids["<|bcnlt:task:end|>"]
        or row.get("target_token_count") != len(input_ids) - loss_start
    ):
        raise ValueError("packed task framing is invalid")
    return cast(_PackedExample, row)


class BalancedBinaryShardWriter:
    def __init__(
        self,
        binding: NativeTokenizerBinding,
        *,
        max_tokens_per_shard: int = 8_000_000,
    ) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        if binding.vocab_size > 65_536:
            raise ValueError("frozen vocabulary exceeds uint16 capacity")
        if type(max_tokens_per_shard) is not int or max_tokens_per_shard < 1:
            raise ValueError("max_tokens_per_shard must be a positive integer")
        self._binding = binding
        self.max_tokens_per_shard = max_tokens_per_shard

    def materialize(
        self, tasks_root: str | Path, output_root: str | Path
    ) -> BalancedBinaryShardResult:
        tasks_path = Path(tasks_root).resolve()
        output_path = Path(output_root).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        packed_manifest = _load_manifest(tasks_path, self._binding)
        if any(output_path.iterdir()):
            raise ValueError("balanced binary output already contains files")
        examples_path = tasks_path / "examples.jsonl"
        expected_examples_sha256 = str(packed_manifest["examples_sha256"])

        cell_counts: Counter[tuple[str, str, str]] = Counter()
        languages: set[str] = set()
        document_ids: set[str] = set()
        example_count = 0
        total_tokens = 0
        with examples_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = _parse_example(line, self._binding)
                cell = (str(row["split"]), str(row["language"]), str(row["family"]))
                cell_counts[cell] += 1
                languages.add(cell[1])
                document_ids.add(str(row["document_id"]))
                example_count += 1
                total_tokens += len(row["input_ids"])
        if example_count != packed_manifest.get("example_count"):
            raise ValueError("packed example count mismatch")
        if total_tokens != packed_manifest.get("total_sequence_tokens"):
            raise ValueError("packed token count mismatch")
        if len(languages) > 255:
            raise ValueError("language count exceeds binary index capacity")
        if _sha256_file(examples_path) != expected_examples_sha256:
            raise ValueError("packed examples changed during binary materialization")

        cells_by_split: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for cell in sorted(cell_counts):
            cells_by_split[cell[0]].append(cell)
        weights: dict[tuple[str, str, str], float] = {}
        weighted_mass: dict[str, float] = {}
        for _split, cells in sorted(cells_by_split.items()):
            mean_count = sum(cell_counts[cell] for cell in cells) / len(cells)
            for cell in cells:
                weight32 = struct.unpack("<f", struct.pack("<f", mean_count / cell_counts[cell]))[0]
                weights[cell] = weight32
                weighted_mass["|".join(cell)] = weight32 * cell_counts[cell]

        language_names = sorted(languages)
        family_names = ["code_to_semantic", "semantic_continuation", "semantic_to_code"]
        split_names = ["train", "validation", "test"]
        language_codes = {value: index for index, value in enumerate(language_names)}
        family_codes = {value: index for index, value in enumerate(family_names)}
        split_codes = {value: index for index, value in enumerate(split_names)}

        shards: list[dict[str, object]] = []
        shard_hashes: list[tuple[str, str]] = []
        token_handle = None
        index_handle = None
        shard_number = -1
        shard_tokens = 0
        shard_examples = 0
        token_path: Path | None = None
        index_path: Path | None = None
        staging_path = Path(
            tempfile.mkdtemp(dir=output_path, prefix=".balanced-binary-stage-")
        )

        def finish_shard() -> None:
            nonlocal token_handle, index_handle, shard_tokens, shard_examples
            if token_handle is None or index_handle is None or token_path is None or index_path is None:
                return
            token_handle.flush()
            index_handle.flush()
            os.fsync(token_handle.fileno())
            os.fsync(index_handle.fileno())
            token_handle.close()
            index_handle.close()
            token_sha = _sha256_file(token_path)
            index_sha = _sha256_file(index_path)
            shards.append(
                {
                    "shard": shard_number,
                    "token_file": token_path.name,
                    "token_sha256": token_sha,
                    "index_file": index_path.name,
                    "index_sha256": index_sha,
                    "token_count": shard_tokens,
                    "example_count": shard_examples,
                }
            )
            shard_hashes.append((token_sha, index_sha))
            token_handle = None
            index_handle = None

        def start_shard() -> None:
            nonlocal token_handle, index_handle, shard_number, shard_tokens, shard_examples, token_path, index_path
            shard_number += 1
            shard_tokens = 0
            shard_examples = 0
            token_path = staging_path / f"tasks-{shard_number:05d}.u16"
            index_path = staging_path / f"tasks-{shard_number:05d}.idx"
            token_handle = token_path.open("wb")
            index_handle = index_path.open("wb")

        try:
            with examples_path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = _parse_example(line, self._binding)
                    input_ids = row["input_ids"]
                    sequence_length = len(input_ids)
                    if sequence_length > self.max_tokens_per_shard:
                        raise ValueError("packed example exceeds max_tokens_per_shard")
                    if sequence_length > 65_535:
                        raise ValueError("packed example exceeds binary index capacity")
                    if token_handle is None:
                        start_shard()
                    elif shard_tokens + sequence_length > self.max_tokens_per_shard:
                        finish_shard()
                        start_shard()
                    if token_handle is None or index_handle is None:
                        raise RuntimeError("shard output handles were not initialized")
                    cell = (
                        str(row["split"]),
                        str(row["language"]),
                        str(row["family"]),
                    )
                    token_handle.write(encode_u16_tokens(input_ids))
                    index_handle.write(
                        INDEX_STRUCT.pack(
                            shard_tokens,
                            sequence_length,
                            int(row["loss_start_index"]),
                            weights[cell],
                            language_codes[cell[1]],
                            family_codes[cell[2]],
                            split_codes[cell[0]],
                            0,
                            bytes.fromhex(str(row["example_id"])),
                            hashlib.sha256(
                                str(row["document_id"]).encode("utf-8")
                            ).digest(),
                            bytes.fromhex(str(row["semantic_spec_sha256"])),
                            bytes.fromhex(str(row["source_sha256"])),
                        )
                    )
                    shard_tokens += sequence_length
                    shard_examples += 1
            finish_shard()
            if _sha256_file(examples_path) != expected_examples_sha256:
                raise ValueError(
                    "packed examples changed during binary materialization"
                )
        except BaseException:
            if token_handle is not None:
                token_handle.close()
            if index_handle is not None:
                index_handle.close()
            for path in staging_path.iterdir():
                if path.is_file():
                    path.unlink()
            staging_path.rmdir()
            raise

        unsigned_manifest = {
            "schema_version": BINARY_MANIFEST_SCHEMA,
            "source_packed_manifest_sha256": packed_manifest["manifest_sha256"],
            "binding_sha256": self._binding.binding_sha256,
            "tokenizer_sha256": self._binding.tokenizer_sha256,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "vocabulary_size": self._binding.vocab_size,
            "token_storage": "little_endian_uint16",
            "index_struct": INDEX_STRUCT.format,
            "index_record_bytes": INDEX_STRUCT.size,
            "index_offset_unit": "uint16_tokens_relative_to_shard",
            "language_codes": language_names,
            "family_codes": family_names,
            "split_codes": split_names,
            "document_ids": {
                hashlib.sha256(document_id.encode("utf-8")).hexdigest(): document_id
                for document_id in sorted(document_ids)
            },
            "balancing_policy": "mean_cell_count_per_split_divided_by_language_family_cell_count",
            "cell_counts": {"|".join(cell): count for cell, count in sorted(cell_counts.items())},
            "cell_sampling_weights": {
                "|".join(cell): weights[cell] for cell in sorted(weights)
            },
            "weighted_cell_mass": weighted_mass,
            "maximum_tokens_per_shard": self.max_tokens_per_shard,
            "example_count": example_count,
            "total_sequence_tokens": total_tokens,
            "shard_count": len(shards),
            "shards": shards,
            "binary_ready": bool(shards) and example_count > 0,
            "training_ready": False,
            "training_blockers": [
                "this is a bounded pilot rather than the full source catalog",
                "held-out semantic correctness review is not complete",
            ],
        }
        manifest_sha256 = _sha256_text(_canonical_json(unsigned_manifest))
        manifest = {**unsigned_manifest, "manifest_sha256": manifest_sha256}
        manifest_path = staging_path / "manifest.json"
        try:
            with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{_canonical_json(manifest)}\n")
                handle.flush()
                os.fsync(handle.fileno())
            audited_rows = list(iter_binary_examples(staging_path))
            if (
                len(audited_rows) != example_count
                or sum(len(cast(list[int], row["input_ids"])) for row in audited_rows)
                != total_tokens
            ):
                raise ValueError("balanced binary output audit count mismatch")
            staged_files = sorted(
                (path for path in staging_path.iterdir() if path.name != "manifest.json"),
                key=lambda path: path.name,
            ) + [manifest_path]
            published: list[tuple[Path, Path]] = []
            try:
                for staged_path in staged_files:
                    destination = output_path / staged_path.name
                    if destination.exists():
                        raise ValueError("balanced binary output file already exists")
                    try:
                        os.link(staged_path, destination)
                    except FileExistsError as error:
                        raise ValueError(
                            "balanced binary output file already exists"
                        ) from error
                    published.append((destination, staged_path))
            except BaseException:
                for destination, staged_path in reversed(published):
                    if destination.exists() and destination.samefile(staged_path):
                        destination.unlink()
                raise
        finally:
            for path in staging_path.iterdir():
                if path.is_file():
                    path.unlink()
            staging_path.rmdir()
        return BalancedBinaryShardResult(
            output_root=output_path,
            example_count=example_count,
            token_count=total_tokens,
            shard_count=len(shards),
            shard_hashes=tuple(shard_hashes),
            manifest_sha256=manifest_sha256,
        )


def iter_binary_examples(root: str | Path) -> Iterator[dict[str, object]]:
    root_path = Path(root).resolve()
    try:
        manifest = json.loads((root_path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("binary manifest is invalid") from error
    if type(manifest) is not dict:
        raise ValueError("binary manifest is invalid")
    if manifest.get("schema_version") != BINARY_MANIFEST_SCHEMA:
        raise ValueError("binary manifest schema mismatch")
    claimed_hash = manifest.get("manifest_sha256")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if claimed_hash != _sha256_text(_canonical_json(unsigned_manifest)):
        raise ValueError("binary manifest SHA-256 mismatch")
    if (
        manifest.get("token_storage") != "little_endian_uint16"
        or manifest.get("index_struct") != INDEX_STRUCT.format
        or manifest.get("index_record_bytes") != INDEX_STRUCT.size
        or manifest.get("index_offset_unit")
        != "uint16_tokens_relative_to_shard"
        or manifest.get("parser_runtime_identity_sha256")
        != PARSER_RUNTIME_IDENTITY_SHA256
    ):
        raise ValueError("binary index struct mismatch")
    vocabulary_size = manifest.get("vocabulary_size")
    maximum_tokens_per_shard = manifest.get("maximum_tokens_per_shard")
    example_count = manifest.get("example_count")
    total_sequence_tokens = manifest.get("total_sequence_tokens")
    shard_count = manifest.get("shard_count")
    if (
        type(vocabulary_size) is not int
        or not 1 <= vocabulary_size <= 65_536
        or type(maximum_tokens_per_shard) is not int
        or maximum_tokens_per_shard < 1
        or type(example_count) is not int
        or example_count < 0
        or type(total_sequence_tokens) is not int
        or total_sequence_tokens < 0
        or type(shard_count) is not int
        or shard_count < 0
    ):
        raise ValueError("binary manifest counts are invalid")
    languages = manifest.get("language_codes")
    families = manifest.get("family_codes")
    splits = manifest.get("split_codes")
    if (
        type(languages) is not list
        or not languages
        or len(languages) > 255
        or any(
            type(language) is not str
            or language not in CURRENT_RUNTIME_SPEC_SHA256
            for language in languages
        )
        or len(set(languages)) != len(languages)
        or families != [
            "code_to_semantic",
            "semantic_continuation",
            "semantic_to_code",
        ]
        or splits != ["train", "validation", "test"]
    ):
        raise ValueError("binary manifest code tables are invalid")
    document_ids = manifest.get("document_ids")
    if type(document_ids) is not dict or any(
        type(digest) is not str
        or len(digest) != 64
        or type(document_id) is not str
        or not document_id
        or _sha256_text(document_id) != digest
        for digest, document_id in document_ids.items()
    ):
        raise ValueError("binary manifest document identities are invalid")
    shards = manifest.get("shards")
    if type(shards) is not list or len(shards) != shard_count:
        raise ValueError("binary manifest shard count mismatch")
    expected_files = {"manifest.json"}
    for shard_number, shard in enumerate(shards):
        if type(shard) is not dict or shard.get("shard") != shard_number:
            raise ValueError("binary manifest shard identity is invalid")
        expected_token_name = f"tasks-{shard_number:05d}.u16"
        expected_index_name = f"tasks-{shard_number:05d}.idx"
        if (
            shard.get("token_file") != expected_token_name
            or shard.get("index_file") != expected_index_name
        ):
            raise ValueError("binary manifest file inventory is invalid")
        expected_files.update((expected_token_name, expected_index_name))
    try:
        actual_entries = list(root_path.iterdir())
    except OSError as error:
        raise ValueError("binary shard directory is invalid") from error
    if (
        {path.name for path in actual_entries} != expected_files
        or any(not path.is_file() for path in actual_entries)
    ):
        raise ValueError("binary manifest file inventory is invalid")

    yielded_examples = 0
    yielded_tokens = 0
    seen_example_ids: set[str] = set()
    for shard_number, shard in enumerate(shards):
        token_path = root_path / f"tasks-{shard_number:05d}.u16"
        index_path = root_path / f"tasks-{shard_number:05d}.idx"
        shard_token_count = shard.get("token_count")
        shard_example_count = shard.get("example_count")
        if (
            type(shard_token_count) is not int
            or not 0 <= shard_token_count <= maximum_tokens_per_shard
            or type(shard_example_count) is not int
            or shard_example_count < 0
        ):
            raise ValueError("binary shard counts are invalid")
        try:
            if _sha256_file(token_path) != shard.get("token_sha256"):
                raise ValueError("binary token shard SHA-256 mismatch")
            if _sha256_file(index_path) != shard.get("index_sha256"):
                raise ValueError("binary index shard SHA-256 mismatch")
            token_bytes = token_path.read_bytes()
            index_bytes = index_path.read_bytes()
        except OSError as error:
            raise ValueError("binary shard file is invalid") from error
        if len(token_bytes) != shard_token_count * 2:
            raise ValueError("binary token shard count mismatch")
        if len(index_bytes) != shard_example_count * INDEX_STRUCT.size:
            raise ValueError("binary index shard count mismatch")
        expected_token_offset = 0
        shard_examples_read = 0
        for offset in range(0, len(index_bytes), INDEX_STRUCT.size):
            (
                token_offset,
                length,
                loss_start,
                sampling_weight,
                language_code,
                family_code,
                split_code,
                reserved,
                example_id,
                document_id,
                semantic_spec_sha256,
                source_sha256,
            ) = INDEX_STRUCT.unpack_from(index_bytes, offset)
            if (
                reserved != 0
                or token_offset != expected_token_offset
                or length < 1
                or not 0 < loss_start < length
                or not math.isfinite(sampling_weight)
                or sampling_weight <= 0
                or language_code >= len(languages)
                or family_code >= len(families)
                or split_code >= len(splits)
            ):
                raise ValueError("binary index record is invalid")
            start = token_offset * 2
            end = start + length * 2
            if end > len(token_bytes):
                raise ValueError("binary index points outside token shard")
            example_id_hex = example_id.hex()
            document_id_hex = document_id.hex()
            language = languages[language_code]
            decoded_tokens = decode_u16_tokens(token_bytes[start:end])
            if (
                example_id_hex in seen_example_ids
                or document_id_hex not in document_ids
                or semantic_spec_sha256.hex()
                != CURRENT_RUNTIME_SPEC_SHA256[language]
                or any(token_id >= vocabulary_size for token_id in decoded_tokens)
            ):
                raise ValueError("binary index lineage is invalid")
            seen_example_ids.add(example_id_hex)
            expected_token_offset += length
            shard_examples_read += 1
            yielded_examples += 1
            yielded_tokens += length
            yield {
                "example_id": example_id_hex,
                "document_id": document_ids[document_id_hex],
                "semantic_spec_sha256": semantic_spec_sha256.hex(),
                "source_sha256": source_sha256.hex(),
                "input_ids": decoded_tokens,
                "loss_start_index": loss_start,
                "sampling_weight": sampling_weight,
                "language": language,
                "family": families[family_code],
                "split": splits[split_code],
            }
        if (
            expected_token_offset != shard_token_count
            or shard_examples_read != shard_example_count
        ):
            raise ValueError("binary shard aggregate count mismatch")
    if yielded_examples != example_count or yielded_tokens != total_sequence_tokens:
        raise ValueError("binary manifest aggregate count mismatch")
