"""Compact uint16 task shards with deterministic balanced sampling weights."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import cast, Iterator, TypedDict

from .current_task_packing import EXAMPLE_SCHEMA, PACKED_MANIFEST_SCHEMA
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
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("packed manifest is invalid") from error
    if manifest.get("schema_version") != PACKED_MANIFEST_SCHEMA:
        raise ValueError("packed manifest schema mismatch")
    claimed_hash = manifest.pop("manifest_sha256", None)
    if claimed_hash != _sha256_text(_canonical_json(manifest)):
        raise ValueError("packed manifest SHA-256 mismatch")
    if _sha256_file(examples_path) != manifest.get("examples_sha256"):
        raise ValueError("packed examples SHA-256 mismatch")
    if (
        manifest.get("binding_sha256") != binding.binding_sha256
        or manifest.get("tokenizer_sha256") != binding.tokenizer_sha256
        or manifest.get("vocabulary_size") != binding.vocab_size
    ):
        raise ValueError("packed tokenizer lineage mismatch")
    return {**manifest, "manifest_sha256": claimed_hash}


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
    if row.get("split") not in {"train", "validation", "test"}:
        raise ValueError("packed split is invalid")
    if row.get("family") not in {
        "code_to_semantic",
        "semantic_continuation",
        "semantic_to_code",
    }:
        raise ValueError("packed family is invalid")
    if type(row.get("language")) is not str or not row["language"]:
        raise ValueError("packed language is invalid")
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
        examples_path = tasks_path / "examples.jsonl"

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

        def finish_shard() -> None:
            nonlocal token_handle, index_handle, shard_tokens, shard_examples
            if token_handle is None or index_handle is None or token_path is None or index_path is None:
                return
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
            token_path = output_path / f"tasks-{shard_number:05d}.u16"
            index_path = output_path / f"tasks-{shard_number:05d}.idx"
            token_handle = token_path.open("wb")
            index_handle = index_path.open("wb")

        with examples_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = _parse_example(line, self._binding)
                input_ids = row["input_ids"]
                sequence_length = len(input_ids)
                if sequence_length > self.max_tokens_per_shard:
                    raise ValueError("packed example exceeds max_tokens_per_shard")
                if token_handle is None:
                    start_shard()
                elif shard_tokens + sequence_length > self.max_tokens_per_shard:
                    finish_shard()
                    start_shard()
                if token_handle is None or index_handle is None:
                    raise RuntimeError("shard output handles were not initialized")
                cell = (str(row["split"]), str(row["language"]), str(row["family"]))
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
                        hashlib.sha256(str(row["document_id"]).encode("utf-8")).digest(),
                        bytes.fromhex(str(row["semantic_spec_sha256"])),
                        bytes.fromhex(str(row["source_sha256"])),
                    )
                )
                shard_tokens += sequence_length
                shard_examples += 1
        finish_shard()

        unsigned_manifest = {
            "schema_version": BINARY_MANIFEST_SCHEMA,
            "source_packed_manifest_sha256": packed_manifest["manifest_sha256"],
            "binding_sha256": self._binding.binding_sha256,
            "tokenizer_sha256": self._binding.tokenizer_sha256,
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
        temporary = output_path / ".manifest.json.tmp"
        temporary.write_text(f"{_canonical_json(manifest)}\n", encoding="utf-8", newline="\n")
        temporary.replace(output_path / "manifest.json")
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
    if manifest.get("schema_version") != BINARY_MANIFEST_SCHEMA:
        raise ValueError("binary manifest schema mismatch")
    claimed_hash = manifest.pop("manifest_sha256", None)
    if claimed_hash != _sha256_text(_canonical_json(manifest)):
        raise ValueError("binary manifest SHA-256 mismatch")
    if manifest.get("index_struct") != INDEX_STRUCT.format:
        raise ValueError("binary index struct mismatch")
    languages = manifest["language_codes"]
    families = manifest["family_codes"]
    splits = manifest["split_codes"]
    document_ids = manifest["document_ids"]
    for shard in manifest["shards"]:
        token_path = root_path / shard["token_file"]
        index_path = root_path / shard["index_file"]
        if _sha256_file(token_path) != shard["token_sha256"]:
            raise ValueError("binary token shard SHA-256 mismatch")
        if _sha256_file(index_path) != shard["index_sha256"]:
            raise ValueError("binary index shard SHA-256 mismatch")
        token_bytes = token_path.read_bytes()
        index_bytes = index_path.read_bytes()
        if len(index_bytes) % INDEX_STRUCT.size:
            raise ValueError("binary index shard is truncated")
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
            if reserved != 0:
                raise ValueError("binary index reserved byte is nonzero")
            start = token_offset * 2
            end = start + length * 2
            if end > len(token_bytes):
                raise ValueError("binary index points outside token shard")
            yield {
                "example_id": example_id.hex(),
                "document_id": document_ids[document_id.hex()],
                "semantic_spec_sha256": semantic_spec_sha256.hex(),
                "source_sha256": source_sha256.hex(),
                "input_ids": decode_u16_tokens(token_bytes[start:end]),
                "loss_start_index": loss_start,
                "sampling_weight": sampling_weight,
                "language": languages[language_code],
                "family": families[family_code],
                "split": splits[split_code],
            }
