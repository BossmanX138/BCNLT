"""Record-boundary task packing for admitted current-spec BCNLT documents."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import cast

from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from .current_corpus import MANIFEST_SCHEMA, RECORD_SCHEMA
from .tokenizer import NativeTokenizerBinding


EXAMPLE_SCHEMA = "bosscrafts-current-task-example-v2"
PACKED_MANIFEST_SCHEMA = "bosscrafts-current-task-manifest-v2"
FAMILIES = ("code_to_semantic", "semantic_continuation", "semantic_to_code")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CurrentTaskPackingResult:
    output_root: Path
    example_count: int
    rejected_count: int
    examples_sha256: str
    rejections_sha256: str
    manifest_sha256: str
    family_counts: dict[str, int]
    split_counts: dict[str, int]
    packing_ready: bool
    training_ready: bool


class CurrentCorpusTaskPacker:
    def __init__(
        self,
        binding: NativeTokenizerBinding,
        *,
        max_sequence: int = 2048,
    ) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        if type(max_sequence) is not int or max_sequence < 64:
            raise ValueError("max_sequence must be an integer of at least 64")
        self._binding = binding
        self.max_sequence = max_sequence
        self._ids = binding.structural_token_ids

    def _id(self, name: str) -> int:
        return self._ids[f"<|bcnlt:{name}|>"]

    def _load_admission(self, root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
        manifest_path = root / "manifest.json"
        records_path = root / "admitted.jsonl"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("admission manifest is invalid") from error
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError("admission manifest schema mismatch")
        claimed_manifest_hash = manifest.pop("manifest_sha256", None)
        if claimed_manifest_hash != _sha256_text(_canonical_json(manifest)):
            raise ValueError("admission manifest SHA-256 mismatch")
        records_text = records_path.read_text(encoding="utf-8")
        if _sha256_text(records_text) != manifest.get("records_sha256"):
            raise ValueError("admission records SHA-256 mismatch")
        if (
            manifest.get("binding_sha256") != self._binding.binding_sha256
            or manifest.get("tokenizer_sha256") != self._binding.tokenizer_sha256
            or manifest.get("vocabulary_size") != self._binding.vocab_size
            or manifest.get("parser_runtime_identity_sha256")
            != PARSER_RUNTIME_IDENTITY_SHA256
        ):
            raise ValueError("admission tokenizer lineage mismatch")
        rows: list[dict[str, object]] = []
        for line in records_text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("admission record JSON is invalid") from error
            if type(row) is not dict or row.get("schema_version") != RECORD_SCHEMA:
                raise ValueError("admission record schema mismatch")
            if (
                row.get("binding_sha256") != self._binding.binding_sha256
                or row.get("tokenizer_sha256") != self._binding.tokenizer_sha256
                or row.get("parser_runtime_identity_sha256")
                != PARSER_RUNTIME_IDENTITY_SHA256
            ):
                raise ValueError("admission record tokenizer lineage mismatch")
            rows.append(row)
        if len(rows) != manifest.get("admitted_count"):
            raise ValueError("admission record count mismatch")
        return {**manifest, "manifest_sha256": claimed_manifest_hash}, rows

    def _semantic_record_fragments(self, row: dict[str, object]) -> list[list[int]]:
        ids = row.get("semantic_token_ids")
        if type(ids) is not list or any(type(value) is not int for value in ids):
            raise ValueError("semantic_token_ids must be an integer list")
        if any(not 0 <= value < self._binding.vocab_size for value in ids):
            raise ValueError("semantic_token_ids exceed frozen vocabulary")
        if ids[:2] != [self._id("stream:begin"), self._id("mode:semantic_code")]:
            raise ValueError("semantic document stream header mismatch")
        if ids[-1:] != [self._id("stream:end")]:
            raise ValueError("semantic document stream end mismatch")
        record_begin = self._id("record:begin")
        record_end = self._id("record:end")
        fragments: list[list[int]] = []
        cursor = 2
        while cursor < len(ids) - 1:
            if ids[cursor] != record_begin:
                raise ValueError("semantic record boundary mismatch")
            try:
                end = ids.index(record_end, cursor + 1)
            except ValueError as error:
                raise ValueError("semantic record is truncated") from error
            fragments.append(ids[cursor : end + 1])
            cursor = end + 1
        if len(fragments) != row.get("semantic_record_count"):
            raise ValueError("semantic record fragment count mismatch")
        return fragments

    @staticmethod
    def _split(document_id: str) -> str:
        bucket = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket < 80:
            return "train"
        if bucket < 90:
            return "validation"
        return "test"

    @staticmethod
    def _source_snippet(source: str, canonical_record: dict[str, object]) -> str:
        span = canonical_record.get("span")
        if type(span) is not list or len(span) != 4:
            raise ValueError("canonical record span mismatch")
        lines = source.splitlines(keepends=True)
        start_line = int(span[0])
        end_line = int(span[2])
        start_index = max(0, start_line - 1 if start_line > 0 else 0)
        end_index = max(start_index + 1, end_line if end_line > 0 else 1)
        snippet = "".join(lines[start_index:end_index])
        if not snippet:
            surface = canonical_record.get("surface")
            snippet = surface if type(surface) is str else ""
        return snippet

    def _stream(self, mode: str, payload: list[int]) -> list[int]:
        return [
            self._id("stream:begin"),
            self._id(f"mode:{mode}"),
            *payload,
            self._id("stream:end"),
        ]

    def pack_row(
        self, row: dict[str, object]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Pack one verified admission row without retaining other documents."""
        examples: list[dict[str, object]] = []
        rejections: list[dict[str, object]] = []
        task_end = self._id("task:end")
        fragments = self._semantic_record_fragments(row)
        canonical_lines = str(row["canonical_jsonl"]).splitlines()
        if len(canonical_lines) != len(fragments):
            raise ValueError("canonical and tokenized semantic record counts differ")
        canonical_records = [json.loads(line) for line in canonical_lines]
        source = str(row["source"])
        code_to_semantic_groups: dict[
            tuple[int, ...], list[tuple[int, list[int]]]
        ] = {}
        for index, target_fragment in enumerate(fragments):
            semantic_target = self._stream("semantic_code", target_fragment)
            snippet = self._source_snippet(source, canonical_records[index])
            try:
                raw_payload = self._binding.encode_payload(snippet)
            except ValueError:
                rejections.append(
                    self._rejection(row, index, "reserved_token_in_source_snippet")
                )
                continue
            raw_target = self._stream("raw_code", raw_payload)
            code_to_semantic_groups.setdefault(tuple(raw_target), []).append(
                (index, target_fragment)
            )

            semantic_to_code = [
                self._id("task:semantic_to_code"),
                *semantic_target,
                *raw_target,
                task_end,
            ]
            self._admit_example(
                examples,
                rejections,
                row=row,
                family="semantic_to_code",
                record_index=index,
                input_ids=semantic_to_code,
                loss_start_index=1 + len(semantic_target),
            )

            if index > 0:
                fixed = 1 + 3 + 1 + 2 + len(target_fragment) + 1 + 1
                available = self.max_sequence - fixed
                context: list[list[int]] = []
                used = 0
                for candidate_index in range(index - 1, -1, -1):
                    candidate = fragments[candidate_index]
                    if used + len(candidate) > available:
                        break
                    context.insert(0, candidate)
                    used += len(candidate)
                if context:
                    prompt_stream = self._stream(
                        "semantic_code",
                        [token for fragment in context for token in fragment],
                    )
                    continuation = [
                        self._id("task:semantic_continuation"),
                        *prompt_stream,
                        *semantic_target,
                        task_end,
                    ]
                    self._admit_example(
                        examples,
                        rejections,
                        row=row,
                        family="semantic_continuation",
                        record_index=index,
                        input_ids=continuation,
                        loss_start_index=1 + len(prompt_stream),
                    )
                else:
                    rejections.append(
                        self._rejection(
                            row,
                            index,
                            "semantic_context_over_max_sequence",
                            "semantic_continuation",
                        )
                    )
        for raw_target_tuple, grouped_records in code_to_semantic_groups.items():
            grouped_records.sort(key=lambda item: item[0])
            first_record_index = grouped_records[0][0]
            semantic_target = self._stream(
                "semantic_code",
                [
                    token
                    for _, target_fragment in grouped_records
                    for token in target_fragment
                ],
            )
            raw_target = list(raw_target_tuple)
            code_to_semantic = [
                self._id("task:code_to_semantic"),
                *raw_target,
                *semantic_target,
                task_end,
            ]
            self._admit_example(
                examples,
                rejections,
                row=row,
                family="code_to_semantic",
                record_index=first_record_index,
                input_ids=code_to_semantic,
                loss_start_index=1 + len(raw_target),
            )
        return examples, rejections

    def _example(
        self,
        *,
        row: dict[str, object],
        family: str,
        record_index: int,
        input_ids: list[int],
        loss_start_index: int,
    ) -> dict[str, object]:
        identity = {
            "document_id": row["document_id"],
            "family": family,
            "record_index": record_index,
            "input_ids_sha256": _sha256_text(_canonical_json(input_ids)),
        }
        return {
            "schema_version": EXAMPLE_SCHEMA,
            "example_id": _sha256_text(_canonical_json(identity)),
            "document_id": row["document_id"],
            "language": row["language"],
            "family": family,
            "record_index": record_index,
            "split": self._split(str(row["document_id"])),
            "input_ids": input_ids,
            "loss_start_index": loss_start_index,
            "target_token_count": len(input_ids) - loss_start_index,
            "binding_sha256": self._binding.binding_sha256,
            "tokenizer_sha256": self._binding.tokenizer_sha256,
            "semantic_spec_sha256": row["semantic_spec_sha256"],
            "parser_runtime_identity_sha256": row[
                "parser_runtime_identity_sha256"
            ],
            "source_sha256": row["source_sha256"],
        }

    def materialize(
        self,
        admission_root: str | Path,
        output_root: str | Path,
    ) -> CurrentTaskPackingResult:
        admission_path = Path(admission_root).resolve()
        output_path = Path(output_root).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        admission_manifest, rows = self._load_admission(admission_path)
        examples: list[dict[str, object]] = []
        rejections: list[dict[str, object]] = []

        for row in rows:
            row_examples, row_rejections = self.pack_row(row)
            examples.extend(row_examples)
            rejections.extend(row_rejections)

        examples.sort(key=lambda item: str(item["example_id"]))
        rejections.sort(
            key=lambda item: (
                str(item["document_id"]),
                cast(int, item["record_index"]),
                str(item.get("family")),
                str(item["reason_code"]),
            )
        )
        examples_text = "".join(f"{_canonical_json(item)}\n" for item in examples)
        rejections_text = "".join(f"{_canonical_json(item)}\n" for item in rejections)
        examples_sha256 = _sha256_text(examples_text)
        rejections_sha256 = _sha256_text(rejections_text)
        family_counts = {
            family: sum(item["family"] == family for item in examples) for family in FAMILIES
        }
        split_counts = {
            split: sum(item["split"] == split for item in examples)
            for split in ("train", "validation", "test")
        }
        unsigned_manifest = {
            "schema_version": PACKED_MANIFEST_SCHEMA,
            "source_admission_manifest_sha256": admission_manifest["manifest_sha256"],
            "binding_sha256": self._binding.binding_sha256,
            "tokenizer_sha256": self._binding.tokenizer_sha256,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "vocabulary_size": self._binding.vocab_size,
            "maximum_sequence_tokens": self.max_sequence,
            "example_count": len(examples),
            "rejected_count": len(rejections),
            "examples_file": "examples.jsonl",
            "examples_sha256": examples_sha256,
            "rejections_file": "rejected.jsonl",
            "rejections_sha256": rejections_sha256,
            "family_counts": family_counts,
            "split_counts": split_counts,
            "maximum_observed_sequence_tokens": max(
                (len(cast(list[int], item["input_ids"])) for item in examples),
                default=0,
            ),
            "total_sequence_tokens": sum(
                len(cast(list[int], item["input_ids"])) for item in examples
            ),
            "loss_mask_policy": "labels_before_loss_start_index_are_masked",
            "code_to_semantic_target_policy": (
                "one_complete_semantic_stream_per_unique_raw_snippet"
            ),
            "split_policy": "sha256(document_id)_mod_100:train_0_79,validation_80_89,test_90_99",
            "packing_ready": bool(examples) and all(family_counts.values()),
            "training_ready": False,
            "training_blockers": [
                "this is a bounded pilot rather than the full source catalog",
                "held-out semantic correctness review is not complete",
            ],
        }
        manifest_sha256 = _sha256_text(_canonical_json(unsigned_manifest))
        manifest = {**unsigned_manifest, "manifest_sha256": manifest_sha256}
        self._replace_text(output_path / "examples.jsonl", examples_text)
        self._replace_text(output_path / "rejected.jsonl", rejections_text)
        self._replace_text(output_path / "manifest.json", f"{_canonical_json(manifest)}\n")
        return CurrentTaskPackingResult(
            output_root=output_path,
            example_count=len(examples),
            rejected_count=len(rejections),
            examples_sha256=examples_sha256,
            rejections_sha256=rejections_sha256,
            manifest_sha256=manifest_sha256,
            family_counts=family_counts,
            split_counts=split_counts,
            packing_ready=bool(examples) and all(family_counts.values()),
            training_ready=False,
        )

    def _admit_example(
        self,
        examples: list[dict[str, object]],
        rejections: list[dict[str, object]],
        *,
        row: dict[str, object],
        family: str,
        record_index: int,
        input_ids: list[int],
        loss_start_index: int,
    ) -> None:
        if len(input_ids) > self.max_sequence:
            rejections.append(
                self._rejection(row, record_index, "example_over_max_sequence", family)
            )
            return
        if not 0 < loss_start_index < len(input_ids):
            raise ValueError("loss_start_index is outside the packed sequence")
        if any(not 0 <= token_id < self._binding.vocab_size for token_id in input_ids):
            raise ValueError("packed task token exceeds frozen vocabulary")
        examples.append(
            self._example(
                row=row,
                family=family,
                record_index=record_index,
                input_ids=input_ids,
                loss_start_index=loss_start_index,
            )
        )

    @staticmethod
    def _rejection(
        row: dict[str, object],
        record_index: int,
        reason_code: str,
        family: str | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": "bosscrafts-current-task-rejection-v1",
            "document_id": row["document_id"],
            "language": row["language"],
            "record_index": record_index,
            "family": family,
            "reason_code": reason_code,
        }

    @staticmethod
    def _replace_text(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
