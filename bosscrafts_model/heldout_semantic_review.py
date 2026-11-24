"""Deterministic held-out review queues for current-spec semantic records."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from .current_task_packing import (
    EXAMPLE_SCHEMA,
    FAMILIES,
    PACKED_MANIFEST_SCHEMA,
    CurrentCorpusTaskPacker,
)
from .jsonl import split_jsonl_records
from .tokenizer import NativeTokenizerBinding

REVIEW_ITEM_SCHEMA = "bosscrafts-heldout-semantic-review-item-v2"
REVIEW_MANIFEST_SCHEMA = "bosscrafts-heldout-semantic-review-manifest-v2"


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


@dataclass(frozen=True)
class HeldoutSemanticReviewResult:
    output_root: Path
    item_count: int
    queue_sha256: str
    manifest_sha256: str
    represented_languages: tuple[str, ...]
    missing_languages: tuple[str, ...]


@dataclass(frozen=True)
class HeldoutSemanticAdjudicationResult:
    output_root: Path
    item_count: int
    decision_counts: dict[str, int]
    adjudicated_sha256: str
    manifest_sha256: str
    review_complete: bool
    semantic_review_passed: bool


class HeldoutSemanticReviewQueue:
    """Select source-aligned semantic records from validation and test documents."""

    def __init__(self, binding: NativeTokenizerBinding, *, per_language: int = 2) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        if type(per_language) is not int or per_language < 1:
            raise ValueError("per_language must be a positive integer")
        self._binding = binding
        self.per_language = per_language

    def _load_tasks(self, root: Path) -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
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
        claimed_manifest_hash = manifest.get("manifest_sha256")
        unsigned_manifest = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        if claimed_manifest_hash != _sha256_text(_canonical_json(unsigned_manifest)):
            raise ValueError("packed manifest SHA-256 mismatch")
        if (
            manifest.get("examples_file") != "examples.jsonl"
            or manifest.get("rejections_file") != "rejected.jsonl"
        ):
            raise ValueError("packed manifest file inventory mismatch")
        try:
            examples_text = examples_path.read_text(encoding="utf-8")
            rejections_text = rejections_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError("packed task files cannot be read") from error
        if _sha256_text(examples_text) != manifest.get("examples_sha256"):
            raise ValueError("packed examples SHA-256 mismatch")
        if _sha256_text(rejections_text) != manifest.get("rejections_sha256"):
            raise ValueError("packed rejections SHA-256 mismatch")
        if (
            manifest.get("binding_sha256") != self._binding.binding_sha256
            or manifest.get("tokenizer_sha256") != self._binding.tokenizer_sha256
            or manifest.get("vocabulary_size") != self._binding.vocab_size
            or manifest.get("parser_runtime_identity_sha256")
            != PARSER_RUNTIME_IDENTITY_SHA256
        ):
            raise ValueError("packed tokenizer lineage mismatch")

        eligible: list[dict[str, object]] = []
        all_languages: set[str] = set()
        example_ids: set[str] = set()
        document_splits: dict[str, set[str]] = {}
        family_counts = {family: 0 for family in FAMILIES}
        split_counts = {split: 0 for split in ("train", "validation", "test")}
        for line in split_jsonl_records(examples_text):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("packed example JSON is invalid") from error
            if type(row) is not dict or row.get("schema_version") != EXAMPLE_SCHEMA:
                raise ValueError("packed example schema mismatch")
            if (
                row.get("binding_sha256") != self._binding.binding_sha256
                or row.get("tokenizer_sha256") != self._binding.tokenizer_sha256
                or row.get("parser_runtime_identity_sha256")
                != PARSER_RUNTIME_IDENTITY_SHA256
            ):
                raise ValueError("packed example tokenizer lineage mismatch")
            language = row.get("language")
            family = row.get("family")
            split = row.get("split")
            document_id = row.get("document_id")
            record_index = row.get("record_index")
            example_id = row.get("example_id")
            input_ids = row.get("input_ids")
            loss_start = row.get("loss_start_index")
            if (
                type(language) is not str
                or language not in CURRENT_RUNTIME_SPEC_SHA256
                or row.get("semantic_spec_sha256")
                != CURRENT_RUNTIME_SPEC_SHA256[language]
                or family not in FAMILIES
                or split not in split_counts
                or type(document_id) is not str
                or not document_id
                or type(record_index) is not int
                or record_index < 0
                or type(example_id) is not str
                or not example_id
                or type(input_ids) is not list
                or any(type(token_id) is not int for token_id in input_ids)
                or any(
                    not 0 <= token_id < self._binding.vocab_size
                    for token_id in input_ids
                )
                or len(input_ids) > manifest.get("maximum_sequence_tokens", 0)
                or type(loss_start) is not int
                or not 0 < loss_start < len(input_ids)
                or row.get("target_token_count") != len(input_ids) - loss_start
                or input_ids[0]
                != self._binding.structural_token_ids[f"<|bcnlt:task:{family}|>"]
                or input_ids[-1]
                != self._binding.structural_token_ids["<|bcnlt:task:end|>"]
            ):
                raise ValueError("packed example content is invalid")
            expected_example_id = _sha256_text(
                _canonical_json(
                    {
                        "document_id": document_id,
                        "family": family,
                        "record_index": record_index,
                        "input_ids_sha256": _sha256_text(
                            _canonical_json(input_ids)
                        ),
                    }
                )
            )
            if example_id != expected_example_id or example_id in example_ids:
                raise ValueError("packed example identity is invalid")
            example_ids.add(example_id)
            document_splits.setdefault(document_id, set()).add(str(split))
            family_counts[str(family)] += 1
            split_counts[str(split)] += 1
            all_languages.add(language)
            if family == "code_to_semantic" and split in {"validation", "test"}:
                eligible.append(row)
        if any(len(splits) != 1 for splits in document_splits.values()):
            raise ValueError("packed examples violate document split isolation")
        if (
            len(example_ids) != manifest.get("example_count")
            or family_counts != manifest.get("family_counts")
            or split_counts != manifest.get("split_counts")
            or len(split_jsonl_records(rejections_text))
            != manifest.get("rejected_count")
        ):
            raise ValueError("packed task manifest counts mismatch")
        if len(all_languages) > 255:
            raise ValueError("review language count exceeds supported range")
        return dict(manifest), eligible, all_languages

    def materialize(
        self,
        admission_root: str | Path,
        tasks_root: str | Path,
        output_root: str | Path,
    ) -> HeldoutSemanticReviewResult:
        admission_path = Path(admission_root).resolve()
        tasks_path = Path(tasks_root).resolve()
        output_path = Path(output_root).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        admission_manifest, admission_rows = CurrentCorpusTaskPacker(
            self._binding
        )._load_admission(admission_path)
        packed_manifest, eligible, all_languages = self._load_tasks(tasks_path)
        if any(output_path.iterdir()):
            raise ValueError("held-out review output already contains files")
        if packed_manifest.get("source_admission_manifest_sha256") != admission_manifest.get(
            "manifest_sha256"
        ):
            raise ValueError("packed task admission lineage mismatch")
        admitted_by_document = {str(row["document_id"]): row for row in admission_rows}

        selected: list[dict[str, object]] = []
        counts: dict[str, int] = {}
        for example in sorted(eligible, key=lambda row: str(row["example_id"])):
            language = str(example["language"])
            if counts.get(language, 0) >= self.per_language:
                continue
            admitted = admitted_by_document.get(str(example["document_id"]))
            if admitted is None:
                raise ValueError("packed example document is absent from admission records")
            canonical_lines = split_jsonl_records(str(admitted["canonical_jsonl"]))
            record_index_value = example["record_index"]
            if type(record_index_value) is not int:
                raise ValueError("packed example record index is not an integer")
            record_index = record_index_value
            if not 0 <= record_index < len(canonical_lines):
                raise ValueError("packed example record index is outside admission records")
            try:
                canonical_record = json.loads(canonical_lines[record_index])
            except json.JSONDecodeError as error:
                raise ValueError("admission canonical record JSON is invalid") from error
            source_excerpt = self._review_source_context(
                str(admitted["source"]), canonical_record
            )
            identity = {
                "example_id": example["example_id"],
                "source_sha256": admitted["source_sha256"],
                "semantic_spec_sha256": admitted["semantic_spec_sha256"],
                "parser_runtime_identity_sha256": admitted[
                    "parser_runtime_identity_sha256"
                ],
            }
            selected.append(
                {
                    "schema_version": REVIEW_ITEM_SCHEMA,
                    "review_id": _sha256_text(_canonical_json(identity)),
                    "example_id": example["example_id"],
                    "document_id": example["document_id"],
                    "record_index": record_index,
                    "language": language,
                    "split": example["split"],
                    "family": example["family"],
                    "source_excerpt": source_excerpt,
                    "canonical_record": canonical_record,
                    "source_sha256": admitted["source_sha256"],
                    "semantic_spec_sha256": admitted["semantic_spec_sha256"],
                    "parser_runtime_identity_sha256": admitted[
                        "parser_runtime_identity_sha256"
                    ],
                    "binding_sha256": self._binding.binding_sha256,
                    "tokenizer_sha256": self._binding.tokenizer_sha256,
                    "review_status": "unreviewed",
                    "rubric": {
                        "event_matches_source": None,
                        "ir_kind_plausible": None,
                        "scope_plausible": None,
                        "span_matches_source": None,
                        "uncertainty_calibrated": None,
                    },
                    "review_notes": None,
                }
            )
            counts[language] = counts.get(language, 0) + 1

        selected.sort(key=lambda row: (str(row["language"]), str(row["review_id"])))
        queue_text = "".join(f"{_canonical_json(row)}\n" for row in selected)
        queue_sha256 = _sha256_text(queue_text)
        represented = tuple(sorted(counts))
        missing = tuple(sorted(all_languages - set(represented)))
        unsigned_manifest = {
            "schema_version": REVIEW_MANIFEST_SCHEMA,
            "source_admission_manifest_sha256": admission_manifest["manifest_sha256"],
            "source_packed_manifest_sha256": packed_manifest["manifest_sha256"],
            "binding_sha256": self._binding.binding_sha256,
            "tokenizer_sha256": self._binding.tokenizer_sha256,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "selection_policy": "lowest_example_id_code_to_semantic_from_validation_or_test_per_language_numbered_context_radius_3",
            "requested_items_per_language": self.per_language,
            "item_count": len(selected),
            "represented_languages": list(represented),
            "missing_languages": list(missing),
            "queue_file": "review_queue.jsonl",
            "queue_sha256": queue_sha256,
            "review_complete": False,
            "semantic_review_passed": False,
            "training_ready": False,
            "training_blockers": [
                "review queue has not been independently adjudicated",
                "this is a bounded pilot rather than the full source catalog",
            ],
        }
        manifest_sha256 = _sha256_text(_canonical_json(unsigned_manifest))
        manifest = {**unsigned_manifest, "manifest_sha256": manifest_sha256}
        self._publish_new_files(
            (
                (output_path / "review_queue.jsonl", queue_text),
                (output_path / "manifest.json", f"{_canonical_json(manifest)}\n"),
            )
        )
        return HeldoutSemanticReviewResult(
            output_root=output_path,
            item_count=len(selected),
            queue_sha256=queue_sha256,
            manifest_sha256=manifest_sha256,
            represented_languages=represented,
            missing_languages=missing,
        )

    def materialize_admission_only(
        self,
        admission_root: str | Path,
        output_root: str | Path,
    ) -> HeldoutSemanticReviewResult:
        """Build one review record per held-out document without task packing."""
        admission_path = Path(admission_root).resolve()
        output_path = Path(output_root).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        admission_manifest, admission_rows = CurrentCorpusTaskPacker(
            self._binding
        )._load_admission(admission_path)
        if any(output_path.iterdir()):
            raise ValueError("held-out review output already contains files")
        all_languages = {str(row["language"]) for row in admission_rows}
        counts: dict[str, int] = {}
        selected: list[dict[str, object]] = []
        for admitted in sorted(
            admission_rows, key=lambda row: (str(row["language"]), str(row["document_id"]))
        ):
            language = str(admitted["language"])
            if counts.get(language, 0) >= self.per_language:
                continue
            split = CurrentCorpusTaskPacker._split(str(admitted["document_id"]))
            if split not in {"validation", "test"}:
                continue
            canonical_lines = split_jsonl_records(str(admitted["canonical_jsonl"]))
            if not canonical_lines:
                continue
            selector = hashlib.sha256(
                f"{admitted['document_id']}:semantic-review".encode()
            ).digest()
            record_index = int.from_bytes(selector[:8], "big") % len(canonical_lines)
            try:
                canonical_record = json.loads(canonical_lines[record_index])
            except json.JSONDecodeError as error:
                raise ValueError("admission canonical record JSON is invalid") from error
            source_excerpt = self._review_source_context(
                str(admitted["source"]), canonical_record
            )
            identity = {
                "document_id": admitted["document_id"],
                "record_index": record_index,
                "source_sha256": admitted["source_sha256"],
                "semantic_spec_sha256": admitted["semantic_spec_sha256"],
                "parser_runtime_identity_sha256": admitted[
                    "parser_runtime_identity_sha256"
                ],
            }
            selected.append(
                {
                    "schema_version": REVIEW_ITEM_SCHEMA,
                    "review_id": _sha256_text(_canonical_json(identity)),
                    "example_id": None,
                    "document_id": admitted["document_id"],
                    "record_index": record_index,
                    "language": language,
                    "split": split,
                    "family": "semantic_record_review",
                    "source_excerpt": source_excerpt,
                    "canonical_record": canonical_record,
                    "source_sha256": admitted["source_sha256"],
                    "semantic_spec_sha256": admitted["semantic_spec_sha256"],
                    "parser_runtime_identity_sha256": admitted[
                        "parser_runtime_identity_sha256"
                    ],
                    "binding_sha256": self._binding.binding_sha256,
                    "tokenizer_sha256": self._binding.tokenizer_sha256,
                    "review_status": "unreviewed",
                    "rubric": {
                        "event_matches_source": None,
                        "ir_kind_plausible": None,
                        "scope_plausible": None,
                        "span_matches_source": None,
                        "uncertainty_calibrated": None,
                    },
                    "review_notes": None,
                }
            )
            counts[language] = counts.get(language, 0) + 1
        selected.sort(key=lambda row: (str(row["language"]), str(row["review_id"])))
        queue_text = "".join(f"{_canonical_json(row)}\n" for row in selected)
        queue_sha256 = _sha256_text(queue_text)
        represented = tuple(sorted(counts))
        missing = tuple(sorted(all_languages - set(represented)))
        unsigned_manifest = {
            "schema_version": REVIEW_MANIFEST_SCHEMA,
            "source_admission_manifest_sha256": admission_manifest["manifest_sha256"],
            "source_packed_manifest_sha256": None,
            "binding_sha256": self._binding.binding_sha256,
            "tokenizer_sha256": self._binding.tokenizer_sha256,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "selection_policy": "one_sha256_selected_record_per_heldout_document_then_lowest_document_id_per_language_numbered_context_radius_3",
            "requested_items_per_language": self.per_language,
            "item_count": len(selected),
            "represented_languages": list(represented),
            "missing_languages": list(missing),
            "queue_file": "review_queue.jsonl",
            "queue_sha256": queue_sha256,
            "review_complete": False,
            "semantic_review_passed": False,
            "training_ready": False,
            "training_blockers": [
                "review queue has not been independently adjudicated",
                "review admission is bounded and does not authorize full training",
            ],
        }
        manifest_sha256 = _sha256_text(_canonical_json(unsigned_manifest))
        manifest = {**unsigned_manifest, "manifest_sha256": manifest_sha256}
        self._publish_new_files(
            (
                (output_path / "review_queue.jsonl", queue_text),
                (output_path / "manifest.json", f"{_canonical_json(manifest)}\n"),
            )
        )
        return HeldoutSemanticReviewResult(
            output_root=output_path,
            item_count=len(selected),
            queue_sha256=queue_sha256,
            manifest_sha256=manifest_sha256,
            represented_languages=represented,
            missing_languages=missing,
        )

    @staticmethod
    def _review_source_context(
        source: str,
        canonical_record: dict[str, object],
        *,
        radius: int = 3,
    ) -> str:
        span = canonical_record.get("span")
        if (
            type(span) is not list
            or len(span) != 4
            or any(type(value) is not int or value < 0 for value in span)
            or span[0] < 1
            or span[2] < span[0]
        ):
            raise ValueError("semantic review record span is invalid")
        # Tree-sitter advances rows only on literal LF bytes.  Preserve CR
        # bytes inside each logical row so span evidence can canonicalize the
        # exact byte slice after extraction instead of silently deleting it.
        source_lines = source.split("\n")
        if not source_lines or span[2] > len(source_lines):
            raise ValueError("semantic review record span is outside the source")
        first = max(1, span[0] - radius)
        last = min(len(source_lines), span[2] + radius)
        header = (
            f"[source context lines {first}-{last}; target span "
            f"{span[0]}:{span[1]}-{span[2]}:{span[3]}]"
        )
        rendered = [header]
        for line_number in range(first, last + 1):
            line = source_lines[line_number - 1]
            if len(line) > 4096:
                line = f"{line[:4096]} [line truncated after 4096 characters]"
            marker = ">" if span[0] <= line_number <= span[2] else " "
            rendered.append(f"{marker}{line_number:06d}| {line}")
        result = "\n".join(rendered) + "\n"
        if len(result.encode("utf-8")) > 32 * 1024:
            raise ValueError("semantic review source context exceeds the size limit")
        return result

    @staticmethod
    def _publish_new_files(files: tuple[tuple[Path, str], ...]) -> None:
        staged: list[tuple[Path, Path]] = []
        published: list[tuple[Path, Path]] = []
        try:
            for path, content in files:
                temporary_handle = tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                )
                temporary = Path(temporary_handle.name)
                try:
                    temporary_handle.write(content)
                    temporary_handle.flush()
                    os.fsync(temporary_handle.fileno())
                finally:
                    temporary_handle.close()
                staged.append((path, temporary))
            for path, temporary in staged:
                if path.exists():
                    raise ValueError("held-out review output file already exists")
                try:
                    os.link(temporary, path)
                except FileExistsError as error:
                    raise ValueError(
                        "held-out review output file already exists"
                    ) from error
                published.append((path, temporary))
        except BaseException:
            for path, temporary in reversed(published):
                if path.exists() and path.samefile(temporary):
                    path.unlink()
            raise
        finally:
            for _, temporary in staged:
                if temporary.exists():
                    temporary.unlink()


class HeldoutSemanticReviewAdjudicator:
    """Validate and freeze explicit human/model decisions for a review queue."""

    _RUBRIC_FIELDS: ClassVar[frozenset[str]] = frozenset(
        (
            "event_matches_source",
            "ir_kind_plausible",
            "scope_plausible",
            "span_matches_source",
            "uncertainty_calibrated",
        )
    )

    def materialize(
        self,
        queue_root: str | Path,
        decisions: Iterable[object],
        output_root: str | Path,
    ) -> HeldoutSemanticAdjudicationResult:
        queue_path = Path(queue_root).resolve()
        output_path = Path(output_root).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            source_manifest = json.loads((queue_path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("review queue manifest is invalid") from error
        if type(source_manifest) is not dict:
            raise ValueError("review queue manifest is invalid")
        if source_manifest.get("schema_version") != REVIEW_MANIFEST_SCHEMA:
            raise ValueError("review queue manifest schema mismatch")
        source_manifest_sha256 = source_manifest.get("manifest_sha256")
        unsigned_source_manifest = {
            key: value
            for key, value in source_manifest.items()
            if key != "manifest_sha256"
        }
        if source_manifest_sha256 != _sha256_text(
            _canonical_json(unsigned_source_manifest)
        ):
            raise ValueError("review queue manifest SHA-256 mismatch")
        if source_manifest.get("queue_file") != "review_queue.jsonl":
            raise ValueError("review queue manifest file inventory mismatch")
        queue_file = queue_path / "review_queue.jsonl"
        queue_text = queue_file.read_text(encoding="utf-8")
        if _sha256_text(queue_text) != source_manifest.get("queue_sha256"):
            raise ValueError("review queue SHA-256 mismatch")
        items: list[dict[str, object]] = []
        for line in split_jsonl_records(queue_text):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("review queue item JSON is invalid") from error
            if type(item) is not dict or item.get("schema_version") != REVIEW_ITEM_SCHEMA:
                raise ValueError("review queue item schema mismatch")
            if (
                item.get("parser_runtime_identity_sha256")
                != source_manifest.get("parser_runtime_identity_sha256")
                or item.get("binding_sha256")
                != source_manifest.get("binding_sha256")
                or item.get("tokenizer_sha256")
                != source_manifest.get("tokenizer_sha256")
            ):
                raise ValueError("review queue tokenizer lineage mismatch")
            example_id = item.get("example_id")
            if example_id is None:
                identity = {
                    "document_id": item.get("document_id"),
                    "record_index": item.get("record_index"),
                    "source_sha256": item.get("source_sha256"),
                    "semantic_spec_sha256": item.get("semantic_spec_sha256"),
                    "parser_runtime_identity_sha256": item.get(
                        "parser_runtime_identity_sha256"
                    ),
                }
            else:
                identity = {
                    "example_id": example_id,
                    "source_sha256": item.get("source_sha256"),
                    "semantic_spec_sha256": item.get("semantic_spec_sha256"),
                    "parser_runtime_identity_sha256": item.get(
                        "parser_runtime_identity_sha256"
                    ),
                }
            if (
                item.get("review_id") != _sha256_text(_canonical_json(identity))
                or item.get("review_status") != "unreviewed"
                or type(item.get("canonical_record")) is not dict
                or type(item.get("source_excerpt")) is not str
            ):
                raise ValueError("review queue item identity is invalid")
            items.append(item)
        if len(items) != source_manifest.get("item_count"):
            raise ValueError("review queue item count mismatch")
        item_ids = [str(item["review_id"]) for item in items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("review queue contains duplicate review IDs")

        try:
            supplied = list(decisions)
        except TypeError as error:
            raise TypeError("decisions must be iterable") from error
        decisions_by_id: dict[str, dict[str, object]] = {}
        for decision in supplied:
            if type(decision) is not dict:
                raise ValueError("review decision must be an object")
            review_id = decision.get("review_id")
            if type(review_id) is not str or review_id not in set(item_ids):
                raise ValueError("review decision references an unknown review ID")
            if review_id in decisions_by_id:
                raise ValueError("review decisions contain a duplicate review ID")
            choice = decision.get("decision")
            if choice not in {"keep", "reject", "uncertain"}:
                raise ValueError("review decision must be keep, reject, or uncertain")
            reviewer = decision.get("reviewer")
            notes = decision.get("notes")
            rubric = decision.get("rubric")
            if type(reviewer) is not str or not reviewer:
                raise ValueError("review decision reviewer is required")
            if type(notes) is not str or not notes:
                raise ValueError("review decision notes are required")
            if type(rubric) is not dict or set(rubric) != self._RUBRIC_FIELDS:
                raise ValueError("review decision rubric fields mismatch")
            if any(type(value) is not bool for value in rubric.values()):
                raise ValueError("review decision rubric values must be boolean")
            decisions_by_id[review_id] = decision
        missing = sorted(set(item_ids) - set(decisions_by_id))
        if missing:
            raise ValueError(f"missing review decisions: {missing}")
        if any(output_path.iterdir()):
            raise ValueError("held-out adjudication output already contains files")

        adjudicated: list[dict[str, object]] = []
        counts = {"keep": 0, "reject": 0, "uncertain": 0}
        for item in items:
            decision = decisions_by_id[str(item["review_id"])]
            choice = str(decision["decision"])
            counts[choice] += 1
            adjudicated.append(
                {
                    **item,
                    "review_status": choice,
                    "rubric": decision["rubric"],
                    "review_notes": decision["notes"],
                    "reviewer": decision["reviewer"],
                }
            )
        adjudicated_text = "".join(f"{_canonical_json(item)}\n" for item in adjudicated)
        adjudicated_sha256 = _sha256_text(adjudicated_text)
        complete = sum(counts.values()) == len(items)
        passed = complete and counts["reject"] == 0 and counts["uncertain"] == 0 and not source_manifest.get(
            "missing_languages"
        )
        blockers = ["review admission is bounded and does not authorize full training"]
        if not passed:
            blockers.insert(0, "held-out semantic correctness review did not pass")
        unsigned_manifest = {
            "schema_version": "bosscrafts-heldout-semantic-adjudication-manifest-v2",
            "source_review_manifest_sha256": source_manifest_sha256,
            "source_queue_sha256": source_manifest["queue_sha256"],
            "binding_sha256": source_manifest["binding_sha256"],
            "tokenizer_sha256": source_manifest["tokenizer_sha256"],
            "parser_runtime_identity_sha256": source_manifest[
                "parser_runtime_identity_sha256"
            ],
            "item_count": len(items),
            "decision_counts": counts,
            "represented_languages": source_manifest["represented_languages"],
            "missing_languages": source_manifest["missing_languages"],
            "adjudicated_file": "adjudicated_queue.jsonl",
            "adjudicated_sha256": adjudicated_sha256,
            "review_complete": complete,
            "semantic_review_passed": passed,
            "training_ready": False,
            "training_blockers": blockers,
        }
        manifest_sha256 = _sha256_text(_canonical_json(unsigned_manifest))
        manifest = {**unsigned_manifest, "manifest_sha256": manifest_sha256}
        HeldoutSemanticReviewQueue._publish_new_files(
            (
                (output_path / "adjudicated_queue.jsonl", adjudicated_text),
                (output_path / "manifest.json", f"{_canonical_json(manifest)}\n"),
            )
        )
        return HeldoutSemanticAdjudicationResult(
            output_root=output_path,
            item_count=len(items),
            decision_counts=counts,
            adjudicated_sha256=adjudicated_sha256,
            manifest_sha256=manifest_sha256,
            review_complete=complete,
            semantic_review_passed=passed,
        )
