"""Full-source replay gate for held-out BCNLT semantic review records."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, TypeGuard

from .bcnlt_source_pipeline import (
    PARSER_RUNTIME_IDENTITY_SHA256,
    BCNLTPreprocessingError,
    BCNLTSourcePipeline,
)
from .current_task_packing import CurrentCorpusTaskPacker
from .heldout_semantic_review import REVIEW_ITEM_SCHEMA, REVIEW_MANIFEST_SCHEMA
from .immutable_artifacts import publish_new_files
from .jsonl import split_jsonl_records
from .tokenizer import NativeTokenizerBinding

_REPLAY_ITEM_SCHEMA = "bosscrafts-source-semantic-replay-item-v1"
_REPLAY_MANIFEST_SCHEMA = "bosscrafts-source-semantic-replay-manifest-v1"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_canonical_records(
    canonical_jsonl: str, *, name: str
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for line in split_jsonl_records(canonical_jsonl):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} contains invalid JSON") from exc
        if type(record) is not dict or line != _canonical(record):
            raise ValueError(f"{name} contains a non-canonical semantic record")
        records.append(record)
    return tuple(records)


class _ReplayDocument(TypedDict):
    canonical_jsonl: str | None
    canonical_sha256: str | None
    records: tuple[dict[str, object], ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class SourceSemanticReplayResult:
    output_root: Path
    item_count: int
    document_count: int
    matched_item_count: int
    mismatched_item_count: int
    complete_source_queue: bool
    training_ready: bool
    results_sha256: str
    manifest_sha256: str


class SourceSemanticReplayAuditor:
    """Reparse admitted full source with the pinned runtime and compare records."""

    def __init__(self, binding: NativeTokenizerBinding) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        self._binding = binding

    def materialize(
        self,
        admission_root: str | Path,
        review_root: str | Path,
        output_root: str | Path,
        *,
        review_ids: tuple[str, ...] | None = None,
    ) -> SourceSemanticReplayResult:
        output = Path(output_root).resolve()
        results_text, manifest = self._evaluate(
            Path(admission_root).resolve(),
            Path(review_root).resolve(),
            review_ids,
        )
        publish_new_files(
            (
                (output / "replay_results.jsonl", results_text.encode("utf-8")),
                (
                    output / "manifest.json",
                    f"{_canonical(manifest)}\n".encode("utf-8"),
                ),
            ),
            artifact_label="source semantic replay",
        )
        return self._result(output, manifest)

    def verify(
        self,
        admission_root: str | Path,
        review_root: str | Path,
        output_root: str | Path,
        *,
        review_ids: tuple[str, ...] | None = None,
    ) -> SourceSemanticReplayResult:
        output = Path(output_root).resolve()
        expected_results, expected_manifest = self._evaluate(
            Path(admission_root).resolve(),
            Path(review_root).resolve(),
            review_ids,
        )
        try:
            actual_results = (output / "replay_results.jsonl").read_text(
                encoding="utf-8"
            )
            actual_manifest = (output / "manifest.json").read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError("source semantic replay evidence is stale or tampered") from exc
        if (
            actual_results != expected_results
            or actual_manifest != f"{_canonical(expected_manifest)}\n"
        ):
            raise ValueError("source semantic replay evidence is stale or tampered")
        return self._result(output, expected_manifest)

    def _evaluate(
        self,
        admission_root: Path,
        review_root: Path,
        review_ids: tuple[str, ...] | None,
    ) -> tuple[str, dict[str, object]]:
        admission_manifest, admission_rows = CurrentCorpusTaskPacker(
            self._binding
        )._load_admission(admission_root)
        review_manifest, review_rows = self._load_review(review_root)
        if (
            review_manifest.get("source_admission_manifest_sha256")
            != admission_manifest.get("manifest_sha256")
            or review_manifest.get("binding_sha256") != self._binding.binding_sha256
            or review_manifest.get("tokenizer_sha256") != self._binding.tokenizer_sha256
            or review_manifest.get("parser_runtime_identity_sha256")
            != PARSER_RUNTIME_IDENTITY_SHA256
        ):
            raise ValueError("review and admission replay lineage mismatch")
        selected = self._select(review_rows, review_ids)
        admission_by_id = {str(row["document_id"]): row for row in admission_rows}
        selected_document_ids = tuple(
            dict.fromkeys(str(row["document_id"]) for row in selected)
        )
        missing = sorted(set(selected_document_ids) - set(admission_by_id))
        if missing:
            raise ValueError(f"review documents are absent from admission: {missing}")

        replayed: dict[str, _ReplayDocument] = {}
        with BCNLTSourcePipeline(
            self._binding,
            persistent_sidecars=True,
            learning_mode="ephemeral",
        ) as pipeline:
            for document_id in selected_document_ids:
                admitted = admission_by_id[document_id]
                source = admitted.get("source")
                language = admitted.get("language")
                relative_path = admitted.get("relative_path")
                dialect = admitted.get("dialect")
                if (
                    type(source) is not str
                    or type(language) is not str
                    or type(relative_path) is not str
                    or not relative_path
                    or (dialect is not None and type(dialect) is not str)
                ):
                    raise ValueError(
                        "admission replay source, language, path, or dialect is invalid"
                    )
                try:
                    prepared = pipeline.prepare_semantic(
                        source,
                        language=language,
                        path=relative_path,
                        dialect=dialect,
                    )
                except BCNLTPreprocessingError as exc:
                    replayed[document_id] = {
                        "canonical_jsonl": None,
                        "canonical_sha256": None,
                        "records": (),
                        "error": str(exc),
                    }
                    continue
                canonical_jsonl = prepared.canonical_jsonl
                if canonical_jsonl is None:
                    raise ValueError("semantic replay unexpectedly returned raw code")
                replayed[document_id] = {
                    "canonical_jsonl": canonical_jsonl,
                    "canonical_sha256": prepared.canonical_sha256,
                    "records": _parse_canonical_records(
                        canonical_jsonl,
                        name="replayed semantic stream",
                    ),
                    "error": None,
                }

        output_rows: list[dict[str, object]] = []
        stream_mismatch_documents: set[str] = set()
        matched = 0
        counts_by_language: dict[str, dict[str, int]] = {}
        for review in selected:
            document_id = str(review["document_id"])
            language = str(review["language"])
            admitted = admission_by_id[document_id]
            replay = replayed[document_id]
            admitted_jsonl = str(admitted["canonical_jsonl"])
            admitted_records = _parse_canonical_records(
                admitted_jsonl,
                name="admitted semantic stream",
            )
            record_index = review.get("record_index")
            if type(record_index) is not int or not 0 <= record_index < len(admitted_records):
                raise ValueError("review record index is outside admission semantics")
            queued_record = review.get("canonical_record")
            admitted_record = admitted_records[record_index]
            replay_records = replay["records"]
            replay_record = (
                replay_records[record_index]
                if record_index < len(replay_records)
                else None
            )
            queue_admission_match = queued_record == admitted_record
            document_stream_match = replay["canonical_jsonl"] == admitted_jsonl
            record_match = (
                queue_admission_match
                and replay_record is not None
                and replay_record == admitted_record
            )
            if not document_stream_match:
                stream_mismatch_documents.add(document_id)
            if record_match:
                matched += 1
            language_counts = counts_by_language.setdefault(
                language, {"matched": 0, "mismatched": 0}
            )
            language_counts["matched" if record_match else "mismatched"] += 1
            output_rows.append(
                {
                    "schema_version": _REPLAY_ITEM_SCHEMA,
                    "review_id": review["review_id"],
                    "document_id": document_id,
                    "language": language,
                    "record_index": record_index,
                    "source_sha256": admitted["source_sha256"],
                    "expected_record_sha256": _sha256_text(
                        _canonical(admitted_record)
                    ),
                    "replayed_record_sha256": (
                        _sha256_text(_canonical(replay_record))
                        if replay_record is not None
                        else None
                    ),
                    "admitted_document_canonical_sha256": admitted[
                        "canonical_sha256"
                    ],
                    "replayed_document_canonical_sha256": replay[
                        "canonical_sha256"
                    ],
                    "queue_admission_match": queue_admission_match,
                    "document_stream_match": document_stream_match,
                    "record_match": record_match,
                    "replay_error": replay["error"],
                    "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                }
            )
        results_text = "".join(f"{_canonical(row)}\n" for row in output_rows)
        results_sha256 = _sha256_text(results_text)
        mismatched = len(selected) - matched
        complete = len(selected) == len(review_rows)
        replay_passed = mismatched == 0 and not stream_mismatch_documents
        blockers = ["source semantic replay alone does not authorize training"]
        if not complete:
            blockers.insert(0, "source replay covers only an explicit review subset")
        if not replay_passed:
            blockers.insert(0, "full-source semantic replay produced mismatches")
        unsigned: dict[str, object] = {
            "schema_version": _REPLAY_MANIFEST_SCHEMA,
            "source_admission_manifest_sha256": admission_manifest["manifest_sha256"],
            "source_review_manifest_sha256": review_manifest["manifest_sha256"],
            "source_queue_sha256": review_manifest["queue_sha256"],
            "binding_sha256": self._binding.binding_sha256,
            "tokenizer_sha256": self._binding.tokenizer_sha256,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "source_review_item_count": len(review_rows),
            "replayed_item_count": len(selected),
            "replayed_document_count": len(selected_document_ids),
            "record_match_count": matched,
            "record_mismatch_count": mismatched,
            "document_stream_mismatch_count": len(stream_mismatch_documents),
            "counts_by_language": {
                key: counts_by_language[key] for key in sorted(counts_by_language)
            },
            "complete_source_queue": complete,
            "semantic_replay_passed": replay_passed,
            "results_file": "replay_results.jsonl",
            "results_sha256": results_sha256,
            "training_ready": False,
            "training_blockers": blockers,
        }
        return results_text, {
            **unsigned,
            "manifest_sha256": _sha256_text(_canonical(unsigned)),
        }

    @staticmethod
    def _select(
        rows: list[dict[str, object]], review_ids: tuple[str, ...] | None
    ) -> list[dict[str, object]]:
        if review_ids is None:
            return list(rows)
        if (
            type(review_ids) is not tuple
            or not review_ids
            or any(type(value) is not str or not value for value in review_ids)
            or len(set(review_ids)) != len(review_ids)
        ):
            raise ValueError("review_ids must be a non-empty tuple of unique IDs")
        by_id = {str(row["review_id"]): row for row in rows}
        unknown = sorted(set(review_ids) - set(by_id))
        if unknown:
            raise ValueError(f"unknown replay review IDs: {unknown}")
        selected_ids = set(review_ids)
        return [row for row in rows if str(row["review_id"]) in selected_ids]

    @staticmethod
    def _load_review(
        root: Path,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("semantic review manifest is invalid") from exc
        if type(manifest) is not dict or manifest.get("schema_version") != REVIEW_MANIFEST_SCHEMA:
            raise ValueError("semantic review manifest schema mismatch")
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if claimed != _sha256_text(_canonical(unsigned)):
            raise ValueError("semantic review manifest SHA-256 mismatch")
        try:
            queue_text = (root / str(manifest.get("queue_file"))).read_text(
                encoding="utf-8"
            )
        except OSError as exc:
            raise ValueError("semantic review queue cannot be read") from exc
        if _sha256_text(queue_text) != manifest.get("queue_sha256"):
            raise ValueError("semantic review queue SHA-256 mismatch")
        rows: list[dict[str, object]] = []
        ids: set[str] = set()
        for line in split_jsonl_records(queue_text):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("semantic review queue JSON is invalid") from exc
            review_id = row.get("review_id") if type(row) is dict else None
            if (
                type(row) is not dict
                or line != _canonical(row)
                or row.get("schema_version") != REVIEW_ITEM_SCHEMA
                or type(review_id) is not str
                or review_id in ids
                or row.get("parser_runtime_identity_sha256")
                != manifest.get("parser_runtime_identity_sha256")
            ):
                raise ValueError("semantic review queue item is invalid")
            ids.add(review_id)
            rows.append(row)
        if len(rows) != manifest.get("item_count"):
            raise ValueError("semantic review queue item count mismatch")
        return manifest, rows

    @staticmethod
    def _result(
        output: Path, manifest: dict[str, object]
    ) -> SourceSemanticReplayResult:
        item_count = manifest.get("replayed_item_count")
        document_count = manifest.get("replayed_document_count")
        matched_item_count = manifest.get("record_match_count")
        mismatched_item_count = manifest.get("record_mismatch_count")
        complete_source_queue = manifest.get("complete_source_queue")
        training_ready = manifest.get("training_ready")
        results_sha256 = manifest.get("results_sha256")
        manifest_sha256 = manifest.get("manifest_sha256")
        if (
            type(item_count) is not int
            or item_count < 0
            or type(document_count) is not int
            or document_count < 0
            or type(matched_item_count) is not int
            or matched_item_count < 0
            or type(mismatched_item_count) is not int
            or mismatched_item_count < 0
            or matched_item_count + mismatched_item_count != item_count
            or type(complete_source_queue) is not bool
            or type(training_ready) is not bool
            or not _is_sha256(results_sha256)
            or not _is_sha256(manifest_sha256)
        ):
            raise ValueError("source semantic replay result is invalid")
        return SourceSemanticReplayResult(
            output_root=output,
            item_count=item_count,
            document_count=document_count,
            matched_item_count=matched_item_count,
            mismatched_item_count=mismatched_item_count,
            complete_source_queue=complete_source_queue,
            training_ready=training_ready,
            results_sha256=results_sha256,
            manifest_sha256=manifest_sha256,
        )
