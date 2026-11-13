"""Build blinded paired semantic controls from verified natural-runtime labels."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .heldout_semantic_review import REVIEW_ITEM_SCHEMA, REVIEW_MANIFEST_SCHEMA


_CHALLENGE_SCHEMA = "bosscrafts-semantic-challenge-review-v1"
_ANSWER_SCHEMA = "bosscrafts-semantic-challenge-answer-v1"
_DEFECT_TYPES = (
    "surface_mismatch",
    "span_mismatch",
    "scope_mismatch",
    "event_ir_mismatch",
    "uncertainty_understated",
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class SemanticChallengeReviewResult:
    output_root: Path
    item_count: int
    languages: tuple[str, ...]
    queue_sha256: str
    answer_key_sha256: str
    manifest_sha256: str


class SemanticChallengeReviewBuilder:
    """Create blind positive/negative pairs without changing preserved sources."""

    def __init__(self, *, pairs_per_language: int = 10) -> None:
        if (
            type(pairs_per_language) is not int
            or pairs_per_language < len(_DEFECT_TYPES)
            or pairs_per_language % len(_DEFECT_TYPES) != 0
        ):
            raise ValueError("pairs_per_language must be a positive multiple of five")
        self.pairs_per_language = pairs_per_language

    def materialize(
        self,
        source_queue_root: str | Path,
        natural_labels_root: str | Path,
        output_root: str | Path,
    ) -> SemanticChallengeReviewResult:
        source_root = Path(source_queue_root).resolve()
        labels_root = Path(natural_labels_root).resolve()
        output_path = Path(output_root).resolve()
        source_manifest, source_items = self._load_source_queue(source_root)
        verified_ids = self._load_natural_labels(labels_root, source_manifest)
        source_by_id = {str(item["review_id"]): item for item in source_items}
        if verified_ids != set(source_by_id):
            raise ValueError("natural labels do not cover the complete source queue")

        grouped: dict[str, list[dict[str, object]]] = {}
        for item in source_items:
            grouped.setdefault(str(item["language"]), []).append(item)
        challenge_items: list[dict[str, object]] = []
        answer_rows: list[dict[str, object]] = []
        for language in sorted(grouped):
            candidates = sorted(grouped[language], key=lambda row: str(row["review_id"]))
            selected = self._select_bases(candidates)
            for pair_index, (source_item, defect_type) in enumerate(
                zip(selected, self._defect_schedule(), strict=True)
            ):
                source_record = deepcopy(source_item["canonical_record"])
                negative_record = self._mutate_record(
                    source_record,
                    defect_type,
                    candidates,
                )
                pair_identity = _sha256_text(
                    _canonical(
                        {
                            "language": language,
                            "pair_index": pair_index,
                            "source_queue_sha256": source_manifest["queue_sha256"],
                            "source_review_id": source_item["review_id"],
                        }
                    )
                )
                for variant, expected, record, defect in (
                    ("a", "keep", source_record, "none"),
                    ("b", "reject", negative_record, defect_type),
                ):
                    record_sha256 = _sha256_text(_canonical(record))
                    review_id = _sha256_text(
                        _canonical(
                            {
                                "pair_identity": pair_identity,
                                "record_sha256": record_sha256,
                                "variant": variant,
                            }
                        )
                    )
                    item = self._challenge_item(source_item, review_id, record)
                    challenge_items.append(item)
                    answer_rows.append(
                        {
                            "schema_version": _ANSWER_SCHEMA,
                            "review_id": review_id,
                            "pair_identity_sha256": pair_identity,
                            "source_review_id": source_item["review_id"],
                            "language": language,
                            "expected_decision": expected,
                            "defect_type": defect,
                            "canonical_record_sha256": record_sha256,
                            "parser_runtime_identity_sha256": source_manifest[
                                "parser_runtime_identity_sha256"
                            ],
                        }
                    )
        challenge_items.sort(key=lambda row: str(row["review_id"]))
        answer_rows.sort(key=lambda row: str(row["review_id"]))
        queue_text = "".join(f"{_canonical(row)}\n" for row in challenge_items)
        answer_text = "".join(f"{_canonical(row)}\n" for row in answer_rows)
        queue_sha256 = _sha256_text(queue_text)
        answer_sha256 = _sha256_text(answer_text)
        expected_counts = {
            label: sum(row["expected_decision"] == label for row in answer_rows)
            for label in ("keep", "reject")
        }
        unsigned = {
            "schema_version": REVIEW_MANIFEST_SCHEMA,
            "challenge_schema_version": _CHALLENGE_SCHEMA,
            "source_admission_manifest_sha256": source_manifest.get(
                "source_admission_manifest_sha256"
            ),
            "source_packed_manifest_sha256": source_manifest.get(
                "source_packed_manifest_sha256"
            ),
            "source_review_manifest_sha256": source_manifest["manifest_sha256"],
            "source_queue_sha256": source_manifest["queue_sha256"],
            "source_natural_labels_manifest_sha256": self._load_manifest(
                labels_root / "manifest.json", "natural labels"
            )["manifest_sha256"],
            "binding_sha256": source_manifest["binding_sha256"],
            "tokenizer_sha256": source_manifest["tokenizer_sha256"],
            "parser_runtime_identity_sha256": source_manifest[
                "parser_runtime_identity_sha256"
            ],
            "selection_policy": (
                "paired_blinded_controls_sha256_order_five_defect_cycle"
            ),
            "pairs_per_language": self.pairs_per_language,
            "requested_items_per_language": self.pairs_per_language * 2,
            "item_count": len(challenge_items),
            "represented_languages": sorted(grouped),
            "missing_languages": [],
            "expected_counts": expected_counts,
            "queue_file": "review_queue.jsonl",
            "queue_sha256": queue_sha256,
            "answer_key_file": "control_answer_key.jsonl",
            "answer_key_sha256": answer_sha256,
            "review_complete": False,
            "semantic_review_passed": False,
            "training_ready": False,
            "training_blockers": [
                "blinded challenge controls require independent review",
                "challenge evidence alone does not authorize training",
            ],
        }
        manifest_sha256 = _sha256_text(_canonical(unsigned))
        manifest = {**unsigned, "manifest_sha256": manifest_sha256}
        _atomic_write(output_path / "review_queue.jsonl", queue_text.encode("utf-8"))
        _atomic_write(
            output_path / "control_answer_key.jsonl", answer_text.encode("utf-8")
        )
        _atomic_write(
            output_path / "manifest.json",
            f"{_canonical(manifest)}\n".encode("utf-8"),
        )
        return self.verify(output_path)

    def verify(self, output_root: str | Path) -> SemanticChallengeReviewResult:
        root = Path(output_root).resolve()
        manifest = self._load_manifest(root / "manifest.json", "challenge")
        if (
            manifest.get("schema_version") != REVIEW_MANIFEST_SCHEMA
            or manifest.get("challenge_schema_version") != _CHALLENGE_SCHEMA
            or manifest.get("pairs_per_language") != self.pairs_per_language
        ):
            raise ValueError("semantic challenge manifest policy mismatch")
        queue_text = self._read_hashed_text(
            root / str(manifest.get("queue_file")),
            manifest.get("queue_sha256"),
            "challenge queue",
        )
        answer_text = self._read_hashed_text(
            root / str(manifest.get("answer_key_file")),
            manifest.get("answer_key_sha256"),
            "challenge answer key",
        )
        queue_rows = self._load_canonical_rows(queue_text, "challenge queue")
        answer_rows = self._load_canonical_rows(answer_text, "challenge answer key")
        if (
            len(queue_rows) != manifest.get("item_count")
            or len(answer_rows) != len(queue_rows)
        ):
            raise ValueError("semantic challenge item inventory mismatch")
        queue_by_id: dict[str, dict[str, object]] = {}
        for row in queue_rows:
            review_id = row.get("review_id")
            if (
                row.get("schema_version") != REVIEW_ITEM_SCHEMA
                or type(review_id) is not str
                or review_id in queue_by_id
                or row.get("parser_runtime_identity_sha256")
                != manifest.get("parser_runtime_identity_sha256")
                or {"expected_decision", "defect_type", "control_kind"}.intersection(row)
            ):
                raise ValueError("semantic challenge queue item is invalid")
            queue_by_id[review_id] = row
        answer_by_id: dict[str, dict[str, object]] = {}
        for row in answer_rows:
            review_id = row.get("review_id")
            if (
                row.get("schema_version") != _ANSWER_SCHEMA
                or type(review_id) is not str
                or review_id in answer_by_id
                or row.get("expected_decision") not in {"keep", "reject"}
                or row.get("defect_type") not in {"none", *_DEFECT_TYPES}
                or row.get("parser_runtime_identity_sha256")
                != manifest.get("parser_runtime_identity_sha256")
            ):
                raise ValueError("semantic challenge answer row is invalid")
            answer_by_id[review_id] = row
        if set(queue_by_id) != set(answer_by_id):
            raise ValueError("semantic challenge queue and answer IDs differ")
        for review_id, answer in answer_by_id.items():
            record = queue_by_id[review_id].get("canonical_record")
            if answer.get("canonical_record_sha256") != _sha256_text(
                _canonical(record)
            ):
                raise ValueError("semantic challenge record identity mismatch")
        actual_counts = {
            label: sum(row["expected_decision"] == label for row in answer_rows)
            for label in ("keep", "reject")
        }
        if actual_counts != manifest.get("expected_counts"):
            raise ValueError("semantic challenge expected counts mismatch")
        languages = tuple(sorted({str(row["language"]) for row in answer_rows}))
        if list(languages) != manifest.get("represented_languages"):
            raise ValueError("semantic challenge language inventory mismatch")
        return SemanticChallengeReviewResult(
            output_root=root,
            item_count=len(queue_rows),
            languages=languages,
            queue_sha256=str(manifest["queue_sha256"]),
            answer_key_sha256=str(manifest["answer_key_sha256"]),
            manifest_sha256=str(manifest["manifest_sha256"]),
        )

    def _defect_schedule(self) -> tuple[str, ...]:
        schedule = list(_DEFECT_TYPES)
        additional = self.pairs_per_language - len(schedule)
        repeatable = _DEFECT_TYPES[:-1]
        schedule.extend(repeatable[index % len(repeatable)] for index in range(additional))
        return tuple(schedule)

    def _select_bases(
        self, candidates: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        if len(candidates) < self.pairs_per_language:
            raise ValueError("language lacks enough verified natural records")
        schedule = self._defect_schedule()
        required_unknown = schedule.count("uncertainty_understated")
        uncertain_bases = [
            item
            for item in candidates
            if item["canonical_record"].get("event") == "UNKNOWN_SEMANTIC"
            and item["canonical_record"].get("uncertainty_band") in {"medium", "high"}
        ][:required_unknown]
        if len(uncertain_bases) != required_unknown:
            raise ValueError(
                "language lacks enough uncertain UNKNOWN_SEMANTIC negative control bases"
            )
        reserved_ids = {str(item["review_id"]) for item in uncertain_bases}
        unused = [
            item for item in candidates if str(item["review_id"]) not in reserved_ids
        ]
        selected: list[dict[str, object]] = []
        for defect in schedule:
            if defect == "uncertainty_understated":
                selected.append(uncertain_bases.pop(0))
            else:
                if not unused:
                    raise ValueError("language lacks enough distinct negative control bases")
                selected.append(unused.pop(0))
        return selected

    @staticmethod
    def _challenge_item(
        source_item: dict[str, object],
        review_id: str,
        record: dict[str, object],
    ) -> dict[str, object]:
        item = deepcopy(source_item)
        item["review_id"] = review_id
        item["canonical_record"] = record
        item["review_status"] = "unreviewed"
        item["rubric"] = {
            "event_matches_source": None,
            "ir_kind_plausible": None,
            "scope_plausible": None,
            "span_matches_source": None,
            "uncertainty_calibrated": None,
        }
        item["review_notes"] = None
        return item

    @staticmethod
    def _mutate_record(
        source_record: dict[str, object],
        defect_type: str,
        candidates: list[dict[str, object]],
    ) -> dict[str, object]:
        record = deepcopy(source_record)
        alternate_records = [
            item["canonical_record"]
            for item in candidates
            if item["canonical_record"] != source_record
        ]
        if defect_type == "surface_mismatch":
            alternate = next(
                (
                    candidate.get("surface")
                    for candidate in alternate_records
                    if candidate.get("surface") != record.get("surface")
                ),
                None,
            )
            surface = str(record.get("surface", ""))
            record["surface"] = (
                alternate
                if type(alternate) is str
                else (surface[::-1] if len(surface) > 1 else f"{surface} ")
            )
        elif defect_type == "span_mismatch":
            span = list(record.get("span", ()))
            if len(span) != 4 or any(type(value) is not int for value in span):
                raise ValueError("source record span is invalid")
            if span[0] == span[2] and span[3] - span[1] >= 2:
                span[1] += 1
            elif span[1] > 0:
                span[1] -= 1
            else:
                span[3] += 1
            record["span"] = span
        elif defect_type == "scope_mismatch":
            alternate = next(
                (
                    candidate.get("scope")
                    for candidate in alternate_records
                    if candidate.get("scope") != record.get("scope")
                ),
                None,
            )
            record["scope"] = (
                alternate if type(alternate) is str else "module.UnrelatedOwner"
            )
        elif defect_type == "event_ir_mismatch":
            alternate = next(
                (
                    candidate
                    for candidate in alternate_records
                    if candidate.get("event") != record.get("event")
                    or candidate.get("ir") != record.get("ir")
                ),
                None,
            )
            if alternate is None:
                record["event"] = "RETURN"
                record["ir"] = {
                    "kind": "return",
                    "name": None,
                    "value": None,
                    "children": [],
                    "metadata": {},
                }
            else:
                for field in ("event", "ir", "defines", "symbol_id", "ast_relation"):
                    record[field] = deepcopy(alternate.get(field))
        elif defect_type == "uncertainty_understated":
            if record.get("event") != "UNKNOWN_SEMANTIC":
                raise ValueError("uncertainty control base must be UNKNOWN_SEMANTIC")
            record["confidence_band"] = "high"
            record["uncertainty_band"] = "low"
            metadata = deepcopy(record.get("metadata", {}))
            metadata["knowledge_state"] = "known"
            metadata["uncertainty_signals"] = []
            metadata["uyr_rounds"] = 0
            record["metadata"] = metadata
        else:
            raise ValueError("unsupported semantic challenge defect type")
        if record == source_record:
            raise ValueError("semantic challenge mutation did not change the record")
        return record

    @staticmethod
    def _load_manifest(path: Path, name: str) -> dict[str, object]:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} manifest is invalid") from exc
        if type(manifest) is not dict:
            raise ValueError(f"{name} manifest must be an object")
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if claimed != _sha256_text(_canonical(unsigned)):
            raise ValueError(f"{name} manifest SHA-256 mismatch")
        return manifest

    @classmethod
    def _load_source_queue(
        cls, root: Path
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        manifest = cls._load_manifest(root / "manifest.json", "source queue")
        if manifest.get("schema_version") != REVIEW_MANIFEST_SCHEMA:
            raise ValueError("source queue manifest schema mismatch")
        queue_text = cls._read_hashed_text(
            root / str(manifest.get("queue_file")),
            manifest.get("queue_sha256"),
            "source queue",
        )
        rows = cls._load_canonical_rows(queue_text, "source queue")
        ids = [row.get("review_id") for row in rows]
        if (
            len(rows) != manifest.get("item_count")
            or len(ids) != len(set(ids))
            or any(row.get("schema_version") != REVIEW_ITEM_SCHEMA for row in rows)
        ):
            raise ValueError("source queue inventory mismatch")
        return manifest, rows

    @classmethod
    def _load_natural_labels(
        cls, root: Path, source_manifest: dict[str, object]
    ) -> set[str]:
        manifest = cls._load_manifest(root / "manifest.json", "natural labels")
        if (
            manifest.get("source_queue_sha256") != source_manifest.get("queue_sha256")
            or manifest.get("parser_runtime_identity_sha256")
            != source_manifest.get("parser_runtime_identity_sha256")
        ):
            raise ValueError("natural labels and source queue lineage mismatch")
        rows: list[dict[str, object]] = []
        for prefix in ("calibration", "heldout"):
            text = cls._read_hashed_text(
                root / str(manifest.get(f"{prefix}_file")),
                manifest.get(f"{prefix}_sha256"),
                f"natural {prefix} labels",
            )
            split_rows = cls._load_canonical_rows(text, f"natural {prefix} labels")
            if len(split_rows) != manifest.get(f"{prefix}_count"):
                raise ValueError("natural label split count mismatch")
            rows.extend(split_rows)
        ids: set[str] = set()
        for row in rows:
            review_id = row.get("review_id")
            if (
                type(review_id) is not str
                or review_id in ids
                or row.get("decision") != "keep"
                or row.get("incorrect") is not False
                or row.get("parser_runtime_identity_sha256")
                != source_manifest.get("parser_runtime_identity_sha256")
            ):
                raise ValueError("natural labels are not verified positive controls")
            ids.add(review_id)
        return ids

    @staticmethod
    def _read_hashed_text(path: Path, expected: object, name: str) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"{name} cannot be read") from exc
        if _sha256_text(text) != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
        return text

    @staticmethod
    def _load_canonical_rows(text: str, name: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} JSON is invalid") from exc
            if type(row) is not dict or line != _canonical(row):
                raise ValueError(f"{name} row is not canonical")
            rows.append(row)
        return rows
