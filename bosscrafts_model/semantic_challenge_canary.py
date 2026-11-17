"""Select a small, paired, defect-balanced canary without exposing its labels."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from .immutable_artifacts import publish_new_files
from .jsonl import split_jsonl_records
from .semantic_challenge_review import SemanticChallengeReviewBuilder

_SCHEMA = "bosscrafts-semantic-challenge-canary-selection-v1"
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class SemanticChallengeCanarySelectionResult:
    output_root: Path
    item_count: int
    pair_count: int
    selection_sha256: str


class SemanticChallengeCanarySelector:
    """Choose one matched pair per controlled defect, preferring new languages."""

    def materialize(
        self, challenge_root: str | Path, output_root: str | Path
    ) -> SemanticChallengeCanarySelectionResult:
        challenge = Path(challenge_root).resolve()
        output = Path(output_root).resolve()
        selection = self._build(challenge)
        publish_new_files(
            (
                (
                    output / "selection.json",
                    f"{_canonical(selection)}\n".encode("utf-8"),
                ),
            ),
            artifact_label="semantic challenge canary",
        )
        return self._result(output, selection)

    def verify(
        self, challenge_root: str | Path, output_root: str | Path
    ) -> SemanticChallengeCanarySelectionResult:
        challenge = Path(challenge_root).resolve()
        output = Path(output_root).resolve()
        try:
            existing = json.loads(
                (output / "selection.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("semantic challenge canary selection is invalid") from exc
        expected = self._build(challenge)
        if existing != expected:
            raise ValueError("semantic challenge canary selection is stale or tampered")
        return self._result(output, existing)

    def _build(self, challenge_root: Path) -> dict[str, object]:
        manifest = self._load_manifest(challenge_root / "manifest.json")
        pairs_per_language = manifest.get("pairs_per_language")
        if type(pairs_per_language) is not int:
            raise ValueError("semantic challenge pairs-per-language is invalid")
        SemanticChallengeReviewBuilder(
            pairs_per_language=pairs_per_language
        ).verify(challenge_root)
        answer_path = challenge_root / str(manifest.get("answer_key_file"))
        try:
            lines = split_jsonl_records(answer_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError("semantic challenge answer key cannot be read") from exc
        answers: list[dict[str, object]] = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("semantic challenge answer key JSON is invalid") from exc
            if type(row) is not dict or line != _canonical(row):
                raise ValueError("semantic challenge answer key is not canonical")
            answers.append(row)
        queue_path = challenge_root / str(manifest.get("queue_file"))
        try:
            queue_lines = split_jsonl_records(queue_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError("semantic challenge queue cannot be read") from exc
        queue_by_id: dict[str, dict[str, object]] = {}
        for line in queue_lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("semantic challenge queue JSON is invalid") from exc
            if type(row) is not dict or line != _canonical(row):
                raise ValueError("semantic challenge queue is not canonical")
            queue_by_id[str(row.get("review_id"))] = row

        paired: dict[str, list[dict[str, object]]] = {}
        for row in answers:
            paired.setdefault(str(row.get("pair_identity_sha256")), []).append(row)
        candidates: dict[str, list[tuple[int, str, str, tuple[str, str]]]] = {
            defect: [] for defect in _DEFECT_TYPES
        }
        for pair_identity, rows in paired.items():
            if len(rows) != 2:
                raise ValueError("semantic challenge pair inventory is invalid")
            positive = next(
                (row for row in rows if row.get("expected_decision") == "keep"),
                None,
            )
            negative = next(
                (row for row in rows if row.get("expected_decision") == "reject"),
                None,
            )
            if positive is None or negative is None:
                raise ValueError("semantic challenge pair labels are invalid")
            defect = negative.get("defect_type")
            if defect not in candidates:
                raise ValueError("semantic challenge defect type is invalid")
            if positive.get("language") != negative.get("language"):
                raise ValueError("semantic challenge pair language mismatch")
            positive_item = queue_by_id.get(str(positive["review_id"]))
            if positive_item is None:
                raise ValueError("semantic challenge pair is absent from the queue")
            evidence_rank = (
                0
                if defect != "scope_mismatch"
                or self._scope_is_evidenced(positive_item)
                else 1
            )
            candidates[str(defect)].append(
                (
                    evidence_rank,
                    str(negative["language"]),
                    pair_identity,
                    (str(positive["review_id"]), str(negative["review_id"])),
                )
            )
        for values in candidates.values():
            values.sort()

        used_languages: set[str] = set()
        selected_pairs: list[str] = []
        selected_ids: list[str] = []
        selected_languages: list[str] = []
        for defect in _DEFECT_TYPES:
            values = candidates[defect]
            if not values:
                raise ValueError(f"semantic challenge lacks {defect} controls")
            choice = next(
                (value for value in values if value[1] not in used_languages),
                values[0],
            )
            _evidence_rank, language, pair_identity, review_ids = choice
            used_languages.add(language)
            selected_languages.append(language)
            selected_pairs.append(pair_identity)
            selected_ids.extend(review_ids)
        selected_ids.sort()
        unsigned = {
            "schema_version": _SCHEMA,
            "challenge_manifest_sha256": manifest["manifest_sha256"],
            "challenge_queue_sha256": manifest["queue_sha256"],
            "challenge_answer_key_sha256": manifest["answer_key_sha256"],
            "selection_policy": (
                "one_matched_pair_per_defect_prefer_distinct_language_then_sha256"
            ),
            "item_count": len(selected_ids),
            "pair_count": len(selected_pairs),
            "represented_languages": sorted(set(selected_languages)),
            "selected_pair_identity_sha256": sorted(selected_pairs),
            "selected_review_ids": selected_ids,
        }
        return {**unsigned, "selection_sha256": _sha256_text(_canonical(unsigned))}

    @staticmethod
    def _scope_is_evidenced(item: dict[str, object]) -> bool:
        record = item.get("canonical_record")
        excerpt = item.get("source_excerpt")
        if type(record) is not dict or type(excerpt) is not str:
            return False
        scope = record.get("scope")
        if type(scope) is not str:
            return False
        parts = [
            part
            for part in scope.split(".")
            if part not in {"module", "compound_statement", "statement_block", "block"}
        ]
        return any(part and part in excerpt for part in parts)

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, object]:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("semantic challenge manifest is invalid") from exc
        if type(manifest) is not dict:
            raise ValueError("semantic challenge manifest must be an object")
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if claimed != _sha256_text(_canonical(unsigned)):
            raise ValueError("semantic challenge manifest SHA-256 mismatch")
        return manifest

    @staticmethod
    def _result(
        output: Path, selection: dict[str, object]
    ) -> SemanticChallengeCanarySelectionResult:
        item_count = selection.get("item_count")
        pair_count = selection.get("pair_count")
        selection_sha256 = selection.get("selection_sha256")
        if (
            type(item_count) is not int
            or item_count < 0
            or type(pair_count) is not int
            or pair_count < 0
            or item_count != pair_count * 2
            or not _is_sha256(selection_sha256)
        ):
            raise ValueError("semantic challenge canary result is invalid")
        return SemanticChallengeCanarySelectionResult(
            output_root=output,
            item_count=item_count,
            pair_count=pair_count,
            selection_sha256=selection_sha256,
        )

