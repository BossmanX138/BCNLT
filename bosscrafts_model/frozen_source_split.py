"""Verified access to a non-training split of the immutable V10 source freeze."""
from __future__ import annotations

from pathlib import Path

from .current_corpus import SQLiteSourceCatalog
from .training_corpus_freeze import VerifiedTrainingSourceFreeze


class VerifiedFrozenSourceSplit(VerifiedTrainingSourceFreeze):
    """Expose one exact held-out split after the full freeze is reverified."""

    _ALLOWED_SPLITS = {"validation", "test", "reserved_heldout"}

    def __init__(
        self,
        freeze_root: str | Path,
        catalog_path: str | Path,
        *,
        split: str,
        expected_manifest_sha256: str,
    ) -> None:
        if split not in self._ALLOWED_SPLITS:
            raise ValueError(
                "frozen evaluation split must be validation, test or reserved_heldout"
            )
        super().__init__(
            freeze_root,
            catalog_path,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        self.split = split
        self._split_ids = {
            document_id
            for document_id, row in self._records.items()
            if row["split"] == split
        }
        expected_count = int(self._manifest["split_counts"][split])
        if len(self._split_ids) != expected_count or expected_count < 1:
            raise ValueError("frozen evaluation split inventory is invalid")

    @property
    def document_count(self) -> int:
        return len(self._split_ids)

    def iter_candidates(self):
        """Yield only exact members of the selected, verified frozen split."""
        seen: set[str] = set()
        catalog = SQLiteSourceCatalog(
            self.catalog_path, expected_sha256=self.source_catalog_sha256
        )
        for candidate in catalog.iter_candidates(limit_per_language=None):
            if candidate.document_id not in self._split_ids:
                continue
            record = self._records[candidate.document_id]
            for record_field, candidate_field in self._CANDIDATE_FIELDS.items():
                if record[record_field] != getattr(candidate, candidate_field):
                    raise ValueError(
                        "frozen evaluation candidate metadata mismatch: "
                        f"{candidate.document_id}"
                    )
            seen.add(candidate.document_id)
            yield candidate
        if seen != self._split_ids:
            raise ValueError("frozen evaluation candidate inventory is incomplete")
