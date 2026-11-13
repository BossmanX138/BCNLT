"""Freeze V10 sources against the post-audit semantic qualification lineage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.training_corpus_freeze import TrainingCorpusFreezer


DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "artifacts"
    / "source-catalogs"
    / "legacy-current-source-v1"
    / "corpus.sqlite"
)
DEFAULT_HELDOUT = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-current-heldout-admission-v1"
)
DEFAULT_QUALIFICATION = (
    PROJECT_ROOT
    / "artifacts"
    / "reviews"
    / "bcnlt-v10-semantic-audit-qualification-v2-code-audit"
    / "qualification.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v10-training-source-freeze-v2-code-audit"
)
CATALOG_SHA256 = "286d92f79f84b75049b845be0270e4308f32056421c8dea92d82d41b4e4003e2"
HELDOUT_MANIFEST_FILE_SHA256 = (
    "82410c866af775a68884480ddc7dc44debe2878463863aaf7f15bf8ed82a1a44"
)
QUALIFICATION_FILE_SHA256 = (
    "94ff8291f7ffe109b306e7b987643505ed1c3ac08184c5e78f6c94c3933c7568"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--qualification", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = TrainingCorpusFreezer().materialize(
        arguments.catalog,
        arguments.heldout,
        arguments.qualification,
        arguments.output,
        expected_catalog_sha256=CATALOG_SHA256,
        expected_heldout_manifest_file_sha256=HELDOUT_MANIFEST_FILE_SHA256,
        expected_qualification_file_sha256=QUALIFICATION_FILE_SHA256,
    )
    print(
        json.dumps(
            {
                "documents_sha256": result.documents_sha256,
                "manifest_sha256": result.manifest_sha256,
                "output_root": str(result.output_root),
                "reserved_heldout_document_count": (
                    result.reserved_heldout_document_count
                ),
                "source_document_count": result.source_document_count,
                "training_document_count": result.training_document_count,
                "training_ready": result.training_ready,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
