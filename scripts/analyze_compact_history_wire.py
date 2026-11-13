"""Measure a lossless conditional-history semantic-wire projection."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.jsonl import split_jsonl_records
from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_ADMISSION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v6-65536-heldout-admission-v2"
    / "admitted.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "audits" / "compact-history-wire-v6-final-v1.json"
MAX_HISTORY_SURFACE_CHARS = 512
MAX_HISTORY_ITEMS = 8


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    binding = NativeTokenizerBinding.load()
    admission_bytes = arguments.admission.read_bytes()
    documents = 0
    records = 0
    derivable = 0
    overrides = 0
    original_semantic_tokens = 0
    saved_tokens = 0
    reconstruction_failures = 0
    exception_documents: set[str] = set()

    for line in split_jsonl_records(admission_bytes.decode("utf-8")):
        document = json.loads(line)
        documents += 1
        semantic_token_ids = document["semantic_token_ids"]
        if type(semantic_token_ids) is not list or any(
            type(token_id) is not int for token_id in semantic_token_ids
        ):
            raise ValueError("held-out semantic_token_ids are invalid")
        original_semantic_tokens += len(semantic_token_ids)
        prior_surfaces: list[str] = []
        for canonical_line in split_jsonl_records(document["canonical_jsonl"]):
            record = json.loads(canonical_line)
            records += 1
            actual = record["recent_history"]
            derived = [
                surface[:MAX_HISTORY_SURFACE_CHARS]
                for surface in prior_surfaces[-MAX_HISTORY_ITEMS:]
            ]
            compact_history: list[str] | None
            if actual == derived:
                derivable += 1
                compact_history = None
                original_payload_tokens = len(binding.encode_payload(_canonical(actual)))
                saved_tokens += original_payload_tokens - 1
            else:
                overrides += 1
                exception_documents.add(str(document["document_id"]))
                compact_history = actual

            reconstructed = derived if compact_history is None else compact_history
            rebuilt = dict(record)
            rebuilt["recent_history"] = reconstructed
            if _canonical(rebuilt) != canonical_line:
                reconstruction_failures += 1
            prior_surfaces.append(str(record["surface"]))

    projected_tokens = original_semantic_tokens - saved_tokens
    result = {
        "schema_version": "bosscrafts-compact-history-wire-analysis-v1",
        "status": "analysis_only_not_promoted",
        "admission_file": str(arguments.admission.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "admission_sha256": hashlib.sha256(admission_bytes).hexdigest(),
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "documents": documents,
        "records": records,
        "derivable_history_records": derivable,
        "override_history_records": overrides,
        "override_documents": len(exception_documents),
        "derivable_percent": derivable * 100.0 / records,
        "reconstruction_failures": reconstruction_failures,
        "original_semantic_tokens": original_semantic_tokens,
        "projected_semantic_tokens": projected_tokens,
        "saved_tokens": saved_tokens,
        "reduction_percent": saved_tokens * 100.0 / original_semantic_tokens,
        "projection": {
            "derive_when": "recent_history equals the previous eight canonical surfaces truncated to 512 Unicode code points",
            "derived_marker": "one frozen value:null token inside the existing recent_history field envelope",
            "override_when": "history cannot be reconstructed exactly",
            "override_payload": "the original canonical recent_history JSON payload",
        },
    }
    if reconstruction_failures:
        raise RuntimeError("conditional-history projection was not lossless")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(_canonical(result) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
