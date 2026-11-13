"""Measure a lossless compact semantic wire while preserving field labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.tokenizer import NativeTokenizerBinding


DEFAULT_ADMISSION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v6-65536-heldout-admission-v2"
    / "admitted.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "audits" / "compact-semantic-wire-v6-final-v2.json"
MAX_HISTORY_SURFACE_CHARS = 512
MAX_HISTORY_ITEMS = 8
FIELD_COUNT = 11
BATCH_SIZE = 2048
INVERSE_BAND = {"high": "low", "medium": "medium", "low": "high"}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compact_ir(node: object) -> object:
    if type(node) is not dict:
        raise ValueError("IR node must be an object")
    return [
        node["kind"],
        node["name"],
        [_compact_ir(child) for child in node["children"]],
        node["metadata"],
    ]


def _expand_ir(compact: object) -> object:
    if type(compact) is not list or len(compact) != 4:
        raise ValueError("compact IR node must have four positions")
    return {
        "kind": compact[0],
        "name": compact[1],
        "value": None,
        "children": [_expand_ir(child) for child in compact[2]],
        "metadata": compact[3],
    }


def _compact_metadata(metadata: object) -> object:
    if type(metadata) is not dict:
        raise ValueError("metadata must be an object")
    optional = {
        key: value
        for key, value in metadata.items()
        if key not in {"knowledge_state", "uncertainty_signals", "uyr_rounds"}
    }
    return [
        metadata["knowledge_state"],
        metadata["uncertainty_signals"],
        metadata["uyr_rounds"],
        optional,
    ]


def _expand_metadata(compact: object) -> object:
    if type(compact) is not list or len(compact) != 4 or type(compact[3]) is not dict:
        raise ValueError("compact metadata must have four positions")
    return {
        "knowledge_state": compact[0],
        "uncertainty_signals": compact[1],
        "uyr_rounds": compact[2],
        **compact[3],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    binding = NativeTokenizerBinding.load()
    admission_hash = hashlib.sha256()
    documents = 0
    records = 0
    original_semantic_tokens = 0
    history_savings = 0
    history_overrides = 0
    reconstruction_failures = 0
    uncertainty_overrides = 0
    ir_original: list[str] = []
    ir_compact: list[str] = []
    metadata_original: list[str] = []
    metadata_compact: list[str] = []
    ir_savings = 0
    metadata_savings = 0

    def flush_payloads() -> None:
        nonlocal ir_savings, metadata_savings
        if not ir_original:
            return
        ir_original_ids = binding.encode_batch(ir_original)
        ir_compact_ids = binding.encode_batch(ir_compact)
        metadata_original_ids = binding.encode_batch(metadata_original)
        metadata_compact_ids = binding.encode_batch(metadata_compact)
        ir_savings += sum(
            len(original) - len(compact)
            for original, compact in zip(ir_original_ids, ir_compact_ids, strict=True)
        )
        metadata_savings += sum(
            len(original) - len(compact)
            for original, compact in zip(
                metadata_original_ids, metadata_compact_ids, strict=True
            )
        )
        ir_original.clear()
        ir_compact.clear()
        metadata_original.clear()
        metadata_compact.clear()

    with arguments.admission.open("rb") as stream:
        for raw_line in stream:
            admission_hash.update(raw_line)
            document = json.loads(raw_line)
            documents += 1
            semantic_token_ids = document["semantic_token_ids"]
            if type(semantic_token_ids) is not list:
                raise ValueError("held-out semantic_token_ids are invalid")
            original_semantic_tokens += len(semantic_token_ids)
            prior_surfaces: list[str] = []
            for canonical_line in document["canonical_jsonl"].splitlines():
                record = json.loads(canonical_line)
                records += 1

                derived_history = [
                    surface[:MAX_HISTORY_SURFACE_CHARS]
                    for surface in prior_surfaces[-MAX_HISTORY_ITEMS:]
                ]
                if record["recent_history"] == derived_history:
                    history_savings += len(
                        binding.encode_payload(_canonical(record["recent_history"]))
                    ) - 1
                    reconstructed_history = derived_history
                else:
                    history_overrides += 1
                    reconstructed_history = record["recent_history"]

                compact_ir = _compact_ir(record["ir"])
                compact_metadata = _compact_metadata(record["metadata"])
                if _expand_ir(compact_ir) != record["ir"]:
                    reconstruction_failures += 1
                if _expand_metadata(compact_metadata) != record["metadata"]:
                    reconstruction_failures += 1
                if reconstructed_history != record["recent_history"]:
                    reconstruction_failures += 1
                try:
                    reconstructed_uncertainty = INVERSE_BAND[record["confidence_band"]]
                except KeyError as error:
                    raise ValueError("confidence band is invalid") from error
                if reconstructed_uncertainty != record["uncertainty_band"]:
                    uncertainty_overrides += 1

                ir_original.append(_canonical(record["ir"]))
                ir_compact.append(_canonical(compact_ir))
                metadata_original.append(_canonical(record["metadata"]))
                metadata_compact.append(_canonical(compact_metadata))
                if len(ir_original) >= BATCH_SIZE:
                    flush_payloads()
                prior_surfaces.append(str(record["surface"]))
    flush_payloads()

    field_end_savings = records * FIELD_COUNT
    # The uncertainty field-end token is already counted in field_end_savings;
    # deriving the field removes only its begin and value tokens in addition.
    uncertainty_savings = (records - uncertainty_overrides) * 2
    total_savings = (
        history_savings
        + field_end_savings
        + uncertainty_savings
        + ir_savings
        + metadata_savings
    )
    projected_tokens = original_semantic_tokens - total_savings
    result = {
        "schema_version": "bosscrafts-compact-semantic-wire-analysis-v2",
        "status": "analysis_only_not_promoted",
        "admission_file": str(arguments.admission.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "admission_sha256": admission_hash.hexdigest(),
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "documents": documents,
        "records": records,
        "reconstruction_failures": reconstruction_failures,
        "history_override_records": history_overrides,
        "uncertainty_override_records": uncertainty_overrides,
        "original_semantic_tokens": original_semantic_tokens,
        "projected_semantic_tokens": projected_tokens,
        "saved_tokens": total_savings,
        "reduction_percent": total_savings * 100.0 / original_semantic_tokens,
        "savings": {
            "conditional_history": history_savings,
            "redundant_field_end_markers": field_end_savings,
            "derived_uncertainty_field": uncertainty_savings,
            "positional_ir": ir_savings,
            "positional_metadata": metadata_savings,
        },
        "wire_policy": {
            "field_begin_markers": "retained",
            "field_end_markers": "omitted because payload encoding cannot emit reserved field markers",
            "record_markers": "retained",
            "history": "derived marker with exact override",
            "uncertainty": "derived from confidence with exact override",
            "ir": "recursive [kind,name,children,metadata]; value reconstructs as null",
            "metadata": "[knowledge_state,uncertainty_signals,uyr_rounds,optional_fields]",
        },
    }
    if reconstruction_failures:
        raise RuntimeError("compact semantic projection was not reconstructible")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(_canonical(result) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
