"""Execute compact-wire encode/decode over the signed held-out admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.bcnlt_adapter import BCNLTCanonicalRecord
from bosscrafts_model.compact_semantic_wire import CompactSemanticWireV2
from bosscrafts_model.jsonl import split_jsonl_records
from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_ADMISSION = (
    PROJECT_ROOT
    / "artifacts"
    / "corpora"
    / "bcnlt-v6-65536-heldout-admission-v2"
    / "admitted.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "audits" / "compact-semantic-wire-v6-final-v2-validation.json"
DEFAULT_ANALYSIS = PROJECT_ROOT / "artifacts" / "audits" / "compact-semantic-wire-v6-final-v2.json"


def _records(language: str, canonical_jsonl: str) -> tuple[BCNLTCanonicalRecord, ...]:
    records = []
    for line in split_jsonl_records(canonical_jsonl):
        payload = json.loads(line)
        records.append(
            BCNLTCanonicalRecord(
                language=language,
                canonical_json=line,
                payload_sha256=hashlib.sha256(line.encode("utf-8")).hexdigest(),
                event=payload["event"],
                surface=payload["surface"],
                confidence_band=payload["confidence_band"],
                uncertainty_band=payload["uncertainty_band"],
                span=tuple(payload["span"]),
            )
        )
    return tuple(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    arguments = parser.parse_args()

    binding = NativeTokenizerBinding.load()
    codec = CompactSemanticWireV2(binding)
    analysis = json.loads(arguments.analysis.read_text(encoding="utf-8"))
    expected_projected_tokens = analysis.get("projected_semantic_tokens")
    if type(expected_projected_tokens) is not int or expected_projected_tokens < 1:
        raise ValueError("compact semantic wire analysis token count is invalid")
    admission_hash = hashlib.sha256()
    documents = 0
    records = 0
    compact_tokens = 0
    roundtrip_failures = 0
    with arguments.admission.open("rb") as stream:
        for raw_line in stream:
            admission_hash.update(raw_line)
            document = json.loads(raw_line)
            documents += 1
            canonical_jsonl = document["canonical_jsonl"]
            document_records = _records(document["language"], canonical_jsonl)
            records += len(document_records)
            encoded = codec.encode(document_records)
            compact_tokens += len(encoded)
            if codec.decode(encoded) != canonical_jsonl:
                roundtrip_failures += 1

    admission_sha256 = admission_hash.hexdigest()
    analysis_admission_matches = analysis.get("admission_sha256") == admission_sha256
    result = {
        "schema_version": "bosscrafts-compact-semantic-wire-validation-v2",
        "status": "heldout_codec_validation",
        "wire_schema_version": codec.schema_version,
        "admission_file": str(arguments.admission.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "admission_sha256": admission_sha256,
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "documents": documents,
        "records": records,
        "compact_token_count": compact_tokens,
        "analysis_projected_token_count": expected_projected_tokens,
        "analysis_admission_matches": analysis_admission_matches,
        "analysis_and_codec_counts_match": compact_tokens == expected_projected_tokens,
        "roundtrip_failures": roundtrip_failures,
    }
    if (
        roundtrip_failures
        or compact_tokens != expected_projected_tokens
        or not analysis_admission_matches
    ):
        raise RuntimeError("compact semantic wire V2 failed held-out validation")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
