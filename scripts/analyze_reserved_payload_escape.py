"""Prove a lossless reserved-literal payload escape without vocabulary changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.tokenizer import NativeTokenizerBinding

DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "artifacts"
    / "source-catalogs"
    / "legacy-current-source-v1"
    / "corpus.sqlite"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "audits" / "reserved-payload-escape-v2.json"
STANDARD_SPECIALS = ("<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    binding = NativeTokenizerBinding.load()
    reserved_literals = tuple(
        sorted((*STANDARD_SPECIALS, *binding.structural_token_ids), key=len, reverse=True)
    )
    reserved_ids = frozenset(range(77))
    failures: list[dict[str, object]] = []
    cases = 0
    texts = tuple(
        text
        for literal in reserved_literals
        for text in (
            literal,
            f"before {literal} after",
            literal + literal,
            f"\n{literal}\n",
        )
    )
    scalar_rows = [binding.encode_payload(text) for text in texts]
    batch_rows = binding.encode_payload_batch(texts)
    for text, ids, batch_ids in zip(texts, scalar_rows, batch_rows, strict=True):
        cases += 1
        decoded = binding.decode(ids)
        if (
            decoded != text
            or ids != batch_ids
            or any(token_id in reserved_ids for token_id in ids)
        ):
            failures.append(
                {"text": text, "decoded": decoded, "ids": ids, "batch_ids": batch_ids}
            )

    catalog_sha256 = hashlib.sha256(arguments.catalog.read_bytes()).hexdigest()
    connection = sqlite3.connect(arguments.catalog)
    try:
        # Only the number of parameter placeholders is generated; all values remain bound.
        collision_rows = connection.execute(
            "SELECT COUNT(*) FROM documents WHERE "  # nosec B608
            + " OR ".join("instr(content, ?) > 0" for _ in reserved_literals),
            reserved_literals,
        ).fetchone()[0]
    finally:
        connection.close()

    result = {
        "schema_version": "bosscrafts-reserved-payload-escape-validation-v2",
        "status": "implemented_binding_validation",
        "binding_sha256": binding.binding_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "vocabulary_changed": False,
        "reserved_literal_count": len(reserved_literals),
        "test_case_count": cases,
        "roundtrip_or_control_id_failures": len(failures),
        "scalar_batch_mismatches": sum(
            scalar != batch for scalar, batch in zip(scalar_rows, batch_rows, strict=True)
        ),
        "failure_examples": failures[:10],
        "catalog_sha256": catalog_sha256,
        "catalog_collision_documents": int(collision_rows),
        "escape_rule": "split each exact reserved literal after its first character, encode fragments independently, concatenate IDs",
        "binding_facade_implemented": True,
        "semantic_surface_literal_ban_removed": True,
        "validation_scope": "all 77 reserved literals across scalar and batch payload encoding; raw, semantic and inference paths are covered by the non-Torch regression suite",
    }
    if failures:
        raise RuntimeError("reserved-literal payload escape was not lossless")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
