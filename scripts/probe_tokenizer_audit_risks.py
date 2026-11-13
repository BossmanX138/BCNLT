"""Execute bounded probes for tokenizer audit risks without modifying artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import MutableMapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_TOKENIZERS_SITE = PROJECT_ROOT / "vendor" / "tokenizers_runtime_0_23"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PINNED_TOKENIZERS_SITE))

from bosscrafts_model.bcnlt_adapter import BCNLTMultilingualOutputAdapter
from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    BCNLTSourcePipeline,
)
from bosscrafts_model.inference_protocol import (
    build_inference_prompt,
    response_protocol_complete,
)
from bosscrafts_model.jsonl import split_jsonl_records
from bosscrafts_model.semantic import NativeSemanticRecordCodec, SemanticRecord
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binding = NativeTokenizerBinding.load()
    stream_end = "<|bcnlt:stream:end|>"
    source = f"x{stream_end}y"
    token_ids = binding.encode(source)
    try:
        binding.encode_payload(source)
        payload_rejected = False
    except ValueError:
        payload_rejected = True
    stream_end_id = binding.structural_token_ids[stream_end]
    results: dict[str, object] = {
        "raw_literal": {
            "source": source,
            "token_ids": token_ids,
            "stream_end_id": stream_end_id,
            "contains_control_id": stream_end_id in token_ids,
            "payload_api_rejected": payload_rejected,
            "roundtrip_exact": binding.decode(token_ids) == source,
        }
    }

    special_literals = ("<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>")
    results["standard_special_literals"] = {
        token: binding.encode(f"left{token}right") for token in special_literals
    }
    semantic_codec = NativeSemanticRecordCodec(binding)
    try:
        semantic_codec.encode(
            SemanticRecord(event="RETURN", surface="return <|eos|>", confidence_band="high")
        )
        semantic_injection_rejected = False
    except ValueError:
        semantic_injection_rejected = True
    results["semantic_surface_standard_special_injection"] = {
        "eos_id": 2,
        "rejected": semantic_injection_rejected,
    }
    ids = binding.structural_token_ids
    malformed_semantic = (
        "<|bcnlt:stream:begin|><|bcnlt:mode:semantic_code|>"
        "<|bcnlt:record:begin|><|bcnlt:event:return|><|bcnlt:record:end|>"
        "<|bcnlt:stream:end|>"
    )
    try:
        build_inference_prompt(binding, task="semantic_to_code", prompt=malformed_semantic)
        malformed_prompt_accepted = True
    except ValueError:
        malformed_prompt_accepted = False
    results["malformed_semantic_prompt_accepted"] = malformed_prompt_accepted
    results["malformed_semantic_response_marked_complete"] = response_protocol_complete(
        binding,
        task="code_to_semantic",
        generated_ids=[
            ids["<|bcnlt:stream:begin|>"],
            ids["<|bcnlt:mode:semantic_code|>"],
            ids["<|bcnlt:record:begin|>"],
            ids["<|bcnlt:event:return|>"],
            ids["<|bcnlt:record:end|>"],
            ids["<|bcnlt:stream:end|>"],
            ids["<|bcnlt:task:end|>"],
        ],
    )
    pipeline = BCNLTSourcePipeline(binding)
    prepared = pipeline.prepare_semantic(
        "def add(left, right):\n    return left + right\n",
        language="python",
        path="audit.py",
    )
    if prepared.canonical_jsonl is None:
        raise ValueError("semantic probe omitted canonical records")
    first_record = json.loads(split_jsonl_records(prepared.canonical_jsonl)[0])
    first_record["metadata"]["audit_non_finite"] = float("nan")
    first_record["ir"]["children"] = ["not-an-ir-node", 123]
    permissive_line = json.dumps(
        first_record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    adapter = BCNLTMultilingualOutputAdapter(
        semantic_codec,
        runtime_spec_sha256=CURRENT_RUNTIME_SPEC_SHA256,
    )
    try:
        adapter.parse_jsonl(permissive_line + "\n", pipeline._provenance("python"))
        permissive_accepted = True
    except ValueError:
        permissive_accepted = False
    results["nonstandard_json_and_invalid_ir_children_accepted"] = {
        "accepted": permissive_accepted,
        "contains_nan": "NaN" in permissive_line,
        "ir_children": first_record["ir"]["children"],
    }
    results["binary_text_roundtrip"] = {
        "nul_crlf_tabs": binding.decode(binding.encode("\x00\r\n\t\u0001"))
        == "\x00\r\n\t\u0001",
        "non_bmp": binding.decode(binding.encode("😀𐐷𝄞")) == "😀𐐷𝄞",
    }
    unicode_failures: list[dict[str, object]] = []
    checked_scalars = 0
    chunk: list[str] = []
    chunk_start = 0
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        if not chunk:
            chunk_start = codepoint
        chunk.append(chr(codepoint))
        if len(chunk) == 4096 or codepoint == 0x10FFFF:
            text = "".join(chunk)
            decoded = binding.decode(binding.encode(text))
            if decoded != text:
                mismatch = next(
                    (
                        index
                        for index, pair in enumerate(zip(text, decoded, strict=False))
                        if pair[0] != pair[1]
                    ),
                    min(len(text), len(decoded)),
                )
                unicode_failures.append(
                    {
                        "chunk_start": chunk_start,
                        "mismatch_index": mismatch,
                        "input_length": len(text),
                        "decoded_length": len(decoded),
                    }
                )
                if len(unicode_failures) == 5:
                    break
            checked_scalars += len(chunk)
            chunk.clear()
    results["unicode_scalar_roundtrip"] = {
        "checked_scalars": checked_scalars,
        "failure_count": len(unicode_failures),
        "failures": unicode_failures,
    }

    structural_mapping = binding.structural_token_ids
    if isinstance(structural_mapping, MutableMapping):
        try:
            structural_mapping[stream_end] = 0
            structural_mapping_mutable = True
        except TypeError:
            structural_mapping_mutable = False
    else:
        structural_mapping_mutable = False
    results["structural_mapping_mutable"] = structural_mapping_mutable
    results["tokenizer_object_mutable"] = hasattr(
        binding._frozen_tokenizer, "add_tokens"
    )
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
