from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.corpus.packing import pack_training_shards
from bcnlt.languages import CANONICAL_LANGUAGE_IDS


_CATEGORIES = (
    "ordinary_instruction",
    "raw_code",
    "semantic_to_code",
    "structural",
    "unicode_roundtrip",
)


def build_replay_records(count: int, *, seed: int) -> tuple[CurriculumRecord, ...]:
    if count <= 0:
        raise ValueError("control replay count must be positive")
    randomizer = random.Random(seed)
    categories = list(_CATEGORIES)
    randomizer.shuffle(categories)
    records = []
    for index in range(count):
        category = categories[index % len(categories)]
        prompt_body, target, target_mode, family, language = _example(category, index)
        prompt = f"Language: {language}\n{prompt_body}"
        content_digest = hashlib.sha256(
            json.dumps(
                [category, prompt, target, target_mode],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        provenance = {
            "canonical_language": language,
            "content_sha256": content_digest,
            "registry_entry_sha256": hashlib.sha256(
                f"control-replay-registry:{language}".encode()
            ).hexdigest(),
            "parser_bundle_sha256": hashlib.sha256(
                f"control-replay-parser:{language}".encode()
            ).hexdigest(),
            "grammar_revision": "control-replay-v1",
            "semantic_spec_sha256": hashlib.sha256(b"bcnlt-control-replay-v1").hexdigest(),
            "extraction_rule_version": "control-replay-v1",
            "replay_category": category,
            "generation_seed": str(seed),
        }
        record_id = hashlib.sha256(
            json.dumps(
                {
                    "index": index,
                    "prompt": prompt,
                    "target": target,
                    "provenance": provenance,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            CurriculumRecord(
                record_id=record_id,
                family=family,
                template_id=f"control-replay-{category}-v1",
                source_document_ids=(f"synthetic-control-{category}-{index:04d}",),
                prompt_mode="text",
                target_mode=target_mode,
                prompt=prompt,
                target=target,
                provenance=tuple(sorted(provenance.items())),
                split="train",
                language=language,
            )
        )
    return tuple(records)


def _example(category: str, index: int) -> tuple[str, str, str, str, str]:
    if category == "ordinary_instruction":
        responses = ("ready", "confirmed", "done", "yes", "no", "okay", "accepted", "complete")
        response = responses[index % len(responses)]
        templates = (
            "Answer using exactly one word: {response}.",
            "Respond with only the word {response}.",
            "Your entire reply must be this one word: {response}.",
            "Give the single-word response {response} and nothing else.",
        )
        prompt = templates[(index // len(responses)) % len(templates)].format(response=response)
        language = CANONICAL_LANGUAGE_IDS[index % len(CANONICAL_LANGUAGE_IDS)]
        return prompt, response, "text", "mixed_instruction", language

    if category == "structural":
        identity = f"module::control_{index:04d}"
        prompt = (
            f"A BCNLT REFERENCE record has defines={identity}. "
            "Return only its exact definition identity."
        )
        language = CANONICAL_LANGUAGE_IDS[index % len(CANONICAL_LANGUAGE_IDS)]
        return prompt, identity, "text", "structural_reasoning", language

    if category == "semantic_to_code":
        variants = (
            ("subtract", "a and b", "the operator a - b", "def subtract(a, b):\n    return a - b\n"),
            ("multiply", "left and right", "the operator left * right", "def multiply(left, right):\n    return left * right\n"),
            ("maximum", "a and b", "a if a >= b else b", "def maximum(a, b):\n    return a if a >= b else b\n"),
            ("minimum", "a and b", "a if a <= b else b", "def minimum(a, b):\n    return a if a <= b else b\n"),
            ("square", "value", "the operator value * value", "def square(value):\n    return value * value\n"),
            ("negate", "value", "the unary operator -value", "def negate(value):\n    return -value\n"),
            ("increment", "value", "the operator value + 1", "def increment(value):\n    return value + 1\n"),
            ("decrement", "value", "the operator value - 1", "def decrement(value):\n    return value - 1\n"),
        )
        name, parameters, expression, target = variants[index % len(variants)]
        prompt = (
            f"Reconstruct parseable Python for a function named {name} with parameters "
            f"{parameters} whose RETURN record contains {expression}."
        )
        return prompt, target, "raw_code", "semantic_to_code", "python"

    if category == "raw_code":
        variable = ("values", "numbers", "samples", "levels", "points", "counts")[index % 6]
        start = index % 17
        target = f"{variable} = [{start}, {start + 1}, {start + 2}]\n"
        prompt = f"Return exactly this Python assignment with a final newline: {target.rstrip()}"
        return prompt, target, "raw_code", "mixed_instruction", "python"

    if category == "unicode_roundtrip":
        symbols = ("π", "雪", "é", "Ж", "Ω", "猫", "δ", "ñ")
        symbol = symbols[index % len(symbols)]
        variable = f"symbol_{index:04d}"
        target = f'{variable} = "{symbol}"\n'
        prompt = f"Return exactly this one Python line, including the final newline: {target.rstrip()}"
        return prompt, target, "raw_code", "mixed_instruction", "python"

    raise ValueError(f"unsupported control replay category: {category}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a bounded BCNLT control-replay dataset")
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")

    from bcnlt.tokenizer import BCNLTTokenizer

    records = build_replay_records(args.count, seed=args.seed)
    output.mkdir(parents=True)
    curriculum = output / "curriculum.jsonl"
    curriculum.write_text(
        "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    tokenizer = BCNLTTokenizer.from_pretrained(
        args.tokenizer_artifact, args.base_tokenizer
    )
    packed = pack_training_shards(records, tokenizer, output / "packed")
    categories = Counter(dict(item.provenance)["replay_category"] for item in records)
    manifest = {
        "schema_version": "bcnlt-control-replay-v1",
        "seed": args.seed,
        "record_count": len(records),
        "category_counts": dict(sorted(categories.items())),
        "curriculum_sha256": hashlib.sha256(curriculum.read_bytes()).hexdigest(),
        "packed_dataset_manifest_sha256": packed.dataset_manifest_sha256,
        "frozen_evaluation_prompts_included": False,
    }
    (output / "replay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
