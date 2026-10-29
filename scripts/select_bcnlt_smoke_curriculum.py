from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path


CORE_FAMILIES = (
    "code_to_semantic",
    "mixed_instruction",
    "semantic_continuation",
    "semantic_to_code",
    "structural_reasoning",
)


def select_smoke_rows(
    lines: Iterable[str],
    *,
    families: Sequence[str] = CORE_FAMILIES,
    limit: int = 200,
) -> tuple[list[str], dict[str, int]]:
    if limit <= 0:
        raise ValueError("smoke family limit must be positive")
    ordered_families = tuple(families)
    if not ordered_families or len(set(ordered_families)) != len(ordered_families):
        raise ValueError("smoke families must be unique and non-empty")
    counts = {family: 0 for family in ordered_families}
    selected = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid curriculum JSON on line {line_number}: {exc}") from exc
        family = row.get("family")
        if family in counts and counts[family] < limit:
            selected.append(line if line.endswith("\n") else line + "\n")
            counts[family] += 1
    for family, count in counts.items():
        if count != limit:
            raise ValueError(f"family {family} has {count} of {limit} required rows")
    return selected, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a deterministic balanced BCNLT smoke curriculum"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--per-family-limit", type=int, default=200)
    args = parser.parse_args(argv)

    source = Path(args.source)
    output = Path(args.output)
    manifest = Path(args.manifest)
    if output.exists() or manifest.exists():
        parser.error("smoke output and manifest paths must not already exist")
    source_bytes = source.read_bytes()
    lines = source_bytes.decode("utf-8").splitlines(keepends=True)
    selected, counts = select_smoke_rows(lines, limit=args.per_family_limit)
    output_bytes = "".join(selected).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(output_bytes)
    payload = {
        "schema_version": "bcnlt-smoke-selection-v1",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "per_family_limit": args.per_family_limit,
        "selected_counts": dict(sorted(counts.items())),
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
